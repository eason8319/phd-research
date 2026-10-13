"""沿 BF16 token 轨迹测量跨层误差，并重建首次分歧前的历史 KV。"""

from __future__ import annotations

from typing import Any

import torch

from phase1.generation import (
    RESIDUAL_TOKENS,
    _decode_step,
    _layer_params,
    _prefill_cache,
    _quantize,
    _recent,
    _rotate_key,
    _write_residual_token,
)
from phase1.runner import cache_lists, decode_step, prefill


def _relative(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """返回整个层向量的相对平方误差。"""
    old, new = reference.float(), candidate.detach().cpu().float()
    denominator = old.square().sum().item()
    if denominator == 0:
        raise ValueError("参考向量零能量，不能计算相对平方误差")
    return (new - old).square().sum().item() / denominator


class ReferenceObserver:
    """只在 CPU 保留单个生成位置的层输出、当前 KV 和词表 logits。"""

    def __init__(self) -> None:
        self.layers: list[dict[str, torch.Tensor]] = []
        self.logits: torch.Tensor | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        if "logits" in event:
            self.logits = event["logits"].detach().cpu()
        else:
            if event["layer"] != len(self.layers):
                raise ValueError("参考层顺序不符")
            self.layers.append({key: event[key].detach().cpu() for key in (
                "hidden", "key_pre", "key_post", "value",
            )})


class ComparisonObserver:
    """同 token、同层比较输出；logits 采用 KL(BF16 || 候选)。"""

    def __init__(self, reference: ReferenceObserver, *, collect_layers: bool = True) -> None:
        self.reference = reference
        self.collect_layers = collect_layers
        self.layers: list[dict[str, float]] = []
        self.output: dict[str, Any] = {}

    def __call__(self, event: dict[str, Any]) -> None:
        if "logits" not in event:
            if not self.collect_layers:
                return
            index = event["layer"]
            self.layers.append({key: _relative(event[key], self.reference.layers[index][key])
                                for key in ("hidden",)})
            return
        old = self.reference.logits.float().flatten()
        new = event["logits"].detach().cpu().float().flatten()
        old_log, new_log = old.log_softmax(0), new.log_softmax(0)
        best = int(old.argmax())
        candidate = int(new.argmax())
        top = old.topk(2).values
        alternatives = new.clone()
        alternatives[best] = -torch.inf
        self.output = {
            "bf16_argmax": best, "candidate_argmax": candidate,
            "kl": max(0.0, float((old_log.exp() * (old_log - new_log)).sum())),
            "bf16_margin": float(top[0] - top[1]),
            "candidate_reference_margin": float(new[best] - alternatives.max()),
        }

        if self.collect_layers:
            self.output["layers"] = self.layers


def reference_run(model: Any, ids: torch.Tensor, eos: set[int], limit: int,
                  *, consume_cache: bool = False,
                  ) -> tuple[list[int], str, list[ReferenceObserver]]:
    """运行完整 BF16 自由轨迹；只保留后续配对所需的 CPU 中间量。"""
    outputs = prefill(model, ids)
    token = int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())
    keys, values = cache_lists(outputs.past_key_values)
    del outputs
    tokens, observers = [token], []
    while token not in eos and len(tokens) < limit:
        observer = ReferenceObserver()
        logits, keys, values = decode_step(
            model, keys, values, token, ids.shape[1] + len(tokens) - 1, observer=observer,
            consume_cache=consume_cache,
        )
        observers.append(observer)
        token = int(logits.flatten().argmax())
        tokens.append(token)
    return tokens, "eos" if token in eos else "limit", observers


def _rebuild_generated_history(
    model: Any, buffers: tuple[Any, ...], references: list[ReferenceObserver],
    config: dict[str, Any], calibrated: dict[str, Any], h: int, start: int, count: int,
) -> None:
    """从 BF16 原值重建已生成历史，保持同一量化参数、RoPE 顺序和近期窗口。"""
    _, keys, values, read_keys, read_values, _ = buffers
    if count == 0:
        return
    positions = torch.arange(start, start + count, device="cuda").view(1, -1)
    for layer in range(len(keys)):
        source = {name: torch.cat([ref.layers[layer][name] for ref in references[:count]],
                                  dim=2)[0].to("cuda")
                  for name in ("key_pre", "key_post", "value")}
        key, bad_k = _quantize(
            source[f"key_{config['rope']}"], config["k_bits"], config["key_mode"],
            _layer_params(calibrated, "key", layer, config["k_bits"]),
        )
        if config["rope"] == "pre":
            key = _rotate_key(model, key, positions)
        value, bad_v = _quantize(
            source["value"], config["v_bits"], config["value_mode"],
            _layer_params(calibrated, "value", layer, config["v_bits"]),
        )
        bad = bad_k + bad_v
        if h:
            high_k, bad_high_k = _recent(source["key_post"], h)
            high_v, bad_high_v = _recent(source["value"], h)
            bad += bad_high_k + bad_high_v
            for step in range(count):
                part = slice(step, step + 1)
                _write_residual_token(
                    keys[layer], values[layer], read_keys[layer], read_values[layer],
                    key[:, part], value[:, part], high_k[:, part], high_v[:, part], start + step,
                )
        else:
            keys[layer][:, :, start:start + count] = key.unsqueeze(0)
            values[layer][:, :, start:start + count] = value.unsqueeze(0)
        if int(bad):
            raise RuntimeError("历史重建出现无效量化组")


def candidate_run(
    model: Any, ids: torch.Tensor, tokens: list[int], references: list[ReferenceObserver],
    config: dict[str, Any], calibrated: dict[str, Any], h: int, limit: int,
    *, offload_layers: int = 0,
) -> dict[str, Any]:
    """独立维护候选 KV，完整回放，并对首次分歧做一次同格式历史重建。"""
    buffers = _prefill_cache(model, ids, config, calibrated, limit, h,
                             offload_layers=offload_layers)
    first, keys, values, read_keys, read_values, invalid = buffers
    if invalid or first != tokens[0]:
        raise RuntimeError("候选 prefill 无效或首 token 与共同参考不一致")
    rows, first_different = [], None
    for step, reference in enumerate(references):
        observer = ComparisonObserver(reference)
        predicted, invalid = _decode_step(
            model, keys, values, read_keys, read_values, tokens[step], ids.shape[1] + step,
            config, calibrated, h, observer=observer,
        )
        if invalid:
            raise RuntimeError("完整回放出现无效量化组")
        rows.append({"decode_step": step, "prediction_index": step + 1, **observer.output})
        if predicted != tokens[step + 1] and first_different is None:
            first_different = step
    del buffers, keys, values, read_keys, read_values
    torch.cuda.empty_cache()
    reset = None
    if first_different is not None:
        buffers = _prefill_cache(model, ids, config, calibrated, limit, h,
                                 offload_layers=offload_layers)
        if buffers[-1]:
            raise RuntimeError("重建对照 prefill 无效")
        _rebuild_generated_history(
            model, buffers, references, config, calibrated, h, ids.shape[1], first_different,
        )
        observer = ComparisonObserver(references[first_different], collect_layers=False)
        _, invalid = _decode_step(
            model, *buffers[1:5], tokens[first_different], ids.shape[1] + first_different,
            config, calibrated, h, observer=observer,
        )
        if invalid:
            raise RuntimeError("历史重建对照出现无效量化组")
        reset = {"decode_step": first_different, "history_generated_tokens": first_different,
                 **observer.output}
    return {"representation": config["name"], "h": h, "r": RESIDUAL_TOKENS if h else 0,
            "first_divergence_index": first_different + 1 if first_different is not None else None,
            "steps": rows, "history_reset": reset, "invalid_groups": 0}
