"""E3：全层同时量化，按冻结 C 的 next-token KL 决定主预算的 V 路线。"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.e1 import MODELS, _load_c_items, _render_gsm
from phase1.e2_quant import _config, _decode_step, _prefill_cache
from phase1.runner import cache_lists, decode_step, load_model, prefill


def _kl(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    old = torch.log_softmax(reference.float().reshape(-1), dim=-1)
    new = torch.log_softmax(candidate.float().reshape(-1), dim=-1)
    return float(torch.sum(old.exp() * (old - new)))


def _candidate_logits(
    model: Any, ids: torch.Tensor, token: int,
    config: dict[str, Any], residual_h: int,
) -> tuple[torch.Tensor | None, int]:
    first, keys, values, read_keys, read_values, invalid = _prefill_cache(
        model, ids, config, {}, 1, residual_h
    )
    if first != token:
        raise RuntimeError("候选与高精度的第一个生成 token 不一致")
    if invalid:
        return None, invalid
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured.append(output.detach())

    hook = model.lm_head.register_forward_hook(capture)
    try:
        _next, bad = _decode_step(
            model, keys, values, read_keys, read_values,
            token, int(ids.shape[1]), config, {}, residual_h,
        )
    finally:
        hook.remove()
    if bad:
        return None, bad
    if len(captured) != 1:
        raise RuntimeError("未捕获唯一的 next-token logits")
    return captured[0], 0


def run(model_key: str, commit: str) -> None:
    root = project_root()
    folder = root / "research/phase1/results/e3" / model_key
    allocations = json.loads((folder / "allocations.json").read_text("utf-8"))
    selected_name = allocations["representation"]
    base = _config(selected_name)
    if base["key_mode"] != "per_token" or base["value_mode"] != "per_token":
        raise RuntimeError("当前 E3 门槛实现需要在线逐 token 的 K/V 轴")
    residual_h = int(allocations["residual_bits"])
    names = [
        "p50_sensitivity", "p50_reconstruction", "p50_inverse",
        *(f"p50_random_{seed}" for seed in allocations["random_seeds"]),
    ]
    specs = {name: allocations["allocations"][name] for name in names}
    if len({spec["resident_bytes_32k"] for spec in specs.values()}) != 1:
        raise RuntimeError("主预算全层门槛的完整字节不一致")
    items = _load_c_items(model_key)
    ids_order = [item["id"] for item in items]
    source_index = [hashlib.sha256(
        (item.get("text") or item["question"]).encode("utf-8")
    ).hexdigest() for item in items]
    allocation_digest = hashlib.sha256(json.dumps(
        {name: specs[name]["layer_bits"] for name in names}, sort_keys=True
    ).encode("utf-8")).hexdigest()
    progress = root / "research/.runtime" / f"e3-gate-{model_key}.json"
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ids_order
                or state["source_index"] != source_index
                or state["allocation_digest"] != allocation_digest):
            raise RuntimeError("E3 门槛进度与冻结输入或分配不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ids_order[:len(rows)]:
            raise RuntimeError("E3 门槛进度顺序不一致")
    else:
        rows = []
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    with torch.inference_mode():
        for item in items[len(rows):]:
            if item["kind"] == "math":
                ids = _render_gsm(tokenizer, item["question"], model_key == "llama")
            else:
                ids = tokenizer(
                    item["text"], add_special_tokens=False, return_tensors="pt"
                ).input_ids.to("cuda")
                if ids.shape[1] != item["tokens"]:
                    raise RuntimeError(f"C 检索 token 长度不符：{item['id']}")
            outputs = prefill(model, ids)
            token = int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())
            past_k, past_v = cache_lists(outputs.past_key_values)
            reference, _, _ = decode_step(
                model, past_k, past_v, token, int(ids.shape[1])
            )
            del outputs, past_k, past_v
            kl: dict[str, float | None] = {}
            invalid: dict[str, int] = {}
            for name, spec in specs.items():
                config = {**base, "layer_bits": spec["layer_bits"]}
                candidate, bad = _candidate_logits(
                    model, ids, token, config, residual_h
                )
                kl[name] = None if candidate is None else _kl(reference, candidate)
                invalid[name] = bad
            rows.append({
                "id": item["id"], "kind": item["kind"],
                "input_tokens": int(ids.shape[1]),
                "kl": kl, "invalid_groups": invalid,
            })
            progress.write_text(json.dumps({
                "commit": commit, "ids": ids_order,
                "source_index": source_index,
                "allocation_digest": allocation_digest,
                "rows": rows,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            print(model_key, item["id"], item["kind"], len(rows), flush=True)
    if len(rows) != 96 or sum(row["kind"] == "math" for row in rows) != 64:
        raise RuntimeError("E3 门槛没有覆盖全部冻结 C")
    means: dict[str, dict[str, float | None]] = {}
    for kind in ("math", "retrieval"):
        subset = [row for row in rows if row["kind"] == kind]
        means[kind] = {}
        for name in names:
            values = [row["kl"][name] for row in subset]
            means[kind][name] = (
                sum(values) / len(values) if all(value is not None for value in values)
                else None
            )
    sensitivity = means["retrieval"]["p50_sensitivity"]
    random_kl = [
        means["retrieval"][f"p50_random_{seed}"]
        for seed in allocations["random_seeds"]
    ]
    gate_passes = (
        sensitivity is not None and all(
            value is not None and sensitivity < value for value in random_kl
        )
    )
    selected_nonrandom = (
        "p50_sensitivity" if gate_passes else "p50_reconstruction"
    )
    payload = {
        "protocol": "phase1-protocol v7", "commit": commit,
        "model": model_key, "allocation_commit": allocations["commit"],
        "allocation_digest": allocation_digest,
        "counts": {"math": 64, "retrieval": 32},
        "mean_next_token_kl": means,
        "sensitivity_strictly_below_all_random": gate_passes,
        "selected_nonrandom_for_v": [selected_nonrandom, "p50_inverse"],
        "random_for_v": [
            f"p50_random_{seed}" for seed in allocations["random_seeds"]
        ],
        "rows": rows,
    }
    (folder / "gate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, "gate", gate_passes, selected_nonrandom, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
