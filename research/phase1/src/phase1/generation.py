"""第一阶段共用的持续量化缓存与真实生成，不安排实验条件。"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import torch

from kvstudy.quantize import STATUS_INVALID, dequantize, fit_group_params, quantize_with_params
from phase1.runner import attention_from_batch, cache_lists, modeling_module, prefill, project_qkv
from phase1.runtime_sources import _protocol_matches, _runtime_path


G_C = 32
MAX_CONTEXT = 32768
RESIDUAL_TOKENS = 128


def _config(name: str) -> dict[str, Any]:
    """从位宽与表示交互的配置名解析 K/V 位宽、量化轴及 RoPE 顺序。"""
    parts = name.split("-")
    if len(parts) not in (3, 4) or parts[0] not in ("kv8", "kv4", "k8v4", "k4v8"):
        raise ValueError(f"未知配置：{name}")
    if parts[1] not in ("per_channel", "per_token") or parts[2] not in ("pre", "post"):
        raise ValueError(f"未知配置：{name}")
    if len(parts) == 4 and parts[3] != "vchannel":
        raise ValueError(f"未知配置：{name}")
    k_bits, v_bits = {
        "kv8": (8, 8), "kv4": (4, 4), "k8v4": (8, 4), "k4v8": (4, 8)
    }[parts[0]]
    return {
        "name": name,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "key_mode": parts[1],
        "rope": parts[2],
        "value_mode": "per_channel" if len(parts) == 4 else "per_token",
    }


def _calibration(
    root: Path, model_key: str, config: dict[str, Any], *, device: str = "cuda",
) -> dict[str, Any]:
    """仅离线轴读取全部 C 拟合的参数，并核对 C 的冻结 ID。"""
    if config["key_mode"] != "per_channel" and config["value_mode"] != "per_channel":
        return {}
    path = _runtime_path(
        root, f"generation-calibration-{model_key}.pt", f"e2-calibration-{model_key}.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    retrieval = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
    expected = [item["id"] for item in gsm["c"]]
    expected.extend(item["id"] for item in retrieval if item["id"].startswith("synth-c-"))
    if (not _protocol_matches(payload["protocol"], "phase1-full-key-retrieval")
            or payload["ids"] != expected):
        raise RuntimeError("全部 C 的离线参数与当前冻结划分不一致")
    result: dict[str, Any] = {"commit": payload["commit"]}
    if config["key_mode"] == "per_channel":
        bits_needed = {config["k_bits"]}
        bits_needed.update(row["k"] for row in config.get("layer_bits", []))
        result["key_by_bits"] = {
            bits: [tuple(t.to(device) for t in row)
                   for row in payload["params"][f"k_{config['rope']}{bits}"]]
            for bits in bits_needed
        }
        result["key"] = result["key_by_bits"][config["k_bits"]]
    if config["value_mode"] == "per_channel":
        value = f"v{config['v_bits']}"
        result["value"] = [tuple(t.to(device) for t in row) for row in payload["params"][value]]
    return result


def _layer_params(
    calibrated: dict[str, Any], side: str, layer: int, bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """混合层按实际位宽读取离线参数；未校准的在线轴返回 None。"""
    by_bits = calibrated.get(f"{side}_by_bits")
    if by_bits is not None:
        return by_bits[bits][layer]
    params = calibrated.get(side)
    return params[layer] if params is not None else None


def _rotate_key(model: Any, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """量化后按模型 RoPE 旋转 [heads, seq, dim] Key。"""
    hidden = torch.zeros(
        1, key.shape[1], model.config.hidden_size, device=key.device, dtype=torch.bfloat16
    )
    cos, sin = model.model.rotary_emb(hidden, positions)
    packed = key.unsqueeze(0)
    _, rotated = modeling_module(model).apply_rotary_pos_emb(packed, packed, cos, sin)
    return rotated[0]


def _quantize(
    values: torch.Tensor,
    bits: int,
    mode: str,
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 FP32 还原值与无效组数，始终从高精度输入量化。"""
    if mode == "per_token":
        heads, seq, dim = values.shape
        if dim % G_C:
            raise RuntimeError("真实生成与误差来源模型的 head_dim 不能被 G_C 整除")
        grouped = values.reshape(heads, seq, dim // G_C, G_C)
        scale, offset, status = fit_group_params(grouped, bits)
        codes = quantize_with_params(grouped, scale, offset, status, bits)
        restored = dequantize(codes, scale, offset, status).reshape_as(values)
    else:
        if params is None:
            raise RuntimeError("逐通道量化缺少全部 C 校准参数")
        heads, seq, dim = values.shape
        scale, offset, status = params
        scale = scale.unsqueeze(1).expand(heads, seq, dim)
        offset = offset.unsqueeze(1).expand(heads, seq, dim)
        status = status.unsqueeze(1).expand(heads, seq, dim)
        grouped = values.unsqueeze(-1)
        codes = quantize_with_params(grouped, scale, offset, status, bits)
        restored = dequantize(codes, scale, offset, status).squeeze(-1)
    return restored, (status == STATUS_INVALID).sum()


def _recent(values: torch.Tensor, h_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """从高精度原值生成最近窗口的 H=8 或 H=16 副本。"""
    if h_bits == 16:
        return values.float(), torch.zeros((), dtype=torch.int64, device=values.device)
    if h_bits == 8:
        return _quantize(values, 8, "per_token", None)
    raise ValueError(f"未知残留位宽：{h_bits}")


def _layer_bits(config: dict[str, Any], index: int) -> tuple[int, int]:
    """静态层分配可按层覆盖 K/V 位宽；真实生成与误差来源保持全层同一配置。"""
    if "layer_bits" not in config:
        return config["k_bits"], config["v_bits"]
    row = config["layer_bits"][index]
    if row["k"] not in (4, 8) or row["v"] not in (4, 8):
        raise ValueError("静态层分配层位宽只允许 INT4 或 INT8")
    return row["k"], row["v"]


def _prefill_cache(
    model: Any,
    input_ids: torch.Tensor,
    config: dict[str, Any],
    calibrated: dict[str, Any],
    limit: int,
    h_bits: int,
    *,
    offload_layers: int = 0,
    prefill_source: dict[str, Any] | None = None,
) -> tuple[
    int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int
]:
    """高精度 prefill 后量化；前 offload_layers 层以同 dtype 暂存 CPU。"""
    if prefill_source is None:
        pre_keys: list[torch.Tensor] = []
        module = modeling_module(model)
        original = module.apply_rotary_pos_emb
        if config["rope"] == "pre":
            def capture(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor,
                        sin: torch.Tensor, *args: object, **kwargs: object) -> Any:
                pre_keys.append(key.detach().cpu() if offload_layers else key.detach())
                return original(query, key, cos, sin, *args, **kwargs)
            module.apply_rotary_pos_emb = capture
        try:
            outputs = prefill(model, input_ids)
        finally:
            module.apply_rotary_pos_emb = original
        first = int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())
        original_k, original_v = cache_lists(outputs.past_key_values)
        del outputs
    else:
        if not torch.equal(input_ids, prefill_source["input_ids"]):
            raise RuntimeError("复用的高精度 prefill 与实际输入不一致")
        first = prefill_source["token"]
        original_k = list(prefill_source["keys"])
        original_v = list(prefill_source["values"])
        pre_keys = list(prefill_source["pre_keys"])
    if config["rope"] == "pre" and len(pre_keys) != len(original_k):
        raise RuntimeError("未捕获所有层的 pre-RoPE Key")
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    read_keys: list[torch.Tensor] = []
    read_values: list[torch.Tensor] = []
    invalid = torch.zeros((), dtype=torch.int64, device=input_ids.device)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
    for index in range(len(original_k)):
        k_bits, v_bits = _layer_bits(config, index)
        source = pre_keys[index][0] if config["rope"] == "pre" else original_k[index][0]
        source = source.to(input_ids.device)
        k_params = _layer_params(calibrated, "key", index, k_bits)
        k_hat, bad_k = _quantize(source, k_bits, config["key_mode"], k_params)
        if config["rope"] == "pre":
            k_hat = _rotate_key(model, k_hat, positions)
        v_params = _layer_params(calibrated, "value", index, v_bits)
        v_hat, bad_v = _quantize(
            original_v[index][0], v_bits, config["value_mode"], v_params
        )
        capacity = int(input_ids.shape[1]) + limit
        key_buffer = torch.empty(
            (1, *k_hat.shape[:1], capacity, k_hat.shape[-1]),
            device=k_hat.device, dtype=torch.float32,
        )
        value_buffer = torch.empty_like(key_buffer)
        key_buffer[:, :, :input_ids.shape[1], :] = k_hat.unsqueeze(0)
        value_buffer[:, :, :input_ids.shape[1], :] = v_hat.unsqueeze(0)
        invalid += bad_k + bad_v
        if h_bits:
            start = max(0, int(input_ids.shape[1]) - RESIDUAL_TOKENS)
            high_k, bad_high_k = _recent(original_k[index][0][:, start:, :], h_bits)
            high_v, bad_high_v = _recent(original_v[index][0][:, start:, :], h_bits)
            ring_shape = (1, k_hat.shape[0], RESIDUAL_TOKENS, k_hat.shape[-1])
            low_key_ring = torch.empty(ring_shape, device=k_hat.device, dtype=torch.float32)
            low_value_ring = torch.empty_like(low_key_ring)
            ring_indices = torch.arange(start, int(input_ids.shape[1]),
                                        device=k_hat.device) % RESIDUAL_TOKENS
            low_key_ring[:, :, ring_indices, :] = k_hat[:, start:, :].unsqueeze(0)
            low_value_ring[:, :, ring_indices, :] = v_hat[:, start:, :].unsqueeze(0)
            key_buffer[:, :, start:input_ids.shape[1], :] = high_k.unsqueeze(0)
            value_buffer[:, :, start:input_ids.shape[1], :] = high_v.unsqueeze(0)
            keys.append(low_key_ring)
            values.append(low_value_ring)
            read_keys.append(key_buffer)
            read_values.append(value_buffer)
            invalid += bad_high_k + bad_high_v
        else:
            keys.append(key_buffer)
            values.append(value_buffer)
            read_keys.append(key_buffer)
            read_values.append(value_buffer)
        original_k[index] = None
        original_v[index] = None
        if pre_keys:
            pre_keys[index] = None
        if index < offload_layers:
            buffers = (keys, values, read_keys, read_values)
            copied = {}
            for group in buffers:
                if id(group[-1]) not in copied:
                    copied[id(group[-1])] = group[-1].cpu()
            for group in buffers:
                group[-1] = copied[id(group[-1])]
    return first, keys, values, read_keys, read_values, int(invalid.item())


def _write_residual_token(
    low_key_ring: torch.Tensor,
    low_value_ring: torch.Tensor,
    read_key: torch.Tensor,
    read_value: torch.Tensor,
    k_hat: torch.Tensor,
    v_hat: torch.Tensor,
    high_k: torch.Tensor,
    high_v: torch.Tensor,
    position: int,
) -> None:
    """先恢复滑出的低位宽值，再写当前高精度副本，避免全历史重复缓存。"""
    if position >= RESIDUAL_TOKENS:
        leaving = position - RESIDUAL_TOKENS
        slot = leaving % RESIDUAL_TOKENS
        read_key[:, :, leaving:leaving + 1, :] = low_key_ring[:, :, slot:slot + 1, :]
        read_value[:, :, leaving:leaving + 1, :] = low_value_ring[:, :, slot:slot + 1, :]
    slot = position % RESIDUAL_TOKENS
    low_key_ring[:, :, slot:slot + 1, :] = k_hat.unsqueeze(0)
    low_value_ring[:, :, slot:slot + 1, :] = v_hat.unsqueeze(0)
    read_key[:, :, position:position + 1, :] = high_k.unsqueeze(0)
    read_value[:, :, position:position + 1, :] = high_v.unsqueeze(0)


def _decode_step(
    model: Any,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    read_keys: list[torch.Tensor],
    read_values: list[torch.Tensor],
    token: int,
    position: int,
    config: dict[str, Any],
    calibrated: dict[str, Any],
    h_bits: int,
    *,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int | None, int]:
    """追加 KV 并按共同 FP32 注意力解码；CPU 缓存原样逐层搬运，不转换精度。"""
    device = model.model.embed_tokens.weight.device
    hidden = model.model.embed_tokens(torch.tensor([[token]], device=device))
    positions = torch.tensor([[position]], device=device)
    cos, sin = model.model.rotary_emb(hidden, positions)
    apply_rope = modeling_module(model).apply_rotary_pos_emb
    invalid = torch.zeros((), dtype=torch.int64, device=device)
    for index, layer in enumerate(model.model.layers):
        buffers = (keys, values, read_keys, read_values)
        originals = [group[index] for group in buffers]
        offloaded = read_keys[index].device.type == "cpu"
        if offloaded:
            resident = {}
            for value in originals:
                if id(value) not in resident:
                    resident[id(value)] = value.to(device)
            for group, original in zip(buffers, originals):
                group[index] = resident[id(original)]
        k_bits, v_bits = _layer_bits(config, index)
        residual = hidden
        query, key_pre, value = project_qkv(layer.self_attn, layer.input_layernorm(hidden))
        query, key_post = apply_rope(query, key_pre, cos, sin)
        source = key_pre[0] if config["rope"] == "pre" else key_post[0]
        k_params = _layer_params(calibrated, "key", index, k_bits)
        k_hat, bad_k = _quantize(source, k_bits, config["key_mode"], k_params)
        if config["rope"] == "pre":
            _, rotated = apply_rope(k_hat.unsqueeze(0), k_hat.unsqueeze(0), cos, sin)
            k_hat = rotated[0]
        v_params = _layer_params(calibrated, "value", index, v_bits)
        v_hat, bad_v = _quantize(value[0], v_bits, config["value_mode"], v_params)
        invalid += bad_k + bad_v
        if h_bits:
            high_k, bad_high_k = _recent(key_post[0], h_bits)
            high_v, bad_high_v = _recent(value[0], h_bits)
            _write_residual_token(
                keys[index], values[index], read_keys[index], read_values[index],
                k_hat, v_hat, high_k, high_v, position,
            )
            invalid += bad_high_k + bad_high_v
        else:
            keys[index][:, :, position:position + 1, :] = k_hat.unsqueeze(0)
            values[index][:, :, position:position + 1, :] = v_hat.unsqueeze(0)
        attended = attention_from_batch(
            query, read_keys[index][:, :, :position + 1, :],
            read_values[index][:, :, :position + 1, :],
        )
        hidden = residual + layer.self_attn.o_proj(attended)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        if observer is not None:
            observer({"layer": index, "attention": attended, "hidden": hidden,
                      "query": query, "key_pre": key_pre,
                      "key_post": key_post, "value": value})
        if offloaded:
            for original in {id(value): value for value in originals}.values():
                original.copy_(resident[id(original)])
            for group, original in zip(buffers, originals):
                group[index] = original
            del resident
    bad = int(invalid.item())
    if bad:
        return None, bad
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, -1, :])
    if observer is not None:
        observer({"logits": logits})
    return int(logits.reshape(-1, logits.shape[-1])[0].argmax()), 0


def _generate(
    model: Any,
    ids: list[int],
    eos: set[int],
    limit: int,
    config: dict[str, Any],
    calibrated: dict[str, Any],
    h_bits: int = 0,
    *,
    offload_layers: int = 0,
) -> tuple[list[int], str, int]:
    """返回生成 token、停止原因和累计无效组数。"""
    tensor = torch.tensor([ids], device="cuda")
    first, keys, values, read_keys, read_values, invalid = _prefill_cache(
        model, tensor, config, calibrated, limit, h_bits, offload_layers=offload_layers
    )
    if invalid:
        return [first], "invalid", invalid
    produced = [first]
    if first in eos:
        return produced, "eos", 0
    token = first
    for step in range(limit - 1):
        next_token, bad = _decode_step(
            model, keys, values, read_keys, read_values,
            token, len(ids) + step, config, calibrated, h_bits,
        )
        invalid += bad
        if next_token is None:
            return produced, "invalid", invalid
        token = next_token
        produced.append(token)
        if token in eos:
            return produced, "eos", invalid
    return produced, "limit", invalid
