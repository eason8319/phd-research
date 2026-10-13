"""固定 BF16 投影及 prefill，分步提升实际输出头与 decode 残差流计算精度。"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from phase1.generation import (
    _layer_bits,
    _layer_params,
    _prefill_cache,
    _quantize,
    _recent,
    _write_residual_token,
)
from phase1.runner import attention_from_batch, modeling_module, project_qkv


def output_head(model: Any, hidden: torch.Tensor, mode: str, weight: torch.Tensor) -> Any:
    """FP32 条件实际使用 FP32 GEMM；当前条件保持原 BF16 矩阵计算。"""
    if mode == "current":
        return model.lm_head(hidden)
    bias = model.lm_head.bias
    return functional.linear(hidden.float(), weight, bias.float() if bias is not None else None)


def step(
    model: Any, state: tuple[Any, ...], token: int, position: int, mode: str,
    weight: torch.Tensor, config: dict[str, Any] | None, params: dict[str, Any], h: int,
) -> torch.Tensor:
    """仅残差条件保留 FP32 hidden/加法；RMSNorm 后在投影入口转回 BF16。"""
    device = model.model.embed_tokens.weight.device
    hidden = model.model.embed_tokens(torch.tensor([[token]], device=device))
    positions = torch.tensor([[position]], device=device)
    cos, sin = model.model.rotary_emb(hidden, positions)
    if mode == "residual_fp32":
        hidden = hidden.float()
    apply_rope = modeling_module(model).apply_rotary_pos_emb
    keys, values, read_keys, read_values = state
    invalid = torch.zeros((), dtype=torch.int64, device=device)
    for index, layer in enumerate(model.model.layers):
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        if mode == "residual_fp32":
            normalized = normalized.to(torch.bfloat16)
        query, key_pre, value = project_qkv(layer.self_attn, normalized)
        query, key_post = apply_rope(query, key_pre, cos, sin)
        if config is None:
            keys[index] = torch.cat([keys[index], key_post], dim=2)
            values[index] = torch.cat([values[index], value], dim=2)
            attended = attention_from_batch(query, keys[index], values[index])
        else:
            k_bits, v_bits = _layer_bits(config, index)
            source = key_pre[0] if config["rope"] == "pre" else key_post[0]
            k_hat, bad_k = _quantize(
                source, k_bits, config["key_mode"], _layer_params(params, "key", index, k_bits)
            )
            if config["rope"] == "pre":
                _, rotated = apply_rope(k_hat.unsqueeze(0), k_hat.unsqueeze(0), cos, sin)
                k_hat = rotated[0]
            v_hat, bad_v = _quantize(
                value[0], v_bits, config["value_mode"], _layer_params(params, "value", index,
                                                                  v_bits)
            )
            invalid += bad_k + bad_v
            if h:
                high_k, bad_high_k = _recent(key_post[0], h)
                high_v, bad_high_v = _recent(value[0], h)
                _write_residual_token(keys[index], values[index], read_keys[index],
                                      read_values[index], k_hat, v_hat, high_k, high_v, position)
                invalid += bad_high_k + bad_high_v
            else:
                keys[index][:, :, position:position + 1] = k_hat.unsqueeze(0)
                values[index][:, :, position:position + 1] = v_hat.unsqueeze(0)
            attended = attention_from_batch(query, read_keys[index][:, :, :position + 1],
                                            read_values[index][:, :, :position + 1])
        projected = layer.self_attn.o_proj(attended)
        hidden = residual + (projected.float() if mode == "residual_fp32" else projected)
        residual = hidden
        normalized = layer.post_attention_layernorm(hidden)
        if mode == "residual_fp32":
            normalized = normalized.to(torch.bfloat16)
        projected = layer.mlp(normalized)
        hidden = residual + (projected.float() if mode == "residual_fp32" else projected)
    if bool(invalid):
        raise RuntimeError("精度诊断存在无效量化组")
    logits = output_head(model, model.model.norm(hidden)[:, -1], mode, weight)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("精度诊断出现非有限 logits")
    return logits.reshape(-1)


def initialize(
    model: Any, source: dict[str, Any], config: dict[str, Any] | None,
    params: dict[str, Any], h: int, limit: int,
) -> tuple[Any, ...]:
    """每次测量从同一源值重新建缓存，随后持续更新。"""
    if config is None:
        return list(source["keys"]), list(source["values"]), None, None
    _, keys, values, read_keys, read_values, bad = _prefill_cache(
        model, source["input_ids"], config, params, limit, h, prefill_source=source,
    )
    if bad:
        raise RuntimeError("精度诊断缓存初始化失败")
    return keys, values, read_keys, read_values


def first_logits(
    model: Any, source: dict[str, Any], mode: str, weight: torch.Tensor,
) -> torch.Tensor:
    """prefill 上游保持原 BF16 源值，只在高精度条件重算实际输出头。"""
    return source["logits"].reshape(-1) if mode == "current" else \
        output_head(model, source["last_hidden"], mode, weight).reshape(-1)


def trace(
    model: Any, source: dict[str, Any], answer: list[int], mode: str,
    weight: torch.Tensor, config: dict[str, Any] | None, params: dict[str, Any], h: int,
) -> np.ndarray:
    """在完整指定路径上先测预测、再喂入当前目标 token，包含终止预测。"""
    state = initialize(model, source, config, params, h, len(answer))
    logits = first_logits(model, source, mode, weight)
    saved = []
    for index, token in enumerate(answer):
        if index:
            logits = step(model, state, answer[index - 1],
                          source["input_ids"].shape[1] + index - 1, mode, weight,
                          config, params, h)
        saved.append(logits.detach().float().cpu().numpy().copy())
    return np.stack(saved)


def generate(
    model: Any, source: dict[str, Any], eos: set[int], limit: int, mode: str,
    weight: torch.Tensor, config: dict[str, Any] | None, params: dict[str, Any], h: int,
) -> list[int]:
    """沿实际 argmax 路径自由生成，无额外 logits 处理器或正确前缀。"""
    state = initialize(model, source, config, params, h, limit)
    produced = [int(first_logits(model, source, mode, weight).argmax())]
    for index in range(limit - 1):
        if produced[-1] in eos:
            break
        logits = step(model, state, produced[-1], source["input_ids"].shape[1] + index,
                      mode, weight, config, params, h)
        produced.append(int(logits.argmax()))
    return produced
