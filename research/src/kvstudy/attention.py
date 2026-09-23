"""decode 使用的 FP32 注意力。query 长度为 1 时不使用 is_causal。"""

from __future__ import annotations

import math

import torch


def repeat_kv(values: torch.Tensor, repeats: int) -> torch.Tensor:
    """沿 head 维复制 KV，使它和 query head 对齐。values 形状为 [heads, seq, dim]。"""
    if repeats == 1:
        return values
    return values.repeat_interleave(repeats, dim=0)


def full_history_mask(history: int, *, device: torch.device | None = None) -> torch.Tensor:
    """单个 query 看见全部 history 个位置。返回 bool，形状 [history]。"""
    if history < 0:
        raise ValueError("历史长度不能为负")
    return torch.ones(history, dtype=torch.bool, device=device)


def square_causal_mask(length: int) -> torch.Tensor:
    """方形因果掩码。query 长度不为 length 时不能拿来代替 full_history_mask。"""
    if length < 0:
        raise ValueError("长度不能为负")
    positions = torch.arange(length)
    return positions.unsqueeze(0) <= positions.unsqueeze(1)


def attention_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算单个 query 的注意力。

    query 为 [query_heads, dim]，key 和 value 为 [kv_heads, seq, dim]，都是 FP32。
    返回注意力输出、缩放后的分数和概率，形状分别为
    [query_heads, dim]、[query_heads, seq]、[query_heads, seq]。
    不使用 is_causal。valid 为 False 的位置在 softmax 前被置为 -inf。
    """
    if query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32:
        raise ValueError("注意力路径必须是 FP32")
    query_heads, dim = query.shape
    kv_heads, seq, key_dim = key.shape
    if key_dim != dim or value.shape != key.shape:
        raise ValueError("Q、K、V 的头维度不一致")
    if query_heads % kv_heads != 0:
        raise ValueError("query head 不能整除 KV head")
    expanded_key = repeat_kv(key, query_heads // kv_heads)
    expanded_value = repeat_kv(value, query_heads // kv_heads)
    scores = torch.matmul(expanded_key, query.unsqueeze(-1)).squeeze(-1)
    scores = scores / math.sqrt(dim)
    if valid is not None:
        if valid.shape != (seq,):
            raise ValueError("掩码长度必须等于历史长度")
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    if seq == 0:
        probs = scores
        output = torch.zeros(query_heads, dim, dtype=torch.float32, device=query.device)
        return output, scores, probs
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs.unsqueeze(1), expanded_value).squeeze(1)
    return output, scores, probs


def relative_output_error(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, bool]:
    """返回 ||Δo||^2/||o||^2。分母为 0 时第二个值为 True，调用方不把它计入层平均。"""
    numerator = torch.sum((reference - candidate) ** 2)
    denominator = torch.sum(reference**2)
    if float(denominator) == 0.0:
        return 0.0, True
    return float(numerator / denominator), False


def centered_logit_rms(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """去掉每个 query head 的公共偏移后，分数差的均方根。"""
    delta = candidate - reference
    delta = delta - delta.mean(dim=-1, keepdim=True)
    return torch.sqrt(torch.mean(delta**2, dim=-1))


def probability_kl(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """每个 query head 的 KL(reference || candidate)，单位为 nat。"""
    ref = reference.clamp_min(1e-30)
    cand = candidate.clamp_min(1e-30)
    return torch.sum(ref * (torch.log(ref) - torch.log(cand)), dim=-1)
