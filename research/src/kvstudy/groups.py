"""把 quant.md 的三种轴用到 [kv_heads, seq, dim] 张量上。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kvstudy.quantize import (
    STATUS_CONSTANT,
    STATUS_INVALID,
    dequantize,
    fit_group_params,
    quantize_with_params,
    saturation_mask,
)


@dataclass(frozen=True)
class QuantStats:
    """一组量化的计数。相对重建误差在分母为 0 或结果非有限时为 None。"""

    invalid_groups: int
    constant_groups: int
    saturated_elements: int
    elements: int
    relative_error: float | None


def fit_per_channel(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """沿 token 维拟合每个 head、每个通道。values 形状为 [heads, seq, dim]。"""
    grouped = values.transpose(1, 2)
    return fit_group_params(grouped, bits)


def quantize_per_token(values: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, QuantStats]:
    """每个 token 沿通道分组。不能整除的尾通道单独成组。"""
    restored, status, saturated = _channel_groups(values, bits, group_size)
    return restored, _stats(values, restored, status, saturated)


def quantize_per_channel(
    values: torch.Tensor,
    bits: int,
    scale: torch.Tensor,
    offset: torch.Tensor,
    status: torch.Tensor,
) -> tuple[torch.Tensor, QuantStats]:
    """使用外部的逐通道参数。参数形状为 [heads, dim]。"""
    heads, seq, dim = values.shape
    grouped = values.unsqueeze(-1)
    scale_b = scale.unsqueeze(1).expand(heads, seq, dim)
    offset_b = offset.unsqueeze(1).expand(heads, seq, dim)
    status_b = status.unsqueeze(1).expand(heads, seq, dim)
    codes = quantize_with_params(grouped, scale_b, offset_b, status_b, bits)
    restored = dequantize(codes, scale_b, offset_b, status_b).squeeze(-1)
    saturated = int(saturation_mask(grouped, scale_b, offset_b, status_b, bits).sum())
    return restored, _stats(values, restored, status_b, saturated)


def quantize_cross_token(
    values: torch.Tensor, bits: int, group_tokens: int
) -> tuple[torch.Tensor, QuantStats]:
    """同一通道上每 group_tokens 个 token 成组。未闭合的尾部保持原值。"""
    heads, seq, dim = values.shape
    closed = (seq // group_tokens) * group_tokens
    pieces: list[torch.Tensor] = []
    statuses: list[torch.Tensor] = []
    saturated = 0
    if closed:
        main = values[:, :closed, :].reshape(heads, closed // group_tokens, group_tokens, dim)
        main = main.permute(0, 3, 1, 2)
        scale, offset, status = fit_group_params(main, bits)
        codes = quantize_with_params(main, scale, offset, status, bits)
        restored = dequantize(codes, scale, offset, status)
        restored = restored.permute(0, 2, 3, 1).reshape(heads, closed, dim)
        pieces.append(restored)
        statuses.append(status.reshape(-1))
        saturated += int(saturation_mask(main, scale, offset, status, bits).sum())
    if closed < seq:
        pieces.append(values[:, closed:, :])
    restored_all = torch.cat(pieces, dim=1) if pieces else values[:, :0, :]
    status_all = torch.cat(statuses) if statuses else values.new_zeros(0, dtype=torch.int8)
    return restored_all, _stats(values, restored_all, status_all, saturated)


def _channel_groups(
    values: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """沿通道切组。尾通道不足一组时单独成组。"""
    heads, seq, dim = values.shape
    full = (dim // group_size) * group_size
    parts: list[torch.Tensor] = []
    statuses: list[torch.Tensor] = []
    saturated = 0
    if full:
        main = values[..., :full].reshape(heads, seq, full // group_size, group_size)
        scale, offset, status = fit_group_params(main, bits)
        codes = quantize_with_params(main, scale, offset, status, bits)
        parts.append(dequantize(codes, scale, offset, status).reshape(heads, seq, full))
        statuses.append(status.reshape(-1))
        saturated += int(saturation_mask(main, scale, offset, status, bits).sum())
    if full < dim:
        tail = values[..., full:].reshape(heads, seq, 1, dim - full)
        scale, offset, status = fit_group_params(tail, bits)
        codes = quantize_with_params(tail, scale, offset, status, bits)
        parts.append(dequantize(codes, scale, offset, status).reshape(heads, seq, dim - full))
        statuses.append(status.reshape(-1))
        saturated += int(saturation_mask(tail, scale, offset, status, bits).sum())
    restored = torch.cat(parts, dim=-1)
    return restored, torch.cat(statuses), saturated


def _stats(
    values: torch.Tensor,
    restored: torch.Tensor,
    status: torch.Tensor,
    saturated: int,
) -> QuantStats:
    invalid = int((status == STATUS_INVALID).sum())
    constant = int((status == STATUS_CONSTANT).sum())
    denominator = torch.sum(values.float() ** 2)
    numerator = torch.sum((restored.float() - values.float()) ** 2)
    if int(invalid) > 0 or not torch.isfinite(numerator) or float(denominator) == 0.0:
        error = None
    else:
        error = float(numerator / denominator)
    return QuantStats(
        invalid_groups=invalid,
        constant_groups=constant,
        saturated_elements=saturated,
        elements=int(values.numel()),
        relative_error=error,
    )
