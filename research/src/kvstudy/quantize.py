"""均匀非对称分组量化。参数按 quant.md 舍入到 FP16 后再参与计算。"""

from __future__ import annotations

import torch

# 0 普通组，1 常量组，2 无效组。
STATUS_NORMAL = 0
STATUS_CONSTANT = 1
STATUS_INVALID = 2


def round_fp16(values: torch.Tensor, *, toward: str) -> torch.Tensor:
    """把有限值定向舍入到 FP16。toward 为 down（向 -inf）或 up（向 +inf）。"""
    if toward not in ("down", "up"):
        raise ValueError(f"未知舍入方向: {toward}")
    original = values.to(dtype=torch.float32)
    nearest = original.to(dtype=torch.float16)
    nearest_as_f32 = nearest.to(dtype=torch.float32)
    finite = torch.isfinite(original)
    if toward == "down":
        needs_step = finite & (nearest_as_f32 > original)
        destination = torch.full_like(nearest, float("-inf"))
    else:
        needs_step = finite & (nearest_as_f32 < original)
        destination = torch.full_like(nearest, float("inf"))
    stepped = torch.nextafter(nearest, destination)
    return torch.where(needs_step, stepped, nearest)


def fit_group_params(
    values: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按最后一维拟合每组的 FP16 scale、FP16 偏移和状态。

    values 为 float。返回 scale、offset、status，前缀维度与去掉最后一维后相同。
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"位宽超出可存储范围: {bits}")
    if values.shape[-1] == 0:
        raise ValueError("空组无法拟合")
    data = values.to(dtype=torch.float32)
    finite = torch.isfinite(data)
    safe = torch.where(finite, data, torch.zeros_like(data))
    has_finite = finite.any(dim=-1)
    group_finite = finite.all(dim=-1)
    x_min = torch.where(
        has_finite,
        safe.amin(dim=-1),
        torch.full(data.shape[:-1], float("inf"), device=data.device, dtype=data.dtype),
    )
    x_max = torch.where(
        has_finite,
        safe.amax(dim=-1),
        torch.full(data.shape[:-1], float("-inf"), device=data.device, dtype=data.dtype),
    )
    offset = round_fp16(x_min, toward="down")
    levels = (2**bits) - 1
    scale_real = (x_max - offset.to(dtype=torch.float32)) / levels
    scale = round_fp16(scale_real, toward="up")
    scale_f32 = scale.to(dtype=torch.float32)
    offset_f32 = offset.to(dtype=torch.float32)
    status = torch.full(data.shape[:-1], STATUS_INVALID, dtype=torch.int8, device=data.device)
    constant = group_finite & torch.isfinite(offset_f32) & (scale_real == 0)
    normal = (
        group_finite
        & torch.isfinite(offset_f32)
        & torch.isfinite(scale_f32)
        & (scale_real > 0)
        & (scale_f32 > 0)
    )
    status = torch.where(constant, torch.full_like(status, STATUS_CONSTANT), status)
    status = torch.where(normal, torch.full_like(status, STATUS_NORMAL), status)
    return scale, offset, status


def quantize_with_params(
    values: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    status: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """用已经存储的参数量化最后一维。常量组和无效组的码为 0。"""
    data = values.to(dtype=torch.float32)
    levels = (2**bits) - 1
    scale_f32 = scale.to(dtype=torch.float32).unsqueeze(-1)
    offset_f32 = offset.to(dtype=torch.float32).unsqueeze(-1)
    ratio = (data - offset_f32) / scale_f32
    codes = torch.round(ratio).clamp(0, levels)
    usable = (status == STATUS_NORMAL).unsqueeze(-1)
    codes = torch.where(usable, codes, torch.zeros_like(codes))
    return codes.to(dtype=torch.uint8)


def dequantize(
    codes: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    status: torch.Tensor,
) -> torch.Tensor:
    """只用存储后的参数反量化到 FP32。无效组结果为 NaN，不改写成普通数。"""
    scale_f32 = scale.to(dtype=torch.float32).unsqueeze(-1)
    offset_f32 = offset.to(dtype=torch.float32).unsqueeze(-1)
    restored = codes.to(dtype=torch.float32) * scale_f32 + offset_f32
    invalid = (status == STATUS_INVALID).unsqueeze(-1)
    return torch.where(invalid, torch.full_like(restored, float("nan")), restored)


def saturation_mask(
    values: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    status: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """普通组里落在码范围外、因而被饱和裁剪的元素。"""
    data = values.to(dtype=torch.float32)
    levels = (2**bits) - 1
    ratio = (data - offset.to(dtype=torch.float32).unsqueeze(-1)) / scale.to(
        dtype=torch.float32
    ).unsqueeze(-1)
    clipped = (ratio < 0) | (ratio > levels)
    return clipped & (status == STATUS_NORMAL).unsqueeze(-1)


def pack_int4(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """按 C 顺序打包。较小索引在低四位。奇数个码时补一个 0，返回补码数。"""
    flat = codes.reshape(-1).to(dtype=torch.uint8)
    remainder = int(flat.numel() % 2)
    if remainder:
        flat = torch.cat([flat, flat.new_zeros(1)])
    pair = flat.view(-1, 2)
    packed = (pair[:, 0] & 0x0F) | ((pair[:, 1] & 0x0F) << 4)
    return packed, remainder


def unpack_int4(packed: torch.Tensor, count: int) -> torch.Tensor:
    """解开 pack_int4 的结果，只保留前 count 个码。"""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    flat = torch.stack([low, high], dim=-1).reshape(-1)
    return flat[:count]


def quantize_dequantize_group(
    values: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """拟合、量化再反量化。返回还原值、scale、offset、status。"""
    scale, offset, status = fit_group_params(values, bits)
    codes = quantize_with_params(values, scale, offset, status, bits)
    restored = dequantize(codes, scale, offset, status)
    return restored, scale, offset, status
