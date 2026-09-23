"""E0 里不需要 GPU 的边界、往返和账本核对。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kvstudy.groups import quantize_cross_token, quantize_per_token
from kvstudy.ledger import account_pair
from kvstudy.quantize import (
    STATUS_CONSTANT,
    STATUS_INVALID,
    STATUS_NORMAL,
    fit_group_params,
    pack_int4,
    quantize_dequantize_group,
    quantize_with_params,
    saturation_mask,
    unpack_int4,
)
from kvstudy.paths import project_root


def _case(name: str, ok: bool, detail: dict[str, object]) -> dict[str, object]:
    return {"name": name, "ok": ok, "detail": detail}


def boundary_cases() -> list[dict[str, object]]:
    """逐项核对 quant.md 点名的边界。"""
    cases: list[dict[str, object]] = []
    restored, scale, offset, status = quantize_dequantize_group(
        torch.tensor([[1.0, 1.0, 1.0, 1.0]]), 4
    )
    cases.append(
        _case(
            "constant",
            int(status) == STATUS_CONSTANT and torch.equal(restored, torch.tensor([[1.0] * 4])),
            {"status": int(status), "offset": float(offset)},
        )
    )
    calibration = torch.tensor([[0.0, 1.0]])
    fit_scale, fit_offset, fit_status = fit_group_params(calibration, 4)
    outside = torch.tensor([[5.0]])
    codes = quantize_with_params(outside, fit_scale, fit_offset, fit_status, 4)
    clipped = bool(saturation_mask(outside, fit_scale, fit_offset, fit_status, 4).all())
    cases.append(_case("out_of_range", int(codes) == 15 and clipped, {"code": int(codes)}))

    partial = torch.arange(40, dtype=torch.float32).view(1, 1, 40)
    partial_out, partial_stats = quantize_per_token(partial, 4, 32)
    cases.append(
        _case(
            "partial_group",
            partial_out.shape == partial.shape and partial_stats.invalid_groups == 0,
            {"shape": list(partial_out.shape), "groups_invalid": partial_stats.invalid_groups},
        )
    )

    closed = torch.arange(64 * 4, dtype=torch.float32).view(1, 64, 4)
    _, closed_stats = quantize_cross_token(closed, 4, 32)
    open_values = torch.arange(40 * 4, dtype=torch.float32).view(1, 40, 4)
    open_out, open_stats = quantize_cross_token(open_values, 4, 32)
    tail_same = torch.equal(open_out[:, 32:, :], open_values[:, 32:, :])
    cases.append(
        _case(
            "just_closed",
            closed_stats.invalid_groups == 0,
            {"invalid": closed_stats.invalid_groups},
        )
    )
    cases.append(_case("open_group", tail_same, {"tail_unchanged": tail_same}))

    full = account_pair(
        layers=1,
        kv_heads=1,
        seq=64,
        dim=32,
        k_bits=8,
        v_bits=8,
        k_mode="per_token",
        v_mode="per_token",
        g_c=32,
        g_t=32,
        block_tokens=64,
    )
    tail = account_pair(
        layers=1,
        kv_heads=1,
        seq=80,
        dim=32,
        k_bits=8,
        v_bits=8,
        k_mode="per_token",
        v_mode="per_token",
        g_c=32,
        g_t=32,
        block_tokens=64,
    )
    cases.append(
        _case(
            "full_block",
            full.block_count == 1 and full.tail_valid_tokens == 64,
            {"blocks": full.block_count, "tail": full.tail_valid_tokens},
        )
    )
    cases.append(
        _case(
            "tail_block",
            tail.block_count == 2 and tail.tail_valid_tokens == 16,
            {"blocks": tail.block_count, "tail": tail.tail_valid_tokens},
        )
    )

    non_finite, _, _, non_status = quantize_dequantize_group(
        torch.tensor([[0.0, float("nan")]]), 8
    )
    cases.append(
        _case(
            "non_finite",
            int(non_status) == STATUS_INVALID and torch.isnan(non_finite).all(),
            {"status": int(non_status)},
        )
    )
    tiny, tiny_scale, _, tiny_status = quantize_dequantize_group(
        torch.tensor([[0.0, 1e-12]]), 8
    )
    smallest = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float16),
        torch.tensor(float("inf"), dtype=torch.float16),
    )
    del tiny
    cases.append(
        _case(
            "scale_underflow",
            int(tiny_status) == STATUS_NORMAL and float(tiny_scale) == float(smallest),
            {"scale": float(tiny_scale), "status": int(tiny_status)},
        )
    )
    overflow, _, _, overflow_status = quantize_dequantize_group(
        torch.tensor([[0.0, 1e8]]), 4
    )
    cases.append(
        _case(
            "parameter_overflow",
            int(overflow_status) == STATUS_INVALID and torch.isnan(overflow).all(),
            {"status": int(overflow_status)},
        )
    )
    return cases


def roundtrip_case() -> dict[str, object]:
    """量化、打包、解包、反量化与直接反量化一致。"""
    values = torch.arange(32, dtype=torch.float32).view(1, 32)
    restored, scale, offset, status = quantize_dequantize_group(values, 4)
    from kvstudy.quantize import dequantize

    codes = quantize_with_params(values, scale, offset, status, 4)
    packed, pad = pack_int4(codes)
    unpacked = unpack_int4(packed, int(codes.numel())).view_as(codes)
    via_pack = dequantize(unpacked, scale, offset, status)
    ok = pad == 0 and torch.equal(restored, via_pack)
    return _case("roundtrip", ok, {"packed_bytes": int(packed.numel())})


def write_report(commit: str) -> Path:
    """把 CPU 核对写到 results/e0/quant.json。"""
    cases = boundary_cases()
    cases.append(roundtrip_case())
    passed = all(bool(item["ok"]) for item in cases)
    payload = {
        "protocol": "phase1-protocol v4",
        "commit": commit,
        "passed": passed,
        "cases": cases,
    }
    out = project_root() / "research/phase1/results/e0"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "quant.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("E0 量化边界未通过")
    return path


if __name__ == "__main__":
    import sys

    write_report(sys.argv[1] if len(sys.argv) > 1 else "unknown")
