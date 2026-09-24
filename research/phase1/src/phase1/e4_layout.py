"""E4：对冻结候选与布局对照计数驻留、传输和内存时间下界。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvstudy.ledger import account_pair, align_up
from kvstudy.paths import project_root
from phase1.e2_gsm import MODELS
from phase1.e2_quant import _config

DECODE_STEPS = 256
LENGTHS = (4096, 32768, 131072)
BLOCK_TOKENS = (32, 64, 128)
BANDWIDTH_TBPS = (0.1, 1.0, 3.35)
G_C = 32
G_T = 32
RESIDUAL_TOKENS = 128


@dataclass(frozen=True)
class FormatSpec:
    """一个全网或按层配置；token_tag 表示逐 token 静态标签的理想打包界。"""

    name: str
    layer_bits: tuple[tuple[int, int], ...]
    key_mode: str
    value_mode: str
    rope: str
    residual_bits: int = 0
    token_tag: bool = False


def _from_name(name: str, n_layers: int, residual_bits: int = 0) -> FormatSpec:
    """从已冻结的均匀表示名创建全层配置。"""
    config = _config(name)
    return FormatSpec(
        name=name,
        layer_bits=((config["k_bits"], config["v_bits"]),) * n_layers,
        key_mode=config["key_mode"],
        value_mode=config["value_mode"],
        rope=config["rope"],
        residual_bits=residual_bits,
    )


def _from_allocation(
    name: str, representation: str, rows: list[dict[str, int]],
    residual_bits: int,
) -> FormatSpec:
    """从 E3 已冻结的层分配创建布局。"""
    base = _config(representation)
    return FormatSpec(
        name=name,
        layer_bits=tuple((row["k"], row["v"]) for row in rows),
        key_mode=base["key_mode"],
        value_mode=base["value_mode"],
        rope=base["rope"],
        residual_bits=residual_bits,
    )


def _specs(root: Path, model_key: str, n_layers: int) -> list[FormatSpec]:
    """只依据 F 的已完成选择确定质量候选，再补固定成本对照。"""
    folder = root / "research/phase1/results"
    final = json.loads((folder / "f" / model_key / "selection.json").read_text("utf-8"))
    residual = json.loads(
        (folder / "e2" / model_key / "residual_selection.json").read_text("utf-8")
    )
    if (final["protocol"] != "phase1-protocol v7" or final["model"] != model_key
            or residual["protocol"] != "phase1-protocol v7"
            or residual["model"] != model_key):
        raise RuntimeError("E4 来源不是当前模型的冻结协议结果")
    if final["total_n"] not in (64, 128, 256) or final["next_expansion"]:
        raise RuntimeError("F 尚未完成必要扩样，不能冻结 E4 质量候选")
    specs = []
    for condition in final["candidates"]:
        if condition == "kv8":
            specs.append(_from_name("kv8-per_token-post", n_layers))
        elif condition == "e2":
            specs.append(_from_name(
                residual["e3_representation"], n_layers,
                int(residual["e3_residual_bits"]),
            ))
        elif condition.startswith("e3-"):
            allocations = json.loads(
                (folder / "e3" / model_key / "allocations.json").read_text("utf-8")
            )
            if (allocations["protocol"] != "phase1-protocol v7"
                    or allocations["model"] != model_key):
                raise RuntimeError("E3 分配与当前模型的协议不一致")
            allocation_name = condition[3:]
            rows = allocations["allocations"][allocation_name]["layer_bits"]
            if len(rows) != n_layers:
                raise RuntimeError("E3 层数与模型不一致")
            specs.append(_from_allocation(
                condition, allocations["representation"], rows,
                int(allocations["residual_bits"]),
            ))
        else:
            raise RuntimeError(f"未知 F 候选：{condition}")
    for name in (
        "kv8-per_token-post", "k8v4-per_token-post",
        "k4v8-per_token-post", "kv4-per_token-post",
        "kv8-per_channel-post-vchannel", "kv4-per_channel-post-vchannel",
        "k8v4-per_token-pre",
    ):
        if name not in {spec.name for spec in specs}:
            specs.append(_from_name(name, n_layers))
    if n_layers % 2:
        raise RuntimeError("等比例按层布局要求偶数层")
    specs.append(FormatSpec(
        name="p50-static-layer",
        layer_bits=((8, 8),) * (n_layers // 2) + ((4, 4),) * (n_layers // 2),
        key_mode="per_token",
        value_mode="per_token",
        rope="post",
    ))
    specs.append(FormatSpec(
        name="p50-static-token-tags",
        layer_bits=((4, 8),) * n_layers,
        key_mode="per_token",
        value_mode="per_token",
        rope="post",
        token_tag=True,
    ))
    return specs


def _tagged_components(
    *, n_layers: int, heads: int, dim: int, seq: int, block: int
) -> dict[str, int]:
    """每层每 KV head 每 token 一字节标签；K/V 同时取交替的 4/8 bit。"""
    low = (seq + 1) // 2
    high = seq // 2
    units = n_layers * heads
    payload_side = units * dim * (low * 4 + high * 8) // 8
    params_side = units * seq * math.ceil(dim / G_C) * 4
    labels = units * seq
    raw = {
        "k_payload": payload_side,
        "v_payload": payload_side,
        "k_params": params_side,
        "v_params": params_side,
        "residual": 0,
        "labels": labels,
    }
    full, tail = divmod(seq, block)
    full_payload = heads * dim * (block // 2 * 4 + block // 2 * 8) // 8
    tail_payload = heads * dim * ((tail + 1) // 2 * 4 + tail // 2 * 8) // 8
    payload_stored = full * align_up(full_payload).stored_bytes
    if tail:
        payload_stored += align_up(tail_payload).stored_bytes
    params_stored = _blocked_stored(heads * math.ceil(dim / G_C) * 4, seq, block)
    labels_stored = _blocked_stored(heads, seq, block)
    stored = n_layers * (2 * payload_stored + 2 * params_stored + labels_stored)
    return {**raw, "stored": stored, "padding": stored - sum(raw.values())}


def _blocked_stored(bytes_per_token: int, seq: int, block: int) -> int:
    """每个物理块单独做 128 字节对齐，尾块仅写入有效 token。"""
    full, tail = divmod(seq, block)
    return (
        full * align_up(bytes_per_token * block).stored_bytes
        + (align_up(bytes_per_token * tail).stored_bytes if tail else 0)
    )


def _components(
    spec: FormatSpec, *, heads: int, dim: int, seq: int,
    batch: int, block: int,
) -> dict[str, int]:
    """按侧相加层布局，返回原始字节和 128 字节对齐后的驻留量。"""
    if spec.token_tag:
        return _tagged_components(
            n_layers=len(spec.layer_bits), heads=heads * batch,
            dim=dim, seq=seq, block=block,
        )
    total = {
        "k_payload": 0, "v_payload": 0, "k_params": 0,
        "v_params": 0, "residual": 0, "labels": 0,
        "stored": 0, "padding": 0,
    }
    for k_bits, v_bits in spec.layer_bits:
        ledger = account_pair(
            layers=1, kv_heads=heads * batch, seq=seq, dim=dim,
            k_bits=k_bits, v_bits=v_bits,
            k_mode=spec.key_mode, v_mode=spec.value_mode,
            g_c=G_C, g_t=G_T, block_tokens=block,
            residual_h=spec.residual_bits,
            residual_r=RESIDUAL_TOKENS if spec.residual_bits else 0,
        )
        for key in ("k_payload", "v_payload", "k_params", "v_params",
                    "residual", "labels"):
            total[key] += getattr(ledger, key).raw_bytes
        units = heads * batch
        k_stored = _blocked_stored(units * dim * k_bits // 8, seq, block)
        v_stored = _blocked_stored(units * dim * v_bits // 8, seq, block)
        k_params_stored = (
            _blocked_stored(units * math.ceil(dim / G_C) * 4, seq, block)
            if spec.key_mode == "per_token" else align_up(ledger.k_params.raw_bytes).stored_bytes
        )
        v_params_stored = (
            _blocked_stored(units * math.ceil(dim / G_C) * 4, seq, block)
            if spec.value_mode == "per_token" else align_up(ledger.v_params.raw_bytes).stored_bytes
        )
        layer_stored = (
            k_stored + v_stored + k_params_stored + v_params_stored
            + align_up(ledger.residual.raw_bytes).stored_bytes
        )
        layer_raw = sum(getattr(ledger, key).raw_bytes for key in (
            "k_payload", "v_payload", "k_params", "v_params", "residual"
        ))
        total["stored"] += layer_stored
        total["padding"] += layer_stored - layer_raw
    return total


def _recent_main_bytes(
    spec: FormatSpec, *, heads: int, dim: int, history: int, batch: int
) -> tuple[int, int]:
    """可由残留替代、因此读出时可跳过的主存 K/V 尾部字节。"""
    if not spec.residual_bits:
        return 0, 0
    tokens = min(history, RESIDUAL_TOKENS)
    skipped = [0, 0]
    for k_bits, v_bits in spec.layer_bits:
        for side, (bits, mode) in enumerate((
            (k_bits, spec.key_mode), (v_bits, spec.value_mode)
        )):
            skipped[side] += heads * batch * tokens * dim * bits // 8
            if mode == "per_token":
                skipped[side] += heads * batch * tokens * math.ceil(dim / G_C) * 4
    return skipped[0], skipped[1]


def _residual_write(
    spec: FormatSpec, *, heads: int, dim: int, batch: int
) -> int:
    """滑动窗口驻留量不变，但每个新 token 仍写入一份残留副本。"""
    if not spec.residual_bits:
        return 0
    units = len(spec.layer_bits) * heads * batch
    raw = 2 * units * dim * spec.residual_bits // 8
    if spec.residual_bits == 8:
        raw += 2 * units * math.ceil(dim / G_C) * 4
    return raw


def _movement(
    spec: FormatSpec, *, heads: int, dim: int, history: int,
    batch: int, block: int,
) -> dict[str, int]:
    """一个 decode 步的实际原始字节；读给出替换尾部与全读两种界。"""
    before = _components(
        spec, heads=heads, dim=dim, seq=history, batch=batch, block=block
    )
    after = _components(
        spec, heads=heads, dim=dim, seq=history + 1, batch=batch, block=block
    )
    residual_side = before["residual"] // 2
    full_k = before["k_payload"] + before["k_params"] + residual_side
    full_v = before["v_payload"] + before["v_params"] + residual_side
    skip_k, skip_v = _recent_main_bytes(
        spec, heads=heads, dim=dim, history=history, batch=batch
    )
    main_keys = ("k_payload", "v_payload", "k_params", "v_params", "labels")
    main_write = sum(after[key] - before[key] for key in main_keys)
    return {
        "read_k_lower": full_k - skip_k + before["labels"] // 2,
        "read_v_lower": full_v - skip_v + before["labels"] - before["labels"] // 2,
        "read_k_full": full_k + before["labels"] // 2,
        "read_v_full": full_v + before["labels"] - before["labels"] // 2,
        "write": main_write + _residual_write(
            spec, heads=heads, dim=dim, batch=batch
        ),
    }


def _decode_window(
    spec: FormatSpec, *, heads: int, dim: int, length: int,
    query_heads: int, batch: int, block: int,
) -> dict[str, Any]:
    """以长度为终点，汇总连续 256 个 decode 步。"""
    start = length - DECODE_STEPS
    totals = {
        key: 0 for key in (
            "read_k_lower", "read_v_lower", "read_k_full", "read_v_full", "write"
        )
    }
    qk_flops = 0
    av_flops = 0
    softmax_elements = 0
    onchip_fp32_kv_bytes = 0
    pre_rope_mul_adds = 0
    for history in range(start, length):
        step = _movement(
            spec, heads=heads, dim=dim, history=history, batch=batch, block=block
        )
        for key, value in step.items():
            totals[key] += value
        elements = len(spec.layer_bits) * query_heads * batch * history * dim
        qk_flops += 2 * elements
        av_flops += 2 * elements
        softmax_elements += len(spec.layer_bits) * query_heads * batch * history
        onchip_fp32_kv_bytes += 8 * len(spec.layer_bits) * heads * batch * history * dim
        if spec.rope == "pre":
            pre_rope_mul_adds += (
                3 * len(spec.layer_bits) * heads * batch * history * dim
            )
    averages = {key + "_per_step": value / DECODE_STEPS for key, value in totals.items()}
    read_lower = averages["read_k_lower_per_step"] + averages["read_v_lower_per_step"]
    read_full = averages["read_k_full_per_step"] + averages["read_v_full_per_step"]
    bandwidth_bounds = []
    for tbps in BANDWIDTH_TBPS:
        width = tbps * 1e12
        bandwidth_bounds.append({
            "bandwidth_TBps": tbps,
            "shared_read_lower_s": read_lower / width,
            "shared_read_full_s": read_full / width,
            "split_equal_read_lower_s": max(
                averages["read_k_lower_per_step"] / (width / 2),
                averages["read_v_lower_per_step"] / (width / 2),
            ),
        })
    return {
        "initial_history": start,
        "final_history": length,
        "steps": DECODE_STEPS,
        "totals_bytes": totals,
        "average_bytes": averages,
        "onchip_and_arithmetic": {
            "qk_flops_per_step": qk_flops / DECODE_STEPS,
            "av_flops_per_step": av_flops / DECODE_STEPS,
            "softmax_elements_per_step": softmax_elements / DECODE_STEPS,
            "fp32_kv_intermediate_bytes_per_step": onchip_fp32_kv_bytes / DECODE_STEPS,
            "pre_rope_mul_adds_per_step": pre_rope_mul_adds / DECODE_STEPS,
            "arithmetic_intensity_lower_flops_per_byte": (
                (qk_flops + av_flops) / DECODE_STEPS / read_lower
            ),
        },
        "bandwidth_lower_bounds": bandwidth_bounds,
    }


def _layout(
    spec: FormatSpec, *, heads: int, dim: int, length: int,
    query_heads: int, batch: int, block: int,
) -> dict[str, Any]:
    """一个长度、批次和物理块条件的容量与传输记录。"""
    if spec.key_mode == "cross_token" and block % G_T:
        raise RuntimeError("跨 token 组长度不整除物理块")
    components = _components(
        spec, heads=heads, dim=dim, seq=length, batch=batch, block=block
    )
    elements = 2 * len(spec.layer_bits) * heads * batch * length * dim
    nominal = (
        6.0 if spec.token_tag else
        sum(k_bits + v_bits for k_bits, v_bits in spec.layer_bits)
        / (2 * len(spec.layer_bits))
    )
    result: dict[str, Any] = {
        "length": length,
        "batch": batch,
        "block_tokens": block,
        "block_count": math.ceil(length / block),
        "tail_valid_tokens": length % block or block,
        "nominal_bits": nominal,
        "effective_bits": components["stored"] * 8 / elements,
        "resident_bytes": components["stored"],
        "components": components,
        "parameter_fraction_of_resident": (
            components["k_params"] + components["v_params"]
        ) / components["stored"],
        "label_assumption": (
            "one_byte_per_layer_kv_head_token_for_both_sides" if spec.token_tag else None
        ),
    }
    if length != 131072:
        result["decode_256"] = _decode_window(
            spec, heads=heads, dim=dim, length=length,
            query_heads=query_heads, batch=batch, block=block,
        )
    return result


def run(model_key: str, commit: str) -> None:
    """F 扩样审核后，对正式候选和固定对照完成 E4 布局计数。"""
    root = project_root()
    model = json.loads((root / MODELS[model_key] / "config.json").read_text("utf-8"))
    n_layers = int(model["num_hidden_layers"])
    heads = int(model["num_key_value_heads"])
    query_heads = int(model["num_attention_heads"])
    dim = int(model["hidden_size"]) // int(model["num_attention_heads"])
    specs = _specs(root, model_key, n_layers)
    source_paths = [
        root / "research/phase1/results/f" / model_key / "selection.json",
        root / "research/phase1/results/e2" / model_key / "residual_selection.json",
    ]
    if any(spec.name.startswith("e3-") for spec in specs):
        source_paths.append(
            root / "research/phase1/results/e3" / model_key / "allocations.json"
        )
    source_sha256 = {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }
    rows = []
    for spec in specs:
        for length in LENGTHS:
            for block in BLOCK_TOKENS:
                rows.append({
                    "format": spec.name,
                    "key_mode": spec.key_mode,
                    "value_mode": spec.value_mode,
                    "rope": spec.rope,
                    "residual_bits": spec.residual_bits,
                    "token_tag": spec.token_tag,
                    **_layout(
                        spec, heads=heads, dim=dim, length=length,
                        query_heads=query_heads, batch=1, block=block,
                    ),
                })
    if model_key == "llama":
        simple = specs[0]
        rows.append({
            "format": simple.name,
            "key_mode": simple.key_mode,
            "value_mode": simple.value_mode,
            "rope": simple.rope,
            "residual_bits": simple.residual_bits,
            "token_tag": simple.token_tag,
            **_layout(
                simple, heads=heads, dim=dim, length=32768,
                query_heads=query_heads, batch=4, block=64,
            ),
        })
    payload = {
        "protocol": "phase1-protocol v7",
        "commit": commit,
        "model": model_key,
        "source": "layout_count_and_assumed_bandwidth_lower_bound",
        "alignment_bytes": 128,
        "decode_window_ends_at_length": True,
        "bandwidth_TBps": BANDWIDTH_TBPS,
        "n_layers": n_layers,
        "kv_heads": heads,
        "query_heads": query_heads,
        "head_dim": dim,
        "source_sha256": source_sha256,
        "service_time_model": (
            "T_QK + T_softmax + T_AV + T_dequant + T_pre_rope; "
            "T_QK>=max(D_K/W_K,F_QK/P_QK), "
            "T_AV>=max(D_V/W_V,F_AV/P_AV); "
            "unknown compute rates and dependence costs remain symbolic"
        ),
        "rows": rows,
    }
    folder = root / "research/phase1/results/e4" / model_key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "layout.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
