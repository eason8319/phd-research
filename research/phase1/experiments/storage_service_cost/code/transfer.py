"""把既有有效字节映射到显式事务模型，不重跑布局或质量测量。"""

from __future__ import annotations

import math
from typing import Any


def stream_step(history: int, stride: int, block: int, granule: int,
                mode: str) -> dict[str, int]:
    """计算一个 head 的单流；单位为字节，当前 token 只追加不回读。"""
    if min(stride, block, granule) <= 0 or history < 0:
        raise ValueError("流形状必须为正，历史长度不得为负")
    if block * stride % granule or granule % stride:
        raise ValueError("本模型要求完整块和单 token 均可整分事务")
    valid = history * stride
    offset = valid % granule
    effective = valid
    if mode == "immediate_rmw":
        read = math.ceil(valid / granule) * granule
        write = granule
        rmw = granule if offset else 0
        buffered = 0
    elif mode == "combined":
        read = valid - offset
        write = granule if offset + stride == granule else 0
        rmw = 0
        buffered = offset
    else:
        raise ValueError(f"未知传输策略：{mode}")
    return {"effective_read": effective, "read": read, "append_effective": stride,
            "append_write": write, "rmw_read": rmw, "buffer_read": buffered,
            "read_tail_waste": read + buffered - effective,
            "pending_after": (offset + stride) % granule}


def boundary_flush(history: int, stride: int, granule: int) -> dict[str, int]:
    """请求结束或驱逐刷新；恢复合并流时先读回已有有效尾部。"""
    pending = history * stride % granule
    return {"flush_write": granule if pending else 0,
            "flush_waste": granule - pending if pending else 0,
            "resume_seed_read": granule if pending else 0}


def transfer_row(layout: dict[str, Any], row: dict[str, Any],
                 config: dict[str, Any], mode: str) -> dict[str, Any]:
    """按逐层真实位宽累计 256 步；保留流分量和缓冲守恒。"""
    granule = config["transaction_bytes"]
    shape = layout["actual_configurations"][row["format"]]
    heads = layout["kv_heads"] * row["batch"]
    dim = layout["head_dim"]
    steps = layout["decode_steps"]
    start = row["length"] - steps
    streams: dict[str, dict[str, float]] = {}
    for side, index in (("k", 0), ("v", 1)):
        for kind in ("payload", "params"):
            total: dict[str, float] = {}
            for bits in shape["layer_bits"]:
                stride = (dim * bits[index] // 8 if kind == "payload"
                          else dim // config["group_channels"] * 4)
                for history in range(start, row["length"]):
                    item = stream_step(history, stride, row["block_tokens"], granule, mode)
                    for key, value in item.items():
                        if key != "pending_after":
                            total[key] = total.get(key, 0) + value * heads
            streams[f"{side}_{kind}"] = {key: value / steps for key, value in total.items()}
    effective = sum(v["effective_read"] + v["append_effective"] for v in streams.values())
    interface = sum(v["read"] + v["append_write"] + v["rmw_read"]
                    for v in streams.values())
    tail = sum(v["read_tail_waste"] for v in streams.values())
    buffer = sum(v["buffer_read"] for v in streams.values())
    write_excess = sum(v["append_write"] - v["append_effective"] for v in streams.values())
    rmw = sum(v["rmw_read"] for v in streams.values())
    assert math.isclose(interface - effective, tail - buffer + write_excess + rmw)
    old = row["decode_256"]["average_bytes"]["io_lower_per_step"]
    if not math.isclose(effective, old, rel_tol=1e-12):
        raise RuntimeError("新流拆分没有复现复用的有效字节")
    return {
        "format": row["format"], "length": row["length"], "batch": row["batch"],
        "block_tokens": row["block_tokens"], "mode": mode,
        "steps": steps, "generated_tokens": steps * row["batch"],
        "average_per_batch_step": streams,
        "effective_io_bytes_per_step": effective, "interface_io_bytes_per_step": interface,
        "interface_io_bytes_per_generated_token": interface / row["batch"],
        "difference": {"read_tail_waste": tail, "buffer_served_read": buffer,
                       "append_padding_or_rewrites": write_excess, "rmw_read": rmw},
        "write_buffer_reserved_bytes": (4 * len(shape["layer_bits"]) * heads * granule
                                        if mode == "combined" else 0),
        "window_initial_and_final_pending_bytes": [0, 0],
        "prefill": {**row["prefill_conversion_once"],
                    "interface_write_bytes": row["resident_bytes"],
                    "tail_transaction_waste_bytes": 0},
    }
