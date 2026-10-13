"""保留完整的 K8V4 表示对照，供两个模型使用同一候选集。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CANDIDATE_POLICY = "all-k8v4-representations"
K8V4_CANDIDATES = (
    "k8v4-per_channel-pre",
    "k8v4-per_channel-post",
    "k8v4-per_token-pre",
    "k8v4-per_token-post",
)
KV8_BASELINE = "kv8-per_token-post"


@dataclass(frozen=True)
class RepresentationSelection:
    """完整保留的表示与依据；测量不完整时不生成部分候选集。"""

    kept: tuple[str, ...]
    reason: str


def select_representations(summary: dict[str, dict[str, Any]]) -> RepresentationSelection:
    """核对四种 K8V4 的测量和容量，不按局部误差排名截断候选。"""
    required = (KV8_BASELINE, *K8V4_CANDIDATES)
    missing = [name for name in required if name not in summary]
    if missing:
        raise RuntimeError("主矩阵缺少候选或 KV8 基线：" + ", ".join(missing))
    for name in required:
        for metric in ("error", "invalid", "bytes"):
            value = summary[name].get(metric)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise RuntimeError(f"主矩阵测量无效：{name} {metric}")
        if summary[name]["invalid"] != int(summary[name]["invalid"]):
            raise RuntimeError(f"主矩阵无效组计数不是整数：{name}")
        if summary[name]["bytes"] <= 0:
            raise RuntimeError(f"主矩阵容量不是正数：{name}")
    kv8_bytes = summary[KV8_BASELINE]["bytes"]
    too_large = [name for name in K8V4_CANDIDATES if summary[name]["bytes"] >= kv8_bytes]
    if too_large:
        raise RuntimeError("候选完整容量未低于均匀 KV8：" + ", ".join(too_large))
    return RepresentationSelection(
        K8V4_CANDIDATES, "四种 K8V4 均保留，两个模型使用同一候选集"
    )
