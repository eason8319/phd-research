"""把真实生成与误差来源短题的严重失效中止记录为可核验的部分观察。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from kvstudy.intervals import TOLERANCE_POINTS, binary_drop_interval


def severe_failure(
    rows: list[dict[str, Any]], references: list[dict[str, Any]], task: str,
) -> dict[str, Any] | None:
    """复用两例无效组与数学题持续耗尽失分的停止边界，不扩大统计停止规则。"""
    if sum(row["invalid_groups"] > 0 for row in rows) >= 2:
        return {"reason": "同一任务已有两例无效组样本，格式不可行"}
    if task != "gsm8k" or not 8 <= len(rows) < 64:
        return None
    if any(row["stop"] != "limit" or row["correct"] or row["invalid_groups"] for row in rows):
        return None
    intervals = []
    for reference in references:
        prefix = reference["rows"][:len(rows)]
        for row, other in zip(rows, prefix):
            if any(row[key] != other[key] for key in ("id", "input_sha256", "input_tokens")):
                raise RuntimeError("严重失效比较的题目或输入不一致")
        interval = binary_drop_interval(
            np.asarray([row["correct"] for row in prefix], dtype=bool),
            np.asarray([row["correct"] for row in rows], dtype=bool),
        )
        intervals.append(asdict(interval))
    if any(value["lower"] is not None and value["lower"] > TOLERANCE_POINTS
           for value in intervals):
        return {"reason": "生成均达到上限且失分；配对下降下界越过 2 分容差",
                "versus_bf16": intervals[0], "versus_kv8": intervals[1]}
    return None
