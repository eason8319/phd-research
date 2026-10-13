"""第一阶段共用的结果读取与题目级配对质量区间。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from kvstudy.artifacts import read_json as _read
from kvstudy.intervals import binary_drop_interval, bootstrap_drop_interval


def _paired(
    reference: dict[str, Any], candidate: dict[str, Any], task: str, seed: int, draws: int
) -> dict[str, Any]:
    """同一冻结样本上计算参考减候选的分数与区间。"""
    if reference["n"] != candidate["n"]:
        raise RuntimeError("配对结果样本数不一致")
    if [row["id"] for row in reference["rows"]] != [row["id"] for row in candidate["rows"]]:
        raise RuntimeError("配对结果样本 ID 或顺序不一致")
    if all("input_sha256" in row for row in reference["rows"]):
        for old, new in zip(reference["rows"], candidate["rows"]):
            if (old["input_sha256"] != new.get("input_sha256")
                    or old["input_tokens"] != new.get("input_tokens")):
                raise RuntimeError("配对输入不一致")
    if task == "hotpot":
        if not all("input_sha256" in row for row in reference["rows"]):
            raise RuntimeError("HotpotQA 参考结果缺少输入校验值")
        old_scores = np.asarray([row["f1"] for row in reference["rows"]])
        new_scores = np.asarray([row["f1"] for row in candidate["rows"]])
        interval = bootstrap_drop_interval(old_scores, new_scores, seed=seed, draws=draws)
    else:
        old_scores = np.asarray([row["correct"] for row in reference["rows"]], dtype=bool)
        new_scores = np.asarray([row["correct"] for row in candidate["rows"]], dtype=bool)
        interval = binary_drop_interval(old_scores, new_scores)
    return asdict(interval)


def _interval(value: dict[str, Any]) -> Any:
    """把 JSON 形式的配对区间恢复成判定函数所需对象。"""
    from kvstudy.intervals import DropInterval

    return DropInterval(**value)
