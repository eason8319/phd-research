"""统一逐题质量记录的格式，保留实际执行提交和历史协议含义。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

RESULT_FORMAT_VERSION = 1
RESIDUAL_TOKENS = 128


def is_quality_result(payload: Any) -> bool:
    """识别生成评分；层探针、轨迹和布局账本不属于此格式。"""
    return (
        isinstance(payload, dict) and isinstance(payload.get("task"), str)
        and isinstance(payload.get("rows"), list) and bool(payload["rows"])
        and all(isinstance(row, dict) and "new_tokens" in row for row in payload["rows"])
    )


def normalize_quality_result(
    payload: dict[str, Any], identities: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """补充共同字段，不改已有测量；缺失输入可用核验过的冻结输入重建。"""
    if not is_quality_result(payload):
        raise ValueError("该实体不是逐题质量结果")
    result = copy.deepcopy(payload)
    rows = result["rows"]
    baseline = result.get("config", result.get("source", "")).startswith("bf16")
    result["result_format_version"] = RESULT_FORMAT_VERSION
    result.setdefault("n", len(rows))
    if result["n"] != len(rows):
        raise ValueError("逐题质量结果的 n 与行数不一致")
    result.setdefault("calibration_commit", None)
    result.setdefault("residual_bits", 0)
    result.setdefault("residual_tokens", RESIDUAL_TOKENS if result["residual_bits"] else 0)
    for row in rows:
        if identities is not None:
            identity = identities[row["id"]]
            for key in ("input_sha256", "input_tokens"):
                if key in row and row[key] is not None and row[key] != identity[key]:
                    raise ValueError(f"冻结输入不一致：{row['id']} / {key}")
        else:
            identity = {"input_sha256": None, "input_tokens": None}
        recorded = row.get("input_sha256") is not None
        for key in ("input_sha256", "input_tokens"):
            if row.get(key) is None:
                row[key] = identity[key]
        row.setdefault("input_identity_source", (
            "recorded" if recorded else
            "reconstructed_from_frozen_inputs" if identities is not None else "unrecorded"
        ))
        row.setdefault("stop", None)
        row.setdefault("invalid_groups", 0 if baseline else None)
        if row["invalid_groups"] is None:
            raise ValueError("量化质量记录缺少无效组计数")
        score = row.get("f1")
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 100):
            raise ValueError("逐题 F1 不是有限的 0–100 分数")
        if type(row["new_tokens"]) is not int or row["new_tokens"] < 1:
            raise ValueError("实际生成 token 数不正确")
        for key in ("correct", "em"):
            if key in row and type(row[key]) is not bool:
                raise ValueError(f"逐题 {key} 必须为布尔值")
    invalid = sum(row["invalid_groups"] > 0 for row in rows)
    result.setdefault("invalid_samples", invalid)
    if result["invalid_samples"] != invalid:
        raise ValueError("无效样本汇总不一致")
    return result


def write_quality_result(path: Path, payload: dict[str, Any]) -> None:
    """以共同格式原子写入正式质量结果；未记录的历史字段使用 null。"""
    result = normalize_quality_result(payload)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
