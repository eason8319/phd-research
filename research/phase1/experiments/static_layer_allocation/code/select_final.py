"""按冻结配对区间确定最终候选状态和逐单元扩样计划。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from experiments.generation_error_sources.code.select import KV8
from kvstudy.paths import project_root
from phase1.quality_stats import _read


TASKS = ("gsm8k", "hotpot")
TOTALS = (64, 128, 256)


def _canonical_condition(condition: str) -> str:
    """将内容命名的条件映射到既有结果身份，不改写历史 JSON 字段。"""
    if condition == "uniform_quantization":
        return "e2"
    if condition.startswith("allocation-") and condition[len("allocation-"):]:
        return "e3-" + condition[len("allocation-"):]
    if condition in ("bf16", "kv8", "e2") or (
        condition.startswith("e3-") and condition[3:]
    ):
        return condition
    raise ValueError(f"未知扩样条件：{condition}")


def result_file_name(task: str, condition: str, total: int) -> str:
    """按任务、内容条件和总题数定位扩样结果，兼容历史条件标识。"""
    condition = _canonical_condition(condition)
    if condition == "e2":
        label = "uniform_quantization"
    elif condition.startswith("e3-"):
        label = "allocation-" + condition[3:]
    else:
        label = condition
    return f"{task}_{label}_n{total}.json"


def expanded_result_path(folder: Path, task: str, condition: str, total: int) -> Path:
    """定位唯一扩样实体；较小轮次可读取同条件较大实体的冻结前缀。"""
    for available in TOTALS:
        if available >= total:
            path = folder / result_file_name(task, condition, available)
            if path.exists():
                return path
    raise FileNotFoundError(folder / result_file_name(task, condition, total))


def _v_path(root: Path, model_key: str, task: str, condition: str) -> Path:
    """确定 V 的已审核逐题文件，包含真实生成与误差来源残留条件。"""
    condition = _canonical_condition(condition)
    e2 = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    e3 = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    if condition == "bf16":
        return e2 / f"v_{task}_bf16.json"
    if condition == "kv8":
        return e2 / f"v_{task}_{KV8}.json"
    if condition == "e2":
        residual = _read(e2 / "residual_selection.json")
        name = residual["e3_representation"]
        suffix = f"-residual{residual['e3_residual_bits']}" if residual["e3_residual_bits"] else ""
        return e2 / f"v_{task}_{name}{suffix}.json"
    if condition.startswith("e3-"):
        return e3 / f"v_{task}_{condition[3:]}.json"
    raise ValueError(f"未知扩样条件：{condition}")


def _combined(
    root: Path, model_key: str, task: str, condition: str, total: int,
) -> dict[str, Any]:
    """将冻结 V 与同条件的扩样前缀合并，保留原始逐题索引。"""
    condition = _canonical_condition(condition)
    baseline = _read(_v_path(root, model_key, task, condition))
    if baseline["n"] != 64 or len(baseline["rows"]) != 64:
        raise RuntimeError("冻结验证集的起始样本数不是 64")
    if total == 64:
        return baseline
    folder = (
        root / "research/phase1/experiments/static_layer_allocation/results/expanded_quality"
        / model_key
    )
    path = expanded_result_path(folder, task, condition, total)
    extra = _read(path)
    if (extra["model"] != model_key or extra["task"] != task
            or _canonical_condition(extra["condition"]) != condition
            or extra["extra_n"] < total - 64
            or extra["extra_n"] + 64 != extra["total_n_with_v"]
            or len(extra["rows"]) != extra["extra_n"]):
        raise RuntimeError(f"扩样元数据不符：{path}")
    from experiments.static_layer_allocation.code.expand import _condition

    config, residual_h, source = _condition(root, model_key, condition)
    expected_digest = hashlib.sha256(json.dumps({
        "name": condition, "source": source, "config": config, "residual_h": residual_h,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    if extra["config_digest"] != expected_digest:
        raise RuntimeError(f"扩样配置与当前冻结条件不一致：{path}")
    used = extra["rows"][:total - 64]
    ids = [row["id"] for row in baseline["rows"] + used]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"验证集与扩样题重复：{path}")
    return {"n": total, "rows": baseline["rows"] + used}


def _matching_inputs(results: dict[str, dict[str, Any]], total: int) -> None:
    """核对同单元各条件的题号、顺序和可用的输入全文哈希。"""
    reference = results["bf16"]["rows"]
    for name, result in results.items():
        rows = result["rows"]
        if len(rows) != total:
            raise RuntimeError(f"{name} 的合并样本数不一致")
        for index, (old, new) in enumerate(zip(reference, rows)):
            if old["id"] != new["id"]:
                raise RuntimeError(f"{name} 的第 {index + 1} 题 ID 不一致")
            if "input_sha256" in old and (
                old["input_sha256"] != new.get("input_sha256")
                or old["input_tokens"] != new.get("input_tokens")
            ):
                raise RuntimeError(f"{name} 的第 {index + 1} 题输入不一致")
            if index >= 64 and ("input_sha256" not in old or "input_sha256" not in new):
                raise RuntimeError(f"{name} 的扩样题缺少输入哈希")


def run(model_key: str, total: int, commit: str) -> None:
    """历史命令只转入当前统一判定；旧实体读取函数继续支持审核。"""
    from experiments.static_layer_allocation.code.protocol import require_v_plan
    from experiments.static_layer_allocation.code.quality import summarize

    root = project_root()
    require_v_plan(root)
    if total not in TOTALS:
        raise ValueError("质量判定档位非法")
    summarize(root, model_key)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
