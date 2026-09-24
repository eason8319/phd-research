"""F：按冻结的配对区间确定最终候选的逐单元扩样计划。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kvstudy.intervals import crosses_tolerance, quality_state
from kvstudy.paths import project_root
from phase1.select_e2 import KV8, _interval, _paired, _read

TASKS = ("gsm8k", "hotpot")
TOTALS = (64, 128, 256)


def _v_path(root: Path, model_key: str, task: str, condition: str) -> Path:
    """确定 V 的已审核逐题文件，包含 E2 残留条件。"""
    e2 = root / "research/phase1/results/e2" / model_key
    e3 = root / "research/phase1/results/e3" / model_key
    if condition == "bf16":
        return e2 / f"v_{task}_bf16.json"
    if condition == "kv8":
        return e2 / f"v_{task}_{KV8}.json"
    if condition == "e2":
        residual = _read(e2 / "residual_selection.json")
        name = residual["e3_representation"]
        suffix = f"-h{residual['e3_residual_bits']}" if residual["e3_residual_bits"] else ""
        return e2 / f"v_{task}_{name}{suffix}.json"
    if condition.startswith("e3-"):
        return e3 / f"v_{task}_{condition[3:]}.json"
    raise ValueError(f"未知 F 条件：{condition}")


def _combined(
    root: Path, model_key: str, task: str, condition: str, total: int,
) -> dict[str, Any]:
    """将冻结 V 与同条件的扩样前缀合并，保留原始逐题索引。"""
    baseline = _read(_v_path(root, model_key, task, condition))
    if baseline["n"] != 64 or len(baseline["rows"]) != 64:
        raise RuntimeError("F 的 V 起始样本数不是 64")
    if total == 64:
        return baseline
    path = (
        root / "research/phase1/results/f" / model_key
        / f"{task}_{condition}_n{total}.json"
    )
    extra = _read(path)
    if (extra["model"] != model_key or extra["task"] != task
            or extra["condition"] != condition
            or extra["extra_n"] != total - 64
            or extra["total_n_with_v"] != total
            or len(extra["rows"]) != total - 64):
        raise RuntimeError(f"F 扩样元数据不符：{path}")
    ids = [row["id"] for row in baseline["rows"] + extra["rows"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"F 的 V 与扩样题重复：{path}")
    return {"n": total, "rows": baseline["rows"] + extra["rows"]}


def _matching_inputs(results: dict[str, dict[str, Any]], total: int) -> None:
    """核对同单元各条件的题号、顺序和可用的输入全文哈希。"""
    reference = results["bf16"]["rows"]
    for name, result in results.items():
        rows = result["rows"]
        if len(rows) != total:
            raise RuntimeError(f"{name} 的 F 样本数不一致")
        for index, (old, new) in enumerate(zip(reference, rows)):
            if old["id"] != new["id"]:
                raise RuntimeError(f"{name} 的第 {index + 1} 题 ID 不一致")
            if "input_sha256" in old and (
                old["input_sha256"] != new.get("input_sha256")
                or old["input_tokens"] != new.get("input_tokens")
            ):
                raise RuntimeError(f"{name} 的第 {index + 1} 题输入不一致")
            if index >= 64 and ("input_sha256" not in old or "input_sha256" not in new):
                raise RuntimeError(f"{name} 的 F 扩样缺少输入哈希")


def run(model_key: str, total: int, commit: str) -> None:
    """登记每个最终候选的质量状态，并列出下轮必须同扩的条件。"""
    if total not in TOTALS:
        raise ValueError("F 只接受总样本数 64、128 或 256")
    root = project_root()
    e2 = _read(root / "research/phase1/results/e2" / model_key / "h0_selection.json")
    e3_folder = root / "research/phase1/results/e3" / model_key
    e3 = _read(e3_folder / "selection.json")
    gate = _read(e3_folder / "gate.json")
    seeds = _read(root / "research/phase1/splits/seeds.json")
    simple = "e2" if e2["selected_low_bit"] else "kv8"
    selected = e3["f_protocol_candidate"]
    candidate_conditions = [simple] + ([f"e3-{selected}"] if selected else [])
    cells: dict[str, dict[str, Any]] = {}
    expansion: dict[str, dict[str, Any]] = {}
    folder = root / "research/phase1/results/f" / model_key
    previous = None
    if total > 64:
        previous = _read(folder / "selection.json")
        if previous["total_n"] != (64 if total == 128 else 128):
            raise RuntimeError("F 必须按 64→128→256 的顺序复核")
        if (previous["e2_selection_commit"] != e2["commit"]
                or previous["e3_selection_commit"] != e3["commit"]
                or previous["candidates"] != candidate_conditions):
            raise RuntimeError("F 上轮扩样计划与当前候选不一致")
    for task in TASKS:
        if not e2["baseline_usable"][task]:
            cells[task] = {"status": "高精度基线未达下限"}
            continue
        if previous is not None and task not in previous["next_expansion"]:
            cells[task] = previous["cells"][task]
            continue
        active = candidate_conditions
        if previous is not None:
            requested = previous["next_expansion"][task]
            if requested["next_total_n"] != total:
                raise RuntimeError("F 扩样轮次与上次登记不一致")
            active = [name for name in candidate_conditions if name in requested["conditions"]]
        conditions = ["bf16", "kv8", *active]
        conditions = list(dict.fromkeys(conditions))
        results = {
            name: _combined(root, model_key, task, name, total)
            for name in conditions
        }
        if selected and f"e3-{selected}" in active and total > 64:
            results.update({
                f"e3-{name}": _combined(root, model_key, task, f"e3-{name}", total)
                for name in gate["random_for_v"]
            })
        _matching_inputs(results, total)
        cells[task] = dict(previous["cells"][task]) if previous is not None else {}
        crossing: list[str] = []
        for name in active:
            versus_bf16 = _paired(
                results["bf16"], results[name], task,
                seeds["bootstrap"], seeds["bootstrap_draws"],
            )
            versus_kv8 = None
            if name != "kv8":
                versus_kv8 = _paired(
                    results["kv8"], results[name], task,
                    seeds["bootstrap"], seeds["bootstrap_draws"],
                )
            intervals = [versus_bf16] + ([versus_kv8] if versus_kv8 else [])
            undecided = any(
                value["upper"] is None or crosses_tolerance(_interval(value))
                for value in intervals
            )
            invalid_samples = sum(
                row.get("invalid_groups", 0) > 0 for row in results[name]["rows"]
            )
            state = quality_state(
                _interval(versus_bf16),
                _interval(versus_kv8) if versus_kv8 else _interval(versus_bf16),
                format_ok=invalid_samples < 2,
            )
            if undecided and state != "淘汰":
                state = "证据不足"
            cells[task][name] = {
                "total_n": total, "state": state,
                "invalid_samples": invalid_samples,
                "versus_bf16": versus_bf16,
                "versus_kv8": versus_kv8,
                "interval_crosses_2_or_missing": undecided,
            }
            if undecided and state == "证据不足":
                crossing.append(name)
        if crossing and total < 256:
            needed = ["bf16", "kv8", *crossing]
            if any(name.startswith("e3-") for name in crossing):
                needed += [f"e3-{name}" for name in gate["random_for_v"]]
            expansion[task] = {
                "next_total_n": 128 if total == 64 else 256,
                "conditions": list(dict.fromkeys(needed)),
            }
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "phase1-protocol v7", "commit": commit,
        "model": model_key, "total_n": total,
        "e2_selection_commit": e2["commit"],
        "e3_selection_commit": e3["commit"],
        "candidates": candidate_conditions,
        "cells": cells, "next_expansion": expansion,
        "expansion_history": (
            previous["expansion_history"] + [{
                "from_n": previous["total_n"],
                "plan": previous["next_expansion"],
            }] if previous is not None else []
        ),
        "retrieval_32k_excluded": not e2["baseline_usable"]["retrieval"],
    }
    (folder / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, total, "F 扩样单元", list(expansion), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
