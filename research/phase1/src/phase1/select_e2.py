"""用配对区间整理 E2 的 H=0 结果，并选择残留诊断的低位宽表示。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from kvstudy.intervals import (
    binary_drop_interval,
    bootstrap_drop_interval,
    class_unit_passes,
    quality_state,
)
from kvstudy.paths import project_root

KV8 = "kv8-per_token-post"
KV4 = "kv4-per_token-post"
TASKS = ("gsm8k", "hotpot")


def _read(path: Path) -> dict[str, Any]:
    """读取一份已完成的逐样本结果。"""
    return json.loads(path.read_text(encoding="utf-8"))


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


def run(model_key: str, commit: str) -> None:
    """只用达标单元的 H=0 配对结果给每模型选择一种低位宽表示。"""
    root = project_root()
    folder = root / "research/phase1/results/e2" / model_key
    e1 = _read(root / "research/phase1/results/e1" / model_key / "summary.json")
    seeds = _read(root / "research/phase1/splits/seeds.json")
    candidates = list(dict.fromkeys([KV8, KV4, *e1["kept"]]))
    early_stops = {
        name: _read(folder / f"v_gsm8k_{name}_early_stop.json")
        for name in candidates
        if (folder / f"v_gsm8k_{name}_early_stop.json").exists()
    }
    for name, result in early_stops.items():
        if (name == KV8 or result["config"] != name or not result["stopped_early"]
                or result["versus_bf16"]["lower"] <= 2
                and result["versus_kv8"]["lower"] <= 2):
            raise RuntimeError("提前停止记录不满足严重失效条件")
    active_candidates = [name for name in candidates if name not in early_stops]
    references: dict[str, dict[str, Any]] = {}
    quantized: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TASKS:
        references[task] = _read(folder / f"v_{task}_bf16.json")
        quantized[task] = {
            name: _read(folder / f"v_{task}_{name}.json") for name in active_candidates
        }
    baseline_ok = {
        "gsm8k": references["gsm8k"]["em"] >= 50,
        "hotpot": references["hotpot"]["f1"] >= 40,
    }
    retrieval = _read(folder / "v_retrieval_bf16.json")
    retrieval_ok = retrieval["em"] >= 90
    cells: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        cells[task] = {}
        for name in active_candidates:
            candidate = quantized[task][name]
            if candidate["n"] != 64 or references[task]["n"] != 64:
                raise RuntimeError("V 起始样本数不是 64")
            if name == KV8:
                continue
            versus_bf16 = _paired(
                references[task], candidate, task, seeds["bootstrap"], seeds["bootstrap_draws"]
            )
            versus_kv8 = _paired(
                quantized[task][KV8], candidate, task,
                seeds["bootstrap"], seeds["bootstrap_draws"],
            )
            format_ok = candidate["invalid_samples"] < 2
            if baseline_ok[task]:
                state = quality_state(
                    _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok
                )
                passes = class_unit_passes(
                    _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok
                )
            else:
                state = "基线未达下限"
                passes = None
            cells[task][name] = {
                "score": candidate["em"] if task == "gsm8k" else candidate["f1"],
                "invalid_samples": candidate["invalid_samples"],
                "versus_bf16": versus_bf16,
                "versus_kv8": versus_kv8,
                "state": state,
                "class_unit_passes": passes,
            }
    low_bit = [name for name in active_candidates if name != KV8]
    eligible = [
        name for name in low_bit
        if all(cells[task][name]["state"] != "淘汰"
               for task in TASKS if baseline_ok[task])
    ]
    if not any(baseline_ok.values()):
        selected = None
        reason = "没有达标的质量单元"
    elif not eligible:
        selected = None
        reason = "低位宽表示在参加判断的单元上全部淘汰"
    else:
        def rank(name: str) -> tuple[float, float, str]:
            points = [
                max(cells[task][name]["versus_bf16"]["estimate"],
                    cells[task][name]["versus_kv8"]["estimate"])
                for task in TASKS if baseline_ok[task]
            ]
            return max(points), e1["summary"][name]["bytes"], name
        selected = min(eligible, key=rank)
        reason = "参加判断的单元最大下降点估计最小"
    payload = {
        "protocol": "phase1-protocol v7",
        "commit": commit,
        "model": model_key,
        "e1_commit": e1["commit"],
        "h": 0,
        "baseline_usable": {**baseline_ok, "retrieval": retrieval_ok},
        "retrieval_quality_excluded": not retrieval_ok,
        "early_stops": {
            name: {
                "observed_n": result["observed_n"],
                "planned_n": result["planned_n"],
                "versus_bf16": result["versus_bf16"],
                "versus_kv8": result["versus_kv8"],
                "reason": result["reason"],
            }
            for name, result in early_stops.items()
        },
        "cells": cells,
        "selected_low_bit": selected,
        "fallback_if_none": KV8,
        "reason": reason,
    }
    (folder / "h0_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, "selected", selected, reason, flush=True)


def _interval(value: dict[str, Any]) -> Any:
    """把 JSON 形式的配对区间恢复成判定函数所需对象。"""
    from kvstudy.intervals import DropInterval

    return DropInterval(**value)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
