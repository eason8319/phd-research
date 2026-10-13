"""用配对区间整理真实生成与误差来源的 H=0 结果，并选择残留诊断的低位宽表示。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from experiments.representation_interactions.code.select import (
    CANDIDATE_POLICY,
    select_representations,
)
from kvstudy.intervals import class_unit_passes, quality_state
from kvstudy.paths import project_root
from phase1.quality_stats import _read, _paired, _interval


KV8 = "kv8-per_token-post"
KV4 = "kv4-per_token-post"
TASKS = ("gsm8k", "hotpot")


def generation_candidates(matrix_payload: dict[str, Any]) -> tuple[str, ...]:
    """使用当前共同候选规则；历史 kept 字段仅属于原运行记录。"""
    retained = select_representations(matrix_payload["summary"])
    return (KV8, KV4, *retained.kept)


def require_generation_results(folder: Path, active_candidates: list[str]) -> None:
    """缺测时停止重新选择；active_candidates 已排除核验通过的提前停止配置。"""
    required_files = []
    for task in TASKS:
        required_files.append(folder / f"v_{task}_bf16.json")
        required_files.extend(folder / f"v_{task}_{name}.json" for name in active_candidates)
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise RuntimeError("四种 K8V4 的生成对照尚未齐备：" + ", ".join(missing_files))


def run(model_key: str, commit: str) -> None:
    """只用达标单元的 H=0 配对结果给每模型选择一种低位宽表示。"""
    root = project_root()
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    e1 = _read(
        root / "research/phase1/experiments/representation_interactions/results"
        / model_key / "summary.json"
    )
    seeds = _read(root / "research/phase1/splits/seeds.json")
    candidates = generation_candidates(e1)
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
    require_generation_results(folder, active_candidates)
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
                if state == "保留":
                    state = "开发晋级"  # 起始 V 只给开发证据；保留留给最终候选与扩样扩样后判定。
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
        "protocol": "phase1-full-key-retrieval",
        "commit": commit,
        "model": model_key,
        "e1_commit": e1["commit"],
        "candidate_policy": CANDIDATE_POLICY,
        "representation_candidates": [name for name in candidates if name not in (KV8, KV4)],
        "h": 0,
        "baseline_usable": baseline_ok,
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
    (folder / "no_residual_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, "selected", selected, reason, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
