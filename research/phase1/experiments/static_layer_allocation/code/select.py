"""静态层分配：按同样的 V 输入比较静态层分配，并给最终候选与扩样留下协议候选。"""

from __future__ import annotations

import json
import sys
from typing import Any

from kvstudy.intervals import class_unit_passes, quality_state
from kvstudy.paths import project_root
from experiments.generation_error_sources.code.select import KV8, _interval, _paired, _read

TASKS = ("gsm8k", "hotpot")


def run(model_key: str, commit: str) -> None:
    root = project_root()
    folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    e2_folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    gate = _read(folder / "gate.json")
    allocations = _read(folder / "allocations.json")
    e2 = _read(e2_folder / "no_residual_selection.json")
    seeds = _read(root / "research/phase1/splits/seeds.json")
    names = gate["selected_nonrandom_for_v"] + gate["random_for_v"]
    baseline_ok = e2["baseline_usable"]
    if baseline_ok["retrieval"]:
        raise RuntimeError("32K 检索基线达标时必须补齐该单元的静态层分配 V 结果")
    cells: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        cells[task] = {}
        bf16 = _read(e2_folder / f"v_{task}_bf16.json")
        kv8 = _read(e2_folder / f"v_{task}_{KV8}.json")
        for name in names:
            candidate = _read(folder / f"v_{task}_{name}.json")
            if (candidate["n"] != 64 or candidate["allocation"] != name
                    or candidate["allocation_commit"] != allocations["commit"]
                    or candidate["gate_commit"] != gate["commit"]):
                raise RuntimeError("静态层分配 V 质量结果不属于当前冻结分配或样本数")
            versus_bf16 = _paired(
                bf16, candidate, task,
                seeds["bootstrap"], seeds["bootstrap_draws"],
            )
            versus_kv8 = _paired(
                kv8, candidate, task,
                seeds["bootstrap"], seeds["bootstrap_draws"],
            )
            format_ok = candidate["invalid_samples"] < 2
            if baseline_ok[task]:
                state = quality_state(
                    _interval(versus_bf16), _interval(versus_kv8),
                    format_ok=format_ok,
                )
                if state == "保留":
                    state = "开发晋级"
                passes = class_unit_passes(
                    _interval(versus_bf16), _interval(versus_kv8),
                    format_ok=format_ok,
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
    active_retrieval = [task for task in ("hotpot",) if baseline_ok[task]]
    sensitivity = "p50_sensitivity"
    better_random: bool | None = None
    if gate["sensitivity_strictly_below_all_random"] and active_retrieval:
        better_random = all(
            cells[task][sensitivity]["versus_bf16"]["estimate"]
            < cells[task][random_name]["versus_bf16"]["estimate"]
            for task in active_retrieval for random_name in gate["random_for_v"]
        )
    selected_e3 = None
    if gate["sensitivity_strictly_below_all_random"] and active_retrieval:
        def rank(name: str) -> tuple[float, int, str]:
            worst = max(
                cells[task][name]["versus_bf16"]["estimate"]
                for task in active_retrieval
            )
            return worst, allocations["allocations"][name]["resident_bytes_32k"], name

        selected_e3 = min(gate["selected_nonrandom_for_v"], key=rank)
    payload = {
        "protocol": "phase1-full-key-retrieval", "commit": commit,
        "model": model_key, "gate_commit": gate["commit"],
        "allocation_commit": allocations["commit"],
        "baseline_usable": baseline_ok,
        "retrieval_32k_excluded": not baseline_ok["retrieval"],
        "active_retrieval_units": active_retrieval,
        "cells": cells,
        "sensitivity_better_than_all_random_on_active_retrieval": better_random,
        "e3_evidence": (
            "优于随机" if better_random else "静态层混合证据不足"
        ),
        "f_protocol_candidate": selected_e3,
        "simpler_e2_candidate": (
            e2["selected_low_bit"] or KV8
        ),
        "reason": (
            "C 门槛通过后，按参加判断的检索单元最大下降点估计选择"
            if selected_e3 else "C 门槛未通过或没有参加判断的检索单元"
        ),
    }
    (folder / "selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, "e3", selected_e3, payload["e3_evidence"], flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
