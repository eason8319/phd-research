"""真实生成与误差来源的 H=0/8/16 配对归类与带入静态层分配的残留精度。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kvstudy.intervals import class_unit_passes, quality_state
from kvstudy.paths import project_root
from experiments.generation_error_sources.code.select import KV8, _interval, _paired, _read

TASKS = ("gsm8k", "hotpot")
CLASSES = {"short_generation": "gsm8k", "retrieval": "hotpot"}


def _cell(
    folder: Path, model_key: str, name: str, task: str, h_bits: int,
    seed: int, draws: int, baseline_ok: bool,
) -> dict[str, Any]:
    suffix = f"-residual{h_bits}" if h_bits else ""
    candidate = _read(folder / f"v_{task}_{name}{suffix}.json")
    reference = _read(folder / f"v_{task}_bf16.json")
    kv8 = _read(folder / f"v_{task}_{KV8}.json")
    if (candidate["model"] != model_key or candidate["config"] != name
            or candidate["n"] != 64 or candidate.get("residual_bits", 0) != h_bits):
        raise RuntimeError("残留结果的配置或冻结样本数不符")
    versus_bf16 = _paired(reference, candidate, task, seed, draws)
    versus_kv8 = _paired(kv8, candidate, task, seed, draws)
    format_ok = candidate["invalid_samples"] < 2
    if baseline_ok:
        state = quality_state(
            _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok
        )
        if state == "保留":
            state = "开发晋级"
        passes = class_unit_passes(
            _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok
        )
    else:
        state = "基线未达下限"
        passes = None
    return {
        "score": candidate["em"] if task == "gsm8k" else candidate["f1"],
        "invalid_samples": candidate["invalid_samples"],
        "versus_bf16": versus_bf16,
        "versus_kv8": versus_kv8,
        "state": state,
        "class_unit_passes": passes,
    }


def run(model_key: str, commit: str) -> None:
    """只按预定的两个达标任务单元判定；低于门槛的检索题保留缺口。"""
    root = project_root()
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    selected = _read(folder / "no_residual_selection.json")
    name = selected["selected_low_bit"]
    if name is None:
        payload = {
            "protocol": "phase1-full-key-retrieval", "commit": commit, "model": model_key,
            "selected_low_bit": None, "e3_representation": KV8,
            "e3_residual_bits": 0, "reason": "低位宽在 H=0 下全部淘汰",
        }
    else:
        seeds = _read(root / "research/phase1/splits/seeds.json")
        baseline_ok = selected["baseline_usable"]
        cells = {
            task: {
                str(h_bits): _cell(
                    folder, model_key, name, task, h_bits,
                    seeds["bootstrap"], seeds["bootstrap_draws"], baseline_ok[task],
                )
                for h_bits in (0, 8, 16)
            }
            for task in TASKS
        }
        categories: dict[str, str] = {}
        rescued: list[str] = []
        for class_name, task in CLASSES.items():
            if not baseline_ok[task]:
                categories[class_name] = "未归类"
            elif cells[task]["0"]["class_unit_passes"]:
                categories[class_name] = "无残留可行"
            elif (cells[task]["8"]["class_unit_passes"]
                    or cells[task]["16"]["class_unit_passes"]):
                categories[class_name] = "依赖缓冲"
                rescued.append(class_name)
            else:
                categories[class_name] = "有无残留都不可行"
        h_for_e3 = 0
        if rescued:
            for h_bits in (8, 16):
                if all(cells[CLASSES[class_name]][str(h_bits)]["class_unit_passes"]
                       for class_name in rescued):
                    h_for_e3 = h_bits
                    break
        payload = {
            "protocol": "phase1-full-key-retrieval", "commit": commit, "model": model_key,
            "selected_low_bit": name, "e3_representation": name,
            "e3_residual_bits": h_for_e3,
            "baseline_usable": baseline_ok,
            "retrieval_32k_excluded": not baseline_ok["retrieval"],
            "cells": cells, "categories": categories,
            "rescued_classes": rescued,
            "reason": (
                "最小残留精度同时挽回所有被挽回类" if h_for_e3
                else "没有被挽回的类，或同一残留精度不能同时挽回全部类"
            ),
        }
    (folder / "residual_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, payload["e3_representation"], payload["e3_residual_bits"], flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
