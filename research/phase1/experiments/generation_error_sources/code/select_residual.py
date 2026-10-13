"""真实生成与误差来源的 H=0/8/16 配对归类与带入静态层分配的残留精度。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from experiments.generation_error_sources.code.select import KV8
from kvstudy.intervals import class_unit_passes, quality_state
from kvstudy.ledger import account_pair
from kvstudy.paths import project_root
from phase1.generation import _config
from phase1.models import MODELS
from phase1.quality_stats import _interval, _paired, _read


TASKS = ("gsm8k", "hotpot")
CLASSES = {"short_generation": "gsm8k", "retrieval": "hotpot"}
JOINT_POLICY = "joint_measured_format_residual"


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


def _capacity(
    name: str, h_bits: int, model: dict[str, Any], lengths: list[int],
) -> int:
    """在主矩阵同一批 C 输入长度上累加完整驻留字节，包括 R=128 的残留。"""
    config = _config(name)
    return sum(account_pair(
        layers=model["num_hidden_layers"], kv_heads=model["num_key_value_heads"],
        seq=length, dim=model["hidden_size"] // model["num_attention_heads"],
        k_bits=config["k_bits"], v_bits=config["v_bits"],
        k_mode=config["key_mode"], v_mode=config["value_mode"],
        g_c=32, g_t=32, block_tokens=64, residual_h=h_bits,
        residual_r=128 if h_bits else 0,
    ).resident_bytes for length in lengths)


def _rank(entry: dict[str, Any]) -> tuple[float, int, str, int]:
    """质量最大下降优先；完整字节次之，名称和 H 只用于完全平局。"""
    return (entry["maximum_drop"], entry["complete_bytes"],
            entry["representation"], entry["H"])


def _joint(root: Path, model_key: str, commit: str) -> dict[str, Any]:
    """仅对完整已测且未淘汰的表示加 H 联合排名，不补跑缺失 H。"""
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    h0 = _read(folder / "no_residual_selection.json")
    seeds = _read(root / "research/phase1/splits/seeds.json")
    matrix = _read(root / "research/phase1/experiments/representation_interactions/results"
                   / model_key / "summary.json")
    geometry = _read(root / MODELS[model_key] / "config.json")
    source_index = _read(root / "research/phase1/source_index.json")
    capacity_rows = source_index["shared_capacity_inputs"][model_key]["rows"]
    if len(capacity_rows) != 96 or len({row["id"] for row in capacity_rows}) != 96:
        raise RuntimeError("完整容量需要主矩阵同一批 96 道 C 输入")
    lengths = [row["input_tokens"] for row in capacity_rows]
    tasks = [task for task in TASKS if h0["baseline_usable"][task]]
    names = [name for name in h0["cells"]["gsm8k"]
             if all(h0["cells"][task][name]["state"] != "淘汰" for task in tasks)]
    all_cells, ranking, excluded, missing, sources = {}, [], [], [], {}
    for name in names:
        if _capacity(name, 0, geometry, lengths) != matrix["summary"][name]["bytes"]:
            raise RuntimeError(f"主矩阵完整字节不能由同一批 C 输入复现：{name}")
        all_cells[name] = {task: {} for task in TASKS}
        for h_bits in (0, 8, 16):
            suffix = f"-residual{h_bits}" if h_bits else ""
            paths = {task: folder / f"v_{task}_{name}{suffix}.json" for task in TASKS}
            if not all(path.exists() for path in paths.values()):
                missing.append({"representation": name, "H": h_bits,
                                "tasks": [task for task, path in paths.items()
                                          if not path.exists()]})
                continue
            cells = {}
            for task, path in paths.items():
                value = _read(path)
                if value["residual_tokens"] != (128 if h_bits else 0):
                    raise RuntimeError(f"残留长度不符：{path}")
                cells[task] = _cell(folder, model_key, name, task, h_bits,
                                    seeds["bootstrap"], seeds["bootstrap_draws"],
                                    h0["baseline_usable"][task])
                all_cells[name][task][str(h_bits)] = cells[task]
                relative = path.relative_to(root).as_posix()
                sources[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "execution_commit": value["commit"]}
            entry = {"representation": name, "H": h_bits, "R": 128 if h_bits else 0,
                     "complete_bytes": _capacity(name, h_bits, geometry, lengths),
                     "maximum_drop": max(
                         cells[task][reference]["estimate"] for task in tasks
                         for reference in ("versus_bf16", "versus_kv8")),
                     "class_passes": {task: cells[task]["class_unit_passes"] for task in tasks}}
            if any(cells[task]["state"] == "淘汰" for task in tasks):
                excluded.append(entry)
            else:
                ranking.append(entry)
    ranking.sort(key=_rank)
    if not ranking:
        raise RuntimeError("联合候选为空，需核查基线边界与均匀 KV8 回退")
    selected = ranking[0]
    name, h_bits = selected["representation"], selected["H"]
    cells = all_cells[name]
    categories, rescued = {}, []
    for class_name, task in CLASSES.items():
        measured = cells[task]
        if not h0["baseline_usable"][task]:
            categories[class_name] = "未归类"
        elif measured["0"]["class_unit_passes"]:
            categories[class_name] = "无残留可行"
        elif any(cell["class_unit_passes"] for h, cell in measured.items() if h != "0"):
            categories[class_name] = "依赖缓冲"
            rescued.append(class_name)
        else:
            categories[class_name] = ("有无残留都不可行" if len(measured) == 3
                                      else "残留覆盖不完整，未归类")
    previous = _read(folder / "residual_selection.json")
    previous_source = previous.get("previous_selection", {
        "commit": previous["commit"], "representation": previous["e3_representation"],
        "H": previous["e3_residual_bits"],
        "sha256": hashlib.sha256((folder / "residual_selection.json").read_bytes()).hexdigest(),
    })
    return {
        "protocol": "phase1-full-key-retrieval", "commit": commit, "model": model_key,
        "selection_policy": JOINT_POLICY, "selected_low_bit": name,
        "e3_representation": name, "e3_residual_bits": h_bits,
        "baseline_usable": h0["baseline_usable"],
        "retrieval_32k_excluded": not h0["baseline_usable"].get("retrieval", False),
        "cells": cells, "categories": categories, "rescued_classes": rescued,
        "ranking": ranking, "excluded_conditions": excluded, "missing_conditions": missing,
        "measurement_sources": sources, "previous_selection": previous_source,
        "capacity_scope": "主矩阵 96 道 C 输入长度的完整驻留字节之和，含残留和对齐",
        "quality_scope": "初始 V 两个有效任务各 64 题，不使用后续扩样评分",
        "reason": "全部已测未淘汰格式加 H 联合最小化最大下降，再比较完整字节",
    }


def run(model_key: str, commit: str) -> None:
    """对完整已测且未淘汰的格式与残差精度联合筛选。"""
    root = project_root()
    output = (root / "research/phase1/experiments/generation_error_sources/results"
              / model_key / "residual_selection.json")
    payload = _joint(root, model_key, commit)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(model_key, payload["e3_representation"], payload["e3_residual_bits"], flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
