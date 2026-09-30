"""真实生成与误差来源：按冻结长解码子集给选定表示单独归类。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from kvstudy.intervals import class_unit_passes, quality_state
from kvstudy.paths import project_root
from experiments.generation_error_sources.code.select import KV8, _interval, _paired, _read
from experiments.generation_error_sources.code.sources import _protocol_matches

BASELINE_F1_FLOOR = 40.0
MIN_LONG_TOKENS = 256


def _source_path(folder: Path, condition: str, name: str, h_bits: int) -> Path:
    """确定长解码四份逐题文件的项目内位置。"""
    if condition == "bf16":
        return folder / "v_hotpot_long_bf16.json"
    suffix = f"-residual{h_bits}" if h_bits else ""
    return folder / f"v_hotpot_long_{name}{suffix}.json"


def _validate_results(
    root: Path, model_key: str, sources: dict[str, dict[str, Any]],
) -> list[int]:
    """核对 8 道冻结题的顺序、输入全文与实际长度，返回有效题索引。"""
    frozen = _read(root / "research/phase1/splits/hotpot.json")
    expected_ids = frozen["long_decode"]
    if len(expected_ids) != 8 or len(set(expected_ids)) != 8:
        raise RuntimeError("冻结长解码子集不是 8 道不同题")
    indexed = {row["id"]: row for row in frozen["v"]}
    expected = [indexed[item_id]["models"][model_key] for item_id in expected_ids]
    for condition, result in sources.items():
        expected_protocol = (
            "phase1-hotpot-direct-answer" if condition == "bf16" else "phase1-full-key-retrieval"
        )
        if (not _protocol_matches(result["protocol"], expected_protocol)
                or result["model"] != model_key or result["task"] != "v-hotpot_long"
                or result["n"] != 8 or len(result["rows"]) != 8):
            raise RuntimeError(f"长解码结果元数据不符：{condition}")
        for index, row in enumerate(result["rows"]):
            if (row["id"] != expected_ids[index]
                    or row["input_sha256"] != expected[index]["long_sha256"]
                    or row["input_tokens"] != expected[index]["long_tokens"]
                    or not 1 <= row["new_tokens"] <= 1024
                    or row["input_tokens"] + row["new_tokens"] > 32768
                    or not 0 <= row["f1"] <= 100):
                raise RuntimeError(f"长解码第 {index + 1} 题与冻结输入不符：{condition}")
    return [
        index for index, row in enumerate(sources["bf16"]["rows"])
        if row["new_tokens"] >= MIN_LONG_TOKENS
    ]


def _subset(result: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """以同一 BF16 长度门槛取配对统计样本，不按候选输出筛选。"""
    return {"n": len(indices), "rows": [result["rows"][index] for index in indices]}


def _cell(
    reference: dict[str, Any], kv8: dict[str, Any], candidate: dict[str, Any],
    indices: list[int], seed: int, draws: int,
) -> dict[str, Any]:
    """计算一个残留条件在有效长解码子集上的配对区间与类通过。"""
    old = _subset(reference, indices)
    medium = _subset(kv8, indices)
    new = _subset(candidate, indices)
    versus_bf16 = _paired(old, new, "hotpot", seed, draws)
    versus_kv8 = _paired(medium, new, "hotpot", seed, draws)
    invalid = sum(row.get("invalid_groups", 0) > 0 for row in new["rows"])
    format_ok = invalid < 2
    state = quality_state(
        _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok,
    )
    if state == "保留":
        state = "开发晋级"
    return {
        "n": len(indices),
        "f1": sum(row["f1"] for row in new["rows"]) / len(indices),
        "invalid_samples": invalid,
        "versus_bf16": versus_bf16,
        "versus_kv8": versus_kv8,
        "state": state,
        "class_unit_passes": class_unit_passes(
            _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok,
        ),
    }


def run(model_key: str, commit: str) -> None:
    """真实生成与误差来源第 8 步四份结果审核后记录长解码类别，保持静态层分配选择不变。"""
    root = project_root()
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    selection = _read(folder / "residual_selection.json")
    name = selection["selected_low_bit"]
    if name is None:
        raise RuntimeError("低位宽全部淘汰时，第 8 步按协议跳过")
    h_bits = int(selection["e3_residual_bits"]) or 16
    paths = {
        "bf16": _source_path(folder, "bf16", name, 0),
        "kv8": _source_path(folder, "kv8", KV8, 0),
        "h0": _source_path(folder, "h0", name, 0),
        f"h{h_bits}": _source_path(folder, "residual", name, h_bits),
    }
    sources = {"bf16": _read(paths["bf16"])}
    if sources["bf16"]["config"] != "bf16-h0":
        raise RuntimeError("长解码高精度条件不一致")
    indices = _validate_results(root, model_key, sources)
    baseline_f1 = (
        sum(sources["bf16"]["rows"][index]["f1"] for index in indices) / len(indices)
        if indices else None
    )
    baseline_usable = baseline_f1 is not None and baseline_f1 >= BASELINE_F1_FLOOR
    cells: dict[str, Any] = {}
    category = "未归类"
    if baseline_usable:
        sources.update({
            condition: _read(path) for condition, path in paths.items()
            if condition != "bf16"
        })
        if (sources["kv8"]["config"] != KV8
                or sources["h0"]["config"] != name
                or sources[f"h{h_bits}"]["config"] != name
                or sources[f"h{h_bits}"].get("residual_bits") != h_bits):
            raise RuntimeError("长解码量化条件与选定表示或残留精度不一致")
        _validate_results(root, model_key, sources)
        seeds = _read(root / "research/phase1/splits/seeds.json")
        cells = {
            condition: _cell(
                sources["bf16"], sources["kv8"], sources[condition], indices,
                seeds["bootstrap"], seeds["bootstrap_draws"],
            )
            for condition in ("h0", f"h{h_bits}")
        }
        if cells["h0"]["class_unit_passes"]:
            category = "无残留可行"
        elif cells[f"h{h_bits}"]["class_unit_passes"]:
            category = "依赖缓冲"
        else:
            category = "有无残留都不可行"
    payload = {
        "protocol": "phase1-full-key-retrieval", "commit": commit,
        "model": model_key, "selected_low_bit": name,
        "selection_commit": selection["commit"],
        "diagnostic_residual_bits": h_bits,
        "bf16_total_n": 8, "eligible_n": len(indices),
        "eligible_ids": [sources["bf16"]["rows"][index]["id"] for index in indices],
        "bf16_shorter_than_256_n": 8 - len(indices),
        "bf16_eligible_f1": baseline_f1,
        "baseline_usable": baseline_usable,
        "baseline_failure_reason": (
            None if baseline_usable else
            "no_generation_ge_256_tokens" if not indices else
            "eligible_f1_below_40"
        ),
        "cells": cells, "long_decode_category": category,
        "does_not_change_e3_representation_or_residual": True,
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for condition, path in paths.items() if condition in sources
        },
    }
    (folder / "long_decode_selection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, len(indices), category, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
