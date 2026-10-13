"""只读复核四个已完成实验的唯一实体、统计、依赖与报告。"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any
from unittest.mock import patch

from kvstudy.artifacts import read_json as _read, file_hash as _sha
from phase1.result_format import is_quality_result, normalize_quality_result


EXPERIMENTS = (
    "representation_interactions", "generation_error_sources",
    "static_layer_allocation", "storage_service_cost",
)
BASE = Path("research/phase1/experiments")


def _same(actual: Any, expected: Any, label: str) -> None:
    """浮点复算允许舍入误差；结构、类型和离散判定必须一致。"""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise RuntimeError(f"结构不一致：{label}")
        for key in expected:
            _same(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"行数不一致：{label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, f"{label}/{index}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=1e-10,
        ):
            raise RuntimeError(f"复算数值不一致：{label}")
    elif type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(f"记录不一致：{label}")


def _contained(computed: Any, saved: Any, label: str) -> None:
    """逐题可复算字段逐项核对，汇总附加的逐层审核记录另由来源校验。"""
    if isinstance(computed, dict):
        if not isinstance(saved, dict) or not computed.keys() <= saved.keys():
            raise RuntimeError(f"复算字段缺失：{label}")
        for key in computed:
            _contained(computed[key], saved[key], f"{label}/{key}")
    else:
        _same(computed, saved, label)


def check_hashes(root: Path, sources: dict[str, str], folder: Path | None = None) -> int:
    """核对当前实体的来源；历史快照不作为当前文件替代品。"""
    for relative, expected in sources.items():
        path = root / relative if relative.startswith("research/") else (folder or root) / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f"来源超出项目：{relative}")
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"来源校验失败：{relative}")
    return len(sources)


def check_quality(records: dict[str, Any]) -> dict[str, int]:
    """核对行身份与评分汇总，拒绝同条件重复行；未知历史停止保持 null。"""
    identities: dict[tuple[Any, ...], str] = {}
    files = rows_checked = unknown_stops = 0
    for path, value in records.items():
        if not is_quality_result(value):
            continue
        normalized = normalize_quality_result(value)
        rows = value["rows"]
        if len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError(f"同一结果内题号重复：{path}")
        config = value.get("condition", value.get("config", value.get("source")))
        config = json.dumps(config, sort_keys=True, ensure_ascii=False)
        for row in normalized["rows"]:
            if not row["input_sha256"] or not row["input_tokens"]:
                raise RuntimeError(f"缺少输入身份：{path}/{row['id']}")
            key = (
                value["model"], value["task"], config, value.get("residual_bits", 0),
                value.get("config_digest"), row["id"], row["input_sha256"],
            )
            if key in identities:
                raise RuntimeError(f"同条件重复测量：{identities[key]} 与 {path}")
            identities[key] = path
            unknown_stops += row["stop"] is None
        for name in ("em", "correct", "f1", "score"):
            if name not in value:
                continue
            if name == "f1":
                actual = sum(row["f1"] for row in rows) / len(rows)
            elif name == "score":
                from experiments.static_layer_allocation.code.quality import score_rows

                actual = score_rows(rows, value["task"])
            else:
                correct = sum(row.get("correct", row.get("em", False)) for row in rows)
                actual = correct if name == "correct" else 100 * correct / len(rows)
            _same(actual, value[name], f"{path}/{name}")
        files += 1
        rows_checked += len(rows)
    return {"files": files, "rows": rows_checked, "unknown_historical_stops": unknown_stops}


def _report(root: Path, experiment: str) -> dict[str, Any]:
    path = root / BASE / experiment / "report/index.md"
    text = path.read_text("utf-8")
    if f"experiment: {experiment}" not in text or "reviewed: true" not in text:
        raise RuntimeError(f"报告身份或审核标记缺失：{experiment}")
    headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
    if headings != ["实验目的", "实验结果与结论", "总结"]:
        raise RuntimeError(f"报告二级结构不一致：{experiment}")
    links = 0
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        destination = (path.parent / target.split("#", 1)[0]).resolve()
        if not destination.is_relative_to(root.resolve()) or not destination.exists():
            raise RuntimeError(f"报告链接失效：{experiment}/{target}")
        links += 1
    references = re.findall(r"`((?:research|docs)/[^`\n]+)`", text)
    for target in references:
        destination = (root / target).resolve()
        if not destination.is_relative_to(root.resolve()) or not destination.exists():
            raise RuntimeError(f"报告来源路径失效：{experiment}/{target}")
    images = {target for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)}
    unused = [p.name for p in path.parent.glob("*.png") if p.name not in images]
    if unused:
        raise RuntimeError(f"报告存在未引用图片：{experiment}/{unused}")
    return {"sha256": _sha(path), "local_links_checked": links,
            "project_source_paths_checked": len(references)}


def _representation(root: Path) -> dict[str, Any]:
    from experiments.representation_interactions.code import evaluate
    from experiments.representation_interactions.code.results import summarize_samples

    result = {}
    for model in ("qwen", "llama"):
        folder = root / BASE / "representation_interactions/results" / model
        summary, samples = _read(folder / "summary.json"), _read(folder / "samples.json")
        check_hashes(root, {"samples.json": summary["samples_file"]["sha256"]}, folder)
        settings = summary["paired_diagnostic"]["provenance"]["settings"]
        with patch.object(evaluate, "project_root", return_value=root):
            rebuilt = summarize_samples(
                samples, evaluate._expected_items(model),
                bootstrap_seed=settings["bootstrap_seed"],
                resamples=settings["bootstrap_resamples"],
            )
        _contained(rebuilt, summary["paired_diagnostic"]["metrics"], model)
        result[model] = {"paired_statistics_reproduced": True}
    return result


def _generation(root: Path) -> dict[str, Any]:
    from experiments.generation_error_sources.code.select_residual import _joint
    from experiments.generation_error_sources.code.review_trajectories import validate_trajectories
    from experiments.generation_error_sources.code.new_input_quality import judge_condition

    folder = root / BASE / "generation_error_sources/results"
    sources = _read(folder / "new_input_sources.json")
    check_hashes(root, {sources["summary"]["path"]: sources["summary"]["sha256"]})
    for entity in sources["entities"]:
        check_hashes(root, {entity["path"]: entity["sha256"]})
    review = _read(folder / "trajectory_review.json")
    if not review["reviewed"] or not review["complete"]:
        raise RuntimeError("传播审核未完成")
    for entity in review["sources"]:
        check_hashes(root, {entity["path"]: entity["sha256"]})
    seeds = _read(root / "research/phase1/splits/seeds.json")
    new = _read(folder / "new_input_quality.json")
    for model in ("qwen", "llama"):
        selected = _read(folder / model / "residual_selection.json")
        rebuilt = _joint(root, model, selected["commit"])
        for key in ("e3_representation", "e3_residual_bits", "cells", "categories", "ranking"):
            _same(rebuilt[key], selected[key], f"{model}/{key}")
        validate_trajectories(_read(folder / model / "trajectories.json"))
        for task, record in new["models"][model].items():
            bf16 = _read(folder / model / f"v_{task}_bf16.json")
            kv8 = _read(root / record["kv8"]["path"])
            for cell in record["candidates"].values():
                check_hashes(root, {cell["path"]: cell["sha256"]})
                candidate = _read(root / cell["path"])
                rebuilt = judge_condition(
                    bf16, kv8, candidate, task, seeds["bootstrap"], seeds["bootstrap_draws"],
                )
                _same(rebuilt, {key: cell[key] for key in rebuilt}, f"{model}/{task}")
    return {"joint_selection_reproduced": True, "new_input_intervals_reproduced": True,
            "trajectory_structure_and_reviewed_entities_verified": True}


def _static(root: Path) -> dict[str, Any]:
    from experiments.static_layer_allocation.code.quality import (
        judge, next_round, reference_result, result_path,
    )
    from experiments.static_layer_allocation.code.quality_data import totals
    from experiments.static_layer_allocation.code.numerics_summary import (
        budget_comparisons, path_summary,
    )

    folder = root / BASE / "static_layer_allocation/results"
    for name in ("review.json", "numerics_review.json", "final_quality_review.json"):
        value = _read(folder / name)
        if value["status"] != "passed":
            raise RuntimeError(f"静态分配审核未通过：{name}")
        check_hashes(root, value.get("source_files", value.get("results", {})), folder)
    full = _read(root / "research/phase1/configs/static_layer_allocation.json")
    design = _read(root / "research/phase1/configs/static_numerics.json")
    seeds = _read(root / "research/phase1/splits/seeds.json")
    plan = full["v_plan"]
    for model in ("qwen", "llama"):
        paths = sorted((folder / model / "numerical_questions").glob("*.json"))
        rows = [_read(path) for path in paths]
        summary = _read(folder / model / "numerics_summary.json")
        allocation = _read(folder / model / "numerical_allocations.json")
        _same(path_summary(rows), summary["path_means"], f"{model}/path_means")
        _same(budget_comparisons(rows, allocation, full, design),
              summary["budget_comparisons"], f"{model}/budget_comparisons")
        selected = _read(folder / "expanded_quality" / model / "selection.json")
        if selected["status"] != "complete" or selected["next_expansion"]:
            raise RuntimeError(f"仍需扩样：{model}")
        for history in selected["history"]:
            for condition, tasks in history["cells"].items():
                for task, recorded in tasks.items():
                    count = recorded["requested_n"]
                    values = {}
                    for name in ("bf16", "kv8", condition):
                        value = reference_result(root, model, task, name, count) if name in (
                            "bf16", "kv8",
                        ) else _read(result_path(root, model, task, name))
                        values[name] = {"n": count, "rows": value["rows"][:count]}
                    rebuilt = judge(values["bf16"], values["kv8"], values[condition], task,
                                    plan["tasks"][task], seeds["bootstrap"],
                                    seeds["bootstrap_draws"], True)
                    _same(rebuilt, recorded, f"{model}/{condition}/{task}")
            requests, stops = next_round(
                list(plan["candidates"][model]), history["cells"],
                {task: totals(root, task) for task in plan["tasks"]},
                plan["llama_priority"] if model == "llama" else None,
            )
            _same(requests, history["next_expansion"], f"{model}/next_expansion")
            _same(stops, history["stops"], f"{model}/stops")
    return {"numeric_path_and_budget_statistics_reproduced": True,
            "all_quality_checkpoints_and_stops_reproduced": True,
            "passed_raw_output_review_hashes_verified": True}


def _storage(root: Path) -> dict[str, Any]:
    from experiments.storage_service_cost.code.review import _row

    folder = root / BASE / "storage_service_cost/results"
    service, review = _read(folder / "service.json"), _read(folder / "service_review.json")
    if review["status"] != "passed" or _sha(folder / "service.json") != review["service_sha256"]:
        raise RuntimeError("服务模型审核实体失效")
    from phase1.execution import verify_recorded_code, workspace_root

    recorded_code = {}
    for relative, expected in service["source_sha256"].items():
        if relative.endswith(".py"):
            source = verify_recorded_code(root, relative, expected)
            recorded_code[relative] = source.relative_to(workspace_root(root)).as_posix()
        else:
            check_hashes(root, {relative: expected})
    check_hashes(root, _read(folder / "review.json")["results"])
    for model in ("qwen", "llama"):
        layout = _read(folder / model / "layout.json")
        selection = _read(root / BASE / "static_layer_allocation/results/expanded_quality"
                          / model / "selection.json")
        _same(layout["quality"], selection["cells"], f"{model}/layout_quality")
        for row in layout["rows"]:
            _row(layout, row)
        for condition, tasks in service["models"][model]["quality"].items():
            for task, cell in tasks.items():
                actual = selection["cells"][condition][task]
                _same(cell, {"state": actual["state"], "n": actual["n"]}, "service_quality")
    return {"layout_formulas_reproduced": True, "service_source_dependencies_verified": True,
            "recorded_code_sources_verified": recorded_code,
            "task_quality_states_verified": True, "saved_service_formula_audit_verified": True}


def audit(root: Path, experiment: str | None = None) -> dict[str, Any]:
    """复算已保存评分与统计；不加载模型、不写结果、不重新生成或重新作图。"""
    names = (experiment,) if experiment else EXPERIMENTS
    if any(name not in EXPERIMENTS for name in names):
        raise ValueError("未知实验")
    index = _read(root / "research/phase1/source_index.json")
    indexed = {entry["current_path"]: entry["current_sha256"] for entry in index["files"]}
    records, hashes = {}, {}
    for name in names:
        for path in sorted((root / BASE / name / "results").rglob("*.json")):
            relative = path.relative_to(root).as_posix()
            hashes[relative] = _sha(path)
            if indexed.get(relative) != hashes[relative]:
                raise RuntimeError(f"统一索引不符：{relative}")
            records[relative] = _read(path)
    expected = {path for path in indexed if any(path.startswith(f"{BASE.as_posix()}/{name}/")
                                               for name in names)}
    if expected != hashes.keys():
        raise RuntimeError("统一索引存在失效当前实体")
    duplicate = len(set(hashes.values())) != len(hashes)
    if duplicate:
        raise RuntimeError("当前结果存在逐字节重复实体")
    quality = check_quality(records)
    checks = {
        "representation_interactions": _representation,
        "generation_error_sources": _generation,
        "static_layer_allocation": _static,
        "storage_service_cost": _storage,
    }
    results = {name: {"checks": checks[name](root), "report": _report(root, name),
                      "result_files": sum(f"/{name}/results/" in path for path in records)}
               for name in names}
    if any(_sha(root / path) != digest for path, digest in hashes.items()):
        raise RuntimeError("审核期间正式结果发生变化")
    return {"status": "passed", "experiments": results, "formal_result_files": len(hashes),
            "quality": quality, "duplicate_result_files": 0, "result_hashes_unchanged": True,
            "generation_forward_calls": 0, "mode": "saved_results_read_only"}
