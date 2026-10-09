"""复算静态 Key 分配的正式结果，审核输入身份、冻结分配、成本及逐题汇总。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.allocations import build_choices, cost, layer_bits
from experiments.static_layer_allocation.code.metrics import (
    distribution_summary, rank_agreement, ranking,
)
from experiments.static_layer_allocation.code.protocol import (
    SCHEMA, digest, frozen_sources, load_plan, write_json,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise RuntimeError(f"结果复算不一致：{context}")


def _verify_code(root: Path, path: str, checksum: str, commit: str) -> None:
    target = root / path
    if target.exists():
        _assert_equal(hashlib.sha256(target.read_bytes()).hexdigest(), checksum, "code " + path)
        return
    raw = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=root, check=True, capture_output=True
    ).stdout
    variants = (raw, raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    if checksum not in {hashlib.sha256(value).hexdigest() for value in variants}:
        raise RuntimeError("历史执行源码与已登记校验值不一致：" + path)


def _question_mean(values: list[float]) -> float:
    """复现执行环境 Python 3.11 的逐项 FP64 累加，避免版本改变 sum 的舍入。"""
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def _probe_summary(samples: list[dict[str, Any]], layers: int) -> dict[str, Any]:
    subsets = {}
    ranks = {}
    for kind in ("math", "retrieval"):
        selected = [row for row in samples if row["kind"] == kind]
        rows = []
        for layer in range(layers):
            measured = [sample["layers"][layer] for sample in selected]
            row = {
                "layer": layer, "samples": len(measured),
                "zero_heads": sum(value["zero_heads"] for value in measured),
                "invalid_groups": sum(value["invalid_groups"] for value in measured),
            }
            for metric in ("sensitivity", "reconstruction", "k8v4_error", "k4v4_error"):
                defined = [value[metric] for value in measured if value[metric] is not None]
                row[metric] = (
                    _question_mean(defined) if defined and not row["invalid_groups"] else None
                )
                row[metric + "_valid_questions"] = len(defined)
            rows.append(row)
        subsets[kind] = rows
        ranks[kind] = {
            metric: ranking(rows, metric, descending=True)
            for metric in ("sensitivity", "reconstruction")
        }
    agreements = {
        "math_vs_retrieval_sensitivity": rank_agreement(
            ranks["math"]["sensitivity"], ranks["retrieval"]["sensitivity"], layers // 2
        ),
        "retrieval_sensitivity_vs_reconstruction": rank_agreement(
            ranks["retrieval"]["sensitivity"], ranks["retrieval"]["reconstruction"], layers // 2
        ),
    }
    return {"layers": subsets, "rankings_descending": ranks, "rank_agreement": agreements}


def _format_config(spec: dict[str, Any]) -> dict[str, Any]:
    parts = spec["representation"].split("-")
    return {"key_mode": parts[1], "rope": parts[2]}


def review_model(root: Path, model_key: str, plan: dict[str, Any]) -> dict[str, Any]:
    """复核一个模型的全 C、三格式、三预算，不根据结果更改排名或质量规则。"""
    folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    probe, allocation, gate = (_read(folder / (name + ".json"))
                                for name in ("probe", "allocations", "gate"))
    layers = plan["models"][model_key]["layers"]
    seeds = _read(root / plan["random_seeds_source"])[plan["random_seeds_key"]]
    for name, payload in (("probe", probe), ("allocations", allocation), ("gate", gate)):
        _assert_equal(payload["schema"], SCHEMA, name + " schema")
        _assert_equal(payload["model"], model_key, name + " model")
        _assert_equal(payload["plan_sha256"], digest(plan), name + " plan")
        _assert_equal(payload["frozen_sources"], frozen_sources(root, model_key), name + " sources")
        for path, checksum in payload["code_sha256"].items():
            _verify_code(root, path, checksum, payload["commit"])
    _assert_equal(probe["code_sha256"], gate["code_sha256"], "probe / gate code")
    _assert_equal(probe["commit"], allocation["commit"], "probe / allocation commit")
    _assert_equal(probe["commit"], gate["commit"], "probe / gate commit")
    _assert_equal(probe["counts"], plan["counts"], "probe counts")
    _assert_equal(gate["counts"], plan["counts"], "gate counts")
    _assert_equal(probe["input_digest"], digest(probe["input_identity"]), "input digest")
    _assert_equal(allocation["input_digest"], probe["input_digest"], "allocation inputs")
    _assert_equal(gate["input_digest"], probe["input_digest"], "gate inputs")
    _assert_equal(allocation["probe_sha256"], digest(probe), "probe freeze")
    _assert_equal(allocation["allocation_digest"],
                  digest({k: v for k, v in allocation.items() if k != "allocation_digest"}),
                  "allocation freeze")
    _assert_equal(gate["allocation_digest"], allocation["allocation_digest"], "gate freeze")
    _assert_equal(allocation["random_seeds"], seeds, "random seeds")
    _assert_equal(allocation["n_layers"], layers, "layers")
    registry = _read(root / (
        "research/phase1/experiments/generation_error_sources/results/calibration.json"
    ))
    calibration = {**registry["files"][model_key], "commit": registry["commit"],
                   "protocol": registry["protocol"]}
    for payload in (probe, allocation, gate):
        _assert_equal(payload["calibration_source"], calibration, "calibration provenance")
    identities = probe["input_identity"]
    counts = {kind: sum(row["kind"] == kind for row in identities) for kind in plan["counts"]}
    _assert_equal(counts, plan["counts"], "input identities counts")
    if len({row["id"] for row in identities}) != sum(counts.values()):
        raise RuntimeError("冻结题目身份重复")
    specs = allocation["allocations"]
    model = allocation["model_geometry"]
    _assert_equal(model["num_hidden_layers"], layers, "model geometry")
    duplicates = {}
    formats = {}
    for format_name, format_spec in plan["formats"].items():
        measured = probe["formats"][format_name]
        for key, value in format_spec.items():
            _assert_equal(measured[key], value, "probe format " + key)
        samples = measured["samples"]
        _assert_equal(len(samples), len(identities), "probe sample count")
        for sample, identity in zip(samples, identities):
            _assert_equal({key: sample[key] for key in identity}, identity, "probe input")
            _assert_equal(len(sample["layers"]), layers, "probe layers")
            _assert_equal(sample["query_position"], sample["input_tokens"], "query position")
            for i, row in enumerate(sample["layers"]):
                _assert_equal(row["layer"], i, "layer order")
                _assert_equal(row["invalid_groups"], 0, "invalid probe quantization")
                for metric in ("sensitivity", "reconstruction", "k8v4_error", "k4v4_error"):
                    if row[metric] is not None and (
                            not math.isfinite(row[metric]) or row[metric] < 0):
                        raise RuntimeError("探针误差不是非负有限数")
                if format_spec["residual_bits"] and sample["input_tokens"] + 1 <= 128:
                    _assert_equal(row["sensitivity"], 0.0, "fully residual local change")
                    _assert_equal(row["k8v4_error"], row["k4v4_error"], "same residual output")
        for key, recomputed in _probe_summary(samples, layers).items():
            _assert_equal(measured[key], recomputed, "probe " + key)
        choices, permutations = build_choices(
            measured["layers"]["retrieval"], plan["models"][model_key]["k4_layers"], seeds
        )
        format_allocation = allocation["formats"][format_name]
        _assert_equal(format_allocation["random_permutations"], permutations, "nested random")
        _assert_equal(format_allocation["existing_quality"], measured["existing_quality"],
                      "retained quality status")
        expected_names = []
        canonical = {}
        for budget, strategies in choices.items():
            names = []
            for strategy, selected in strategies.items():
                name = f"{format_name}_{budget}_{strategy}"
                names.append(name)
                expected_names.append(name)
                row = specs[name]
                assignments = layer_bits(layers, selected)
                _assert_equal(row["k4_layers"], sorted(selected), name + " layers")
                _assert_equal(row["layer_bits"], assignments, name + " bits")
                _assert_equal(row["actual_int8_units"], layers - len(selected), name + " int8")
                _assert_equal(row["p_int8"], (layers - len(selected)) / (2 * layers),
                              name + " int8 proportion")
                _assert_equal(row["nominal_bits"], 4 + 2 * (layers - len(selected)) / layers,
                              name + " nominal bits")
            _assert_equal(format_allocation["budget_groups"][budget], names, "budget group")
        expected_names += [format_name + "_k8v4", format_name + "_kv4"]
        for name in expected_names:
            row = specs[name]
            if row["strategy"] in ("k8v4", "kv4"):
                selected = [] if row["strategy"] == "k8v4" else list(range(layers))
                _assert_equal(row["layer_bits"], layer_bits(layers, selected), name + " control")
                _assert_equal(row["k4_layers"], selected, name + " control layers")
            _assert_equal(row["r"], len(row["k4_layers"]), name + " budget")
            signature = digest(row["layer_bits"])
            _assert_equal(row["layer_bits_sha256"], signature, name + " signature")
            canonical.setdefault(signature, name)
            _assert_equal(row["measurement_alias"], canonical[signature], name + " alias")
            if canonical[signature] != name:
                duplicates[name] = canonical[signature]
            for length in plan["cost_lengths"]:
                recomputed = cost(row["layer_bits"], model, _format_config(format_spec),
                                  format_spec["residual_bits"], length)
                _assert_equal(row["cost"][str(length)], recomputed, name + " cost")
        formats[format_name] = {
            "probe_questions": len(samples), "named_allocations": len(expected_names),
            "unique_allocations": len(canonical),
            "fully_residual_math_questions": sum(
                row["kind"] == "math" and row["input_tokens"] + 1 <= 128 for row in samples
            ) if format_spec["residual_bits"] else 0,
        }
    _assert_equal(len(specs), 60, "all named configurations")
    _assert_equal(allocation["duplicate_configurations"], duplicates, "duplicate map")
    for row, identity in zip(gate["rows"], identities):
        _assert_equal({key: row[key] for key in identity}, identity, "gate actual input")
        _assert_equal(set(row["kl"]), {"bf16", "kv8", *specs}, "gate configurations")
        _assert_equal(row["query_position"], row["input_tokens"], "gate query position")
        _assert_equal(row["kl"]["bf16"], 0.0, "zero reference KL")
        for name, value in row["kl"].items():
            if value is None or not math.isfinite(value) or value < 0:
                raise RuntimeError("KL 有缺测、非有限或负值")
            _assert_equal(row["invalid_groups"][name], 0, "invalid candidate groups")
        for alias, name in duplicates.items():
            _assert_equal(row["kl"][alias], row["kl"][name], "reused configuration")
        if row["kind"] == "math" and row["input_tokens"] + 1 <= 128:
            values = {value for name, value in row["kl"].items()
                      if name.startswith("offline_pre_")}
            _assert_equal(len(values), 1, "fully residual full-layer invariance")
    _assert_equal(len(gate["rows"]), len(identities), "gate rows")
    for name, value in distribution_summary(
            gate["rows"], allocation["formats"], seeds).items():
        _assert_equal(gate[name], value, "gate " + name)
    _assert_equal(gate["quality_threshold"], None, "no quality threshold")
    _assert_equal(gate["v_plan_frozen"], False, "pending V plan")
    for length in plan["cost_lengths"]:
        expected = cost([{"k": 8, "v": 8} for _ in range(layers)], model,
                        {"key_mode": "per_token", "rope": "post"}, 0, length)
        _assert_equal(allocation["reference_cost"]["kv8"][str(length)], expected, "KV8 cost")
        bf16 = 4 * layers * model["num_key_value_heads"] * length \
            * (model["hidden_size"] // model["num_attention_heads"])
        _assert_equal(allocation["reference_cost"]["bf16"][str(length)]["resident_bytes"],
                      bf16, "BF16 cost")
    for index, identity in enumerate(identities):
        tokens = {probe["formats"][name]["samples"][index]["query_token"]
                  for name in plan["formats"]}
        _assert_equal(tokens, {gate["rows"][index]["query_token"]}, "common BF16 token")
    numerical = _read(folder / "numerical_check.json")
    _assert_equal(numerical["status"], "passed", "numerical check")
    _assert_equal(numerical["gate_sha256"],
                  hashlib.sha256((folder / "gate.json").read_bytes()).hexdigest(),
                  "numerical gate source")
    _assert_equal(numerical["allocation_digest"], allocation["allocation_digest"],
                  "numerical allocation source")
    _assert_equal(numerical["input_digest"], gate["input_digest"], "numerical input source")
    _assert_equal(numerical["numerical_code_sha256"], hashlib.sha256(
        (root / "research/phase1/experiments/static_layer_allocation/code/numerical_check.py")
        .read_bytes()).hexdigest(), "numerical source")
    retrieval_rows = [row for row in gate["rows"] if row["kind"] == "retrieval"]
    selected = {retrieval_rows[0]["id"], retrieval_rows[-1]["id"],
                max(retrieval_rows, key=lambda row: max(row["kl"].values()))["id"]}
    _assert_equal({row["id"] for row in numerical["checked_questions"]}, selected,
                  "fixed numerical questions")
    _assert_equal(numerical["checked_n"], len(selected), "numerical n")
    _assert_equal(numerical["reference_n"], len(retrieval_rows), "reference n")
    _assert_equal([row["id"] for row in numerical["reference_distributions"]],
                  [row["id"] for row in retrieval_rows], "reference order")
    max_error = 0.0
    for row in numerical["checked_questions"]:
        original = next(value for value in retrieval_rows if value["id"] == row["id"])
        names = {"bf16", "kv8", *(value["measurement_alias"] for value in specs.values())}
        _assert_equal(set(row["measurements"]), names, "all numerical configurations")
        for name, values in row["measurements"].items():
            _assert_equal(values["stable_float64_kl"], original["kl"][name],
                          "recomputed candidate KL")
            error = abs(values["stable_float64_kl"] - values["direct_longdouble_kl"])
            _assert_equal(values["absolute_error_longdouble"], error, "direct error")
            if error > 100 * 2.220446049250313e-16 * max(1.0, values["stable_float64_kl"]):
                raise RuntimeError("扩展精度直接 KL 核验不通过")
            max_error = max(max_error, error)
    _assert_equal(numerical["max_absolute_error_longdouble"], max_error, "max direct error")
    for row, original in zip(numerical["reference_distributions"], retrieval_rows):
        _assert_equal(row["query_token"], original["query_token"], "numerical token")
        if not 0 <= row["maximum_probability"] <= 1 or row["entropy_nat"] < 0:
            raise RuntimeError("参考分布统计无效")
    return {
        "status": "passed", "execution_commit": probe["commit"], "counts": counts,
        "formats": formats, "duplicate_configurations": duplicates,
        "comparisons": gate["budget_comparisons"],
        "minimum_kl": min(value for row in gate["rows"] for value in row["kl"].values()),
        "maximum_kl": max(value for row in gate["rows"] for value in row["kl"].values()),
        "code_sha256": probe["code_sha256"],
        "numerical_check_max_error": max_error,
        "checks": [
            "frozen_input_and_code_provenance", "same_bf16_token_all_conditions",
            "group_then_question_aggregation", "rankings_and_overlap",
            "retrieval_only_allocation", "nested_shared_random_permutations",
            "equal_budget_full_cost", "duplicate_reuse", "nonnegative_finite_kl",
            "per_question_pairs_and_all_budget_summaries", "no_quality_threshold",
        ],
    }


def run() -> None:
    """两模型均通过复算后保存正式审核文件，供作图和报告引用。"""
    root = project_root()
    plan = load_plan(root)
    models = {name: review_model(root, name, plan) for name in plan["models"]}
    folder = root / "research/phase1/experiments/static_layer_allocation/results"
    paths = [folder / name / (stage + ".json")
             for name in models
             for stage in ("probe", "allocations", "gate", "numerical_check")]
    write_json(folder / "review.json", {
        "schema": SCHEMA, "status": "passed", "date": "2026-10-09",
        "statistical_unit": "question", "models": models,
        "source_files": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "review_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_threshold": None, "generation_screen": "not_frozen", "v_plan_frozen": False,
    })


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
