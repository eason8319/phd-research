"""复算独立映射、内容汇总、冻结预算与单层 logits 的实验审核。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.allocations import build_choices, cost, layer_bits
from experiments.static_layer_allocation.code.content_metrics import (
    additive_metrics, question_metrics, validation_summary,
)
from experiments.static_layer_allocation.code.metrics import rank_agreement, ranking
from experiments.static_layer_allocation.code.protocol import digest, load_plan


def read(path: Path) -> dict[str, Any]:
    """读取小型正式记录，不改原始测量。"""
    return json.loads(path.read_text("utf-8"))


def compare(actual: Any, expected: Any, context: str) -> None:
    """结构逐项核对，浮点只容许汇总版本产生的舍入差。"""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise RuntimeError("字段不一致：" + context)
        for key in expected:
            compare(actual[key], expected[key], context + "." + str(key))
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError("长度不一致：" + context)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, context + f"[{index}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=2e-13, atol=2e-15):
            raise RuntimeError("数值不一致：" + context)
    elif actual != expected:
        raise RuntimeError("身份或枚举不一致：" + context)


def code_check(root: Path, record: dict[str, Any]) -> None:
    """执行时的源码按校验值或已提交历史核对，LF/CRLF 显式兼容。"""
    for path, checksum in record["code_sha256"].items():
        target = root / path
        raw = target.read_bytes() if target.exists() else b""
        variants = (raw, raw.replace(b"\r\n", b"\n"),
                    raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        if checksum in {hashlib.sha256(value).hexdigest() for value in variants}:
            continue
        archive = root / "research/.runtime/static-key-content/execution_sources"
        archive = archive / record["commit"] / path
        if archive.exists() and hashlib.sha256(archive.read_bytes()).hexdigest() == checksum:
            continue
        raw = subprocess.run(["git", "show", record["commit"] + ":" + path],
                             cwd=root, check=True, capture_output=True).stdout
        variants = (raw, raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        if checksum not in {hashlib.sha256(value).hexdigest() for value in variants}:
            raise RuntimeError("执行源码不一致：" + path)


def check_model(
    root: Path, folder: Path, model: str, check_logits: bool, recompute_logits: bool = True,
) -> dict[str, Any]:
    """审核所有校准与验证题，不根据结果调整样本、排名或预算。"""
    plan = load_plan(root)
    dataset = read(root / plan["dataset"])
    compare(digest(dataset["rows"]), dataset["identity_digest"], "dataset identity digest")
    for item in dataset["rows"]:
        suffixes = [item["key"][4:], item["value"][4:]]
        suffixes += [part[4:] for pair in item["distractors"] for part in pair]
        compare(len(suffixes), len(set(suffixes)), "unique key and value content")
        if check_logits:
            for record in (item["models"][model], item.get("paired_models", {}).get(model)):
                if record is not None:
                    compare(hashlib.sha256((root / record["path"]).read_bytes()).hexdigest(),
                            record["sha256"], "frozen prompt")
    files = {name: read(folder / model / (name + ".json"))
             for name in ("calibration", "mapping", "probe", "allocations", "gate")}
    probe, allocation, gate = (files[name] for name in ("probe", "allocations", "gate"))
    identities = {row["id"]: row for row in dataset["rows"]}
    cal = [row["id"] for row in dataset["rows"] if row["role"] == "calibration"]
    val = [row["id"] for row in dataset["rows"] if row["role"] == "validation"]
    compare(len(cal), 16, "calibration n")
    compare(len(val), 16, "validation n")
    if set(cal) & set(val):
        raise RuntimeError("校准与验证题重合")
    layers = plan["models"][model]["layers"]
    for stage, payload in files.items():
        compare(payload["model"], model, stage + " model")
        compare(payload["plan_sha256"], digest(plan), stage + " plan")
        compare(payload["dataset_digest"], dataset["identity_digest"], stage + " inputs")
        compare(payload["dataset_sha256"], hashlib.sha256(
            (root / plan["dataset"]).read_bytes()).hexdigest(), stage + " manifest")
        code_check(root, payload)
    compare(probe["calibration_ids"], cal, "probe calibration")
    compare(gate["validation_ids"], val, "gate validation")
    compare(allocation["rank_calibration_ids"], cal, "allocation calibration")
    compare(allocation["probe_sha256"], digest(probe), "allocation probe")
    compare(gate["allocation_digest"], allocation["allocation_digest"], "gate allocation")
    allocation_body = {key: allocation[key] for key in (
        "model", "random_seeds", "budget_groups", "random_permutations", "allocations",
        "probe_sha256", "rank_calibration_ids",
    )}
    compare(allocation["allocation_digest"], digest(allocation_body), "frozen bits")
    math_ids = {row["id"] for row in read(root / "research/phase1/splits/gsm8k.json")["c"]
                if row["fold"] == 0}
    used = files["calibration"]["input_identity"]
    compare({row["id"] for row in used if row["kind"] == "math"}, math_ids, "math fit fold")
    compare([row["id"] for row in used if row["kind"] == "retrieval"], cal, "retrieval fit fold")
    if check_logits:
        params = files["calibration"]
        compare(hashlib.sha256((root / params["path"]).read_bytes()).hexdigest(),
                params["sha256"], "frozen offline parameters")
    seeds = read(root / plan["random_seeds_source"])[plan["random_seeds_key"]]
    compare(allocation["random_seeds"], seeds, "random seeds")
    for name, format_spec in plan["formats"].items():
        measured = probe["formats"][name]["layers"]
        for layer in range(layers):
            values = [row["formats"][name]["layers"][layer] for row in probe["samples"]]
            for key in ("sensitivity", "reconstruction", "k8v4_error", "k4v4_error"):
                compare(measured[layer][key], float(np.mean([row[key] for row in values])),
                        "layer question mean")
            for question in probe["samples"]:
                record = identities[question["id"]]["models"][model]
                frames = question["formats"][name]["frames"]
                compare([f["answer_index"] for f in frames], record["content_indices"],
                        "local content positions")
                compare([f["query_position"] for f in frames],
                        [record["tokens"] + i - 1 for i in record["content_indices"]],
                        "query prediction alignment")
                for key in ("sensitivity", "reconstruction", "k8v4_error", "k4v4_error"):
                    compare(question["formats"][name]["layers"][layer][key],
                            float(np.mean([f["layers"][layer][key] for f in frames])),
                            "position then question mean")
        choices, permutations = build_choices(measured, plan["models"][model]["k4_layers"], seeds)
        compare(allocation["random_permutations"][name], permutations, "random nesting")
        for budget, strategies in choices.items():
            costs = set()
            geometry = {
                "hidden_size": 3584 if model == "qwen" else 4096,
                "num_attention_heads": 28 if model == "qwen" else 32,
                "num_key_value_heads": 4 if model == "qwen" else 8,
                "num_hidden_layers": layers,
            }
            parts = format_spec["representation"].split('-')
            for strategy, selected in strategies.items():
                spec = allocation["allocations"][f"{name}_{budget}_{strategy}"]
                compare(spec["k4_layers"], selected, "ranking selected layers")
                compare(spec["layer_bits"], layer_bits(layers, selected), "frozen bit table")
                compare(spec["actual_int8_units"], layers - len(selected), "budget units")
                compare(spec["p_int8"], (layers - len(selected)) / (2 * layers), "budget ratio")
                for length in plan['cost_lengths']:
                    compare(spec['cost'][str(length)], cost(
                        spec['layer_bits'], geometry, {'key_mode': parts[1], 'rope': parts[2]},
                        format_spec['residual_bits'], length,
                    ), 'independent full cost')
                costs.add(digest(spec["cost"]))
            compare(len(costs), 1, "same budget full bytes")
    for row in gate["rows"]:
        item = identities[row["id"]]
        rec = item["models"][model]
        compare(row["role"], "validation", "held out role")
        compare(row["input_sha256"], rec["sha256"], "actual input")
        compare(row["content_indices"], rec["content_indices"], "actual content")
        compare(row["content_target_ids"], [rec["answer_ids"][i] for i in rec["content_indices"]],
                "teacher forcing targets")
        compare(set(row["schemes"]), {"bf16", "kv8", *allocation["allocations"]},
                "all six strategies and references")
        compare(row["query_positions"],
                [rec["tokens"] + i - 1 for i in rec["content_indices"]], "decode positions")
        for name, spec in allocation["allocations"].items():
            alias = allocation["allocations"][spec["alias"]]
            compare(spec["layer_bits"], alias["layer_bits"], "reused bit configuration")
            compare(row["schemes"][name], row["schemes"][spec["alias"]], "reused measurement")
        for name, scheme in row["schemes"].items():
            compare(scheme["content_tokens"], len(rec["content_indices"]), "content token count")
            compare(scheme["free_exact"], int(scheme["prediction"] == item["value"]), "free EM")
            for key in scheme["frames"][0]:
                compare(scheme[key], float(np.mean([f[key] for f in scheme["frames"]])),
                        "content question mean")
            compare(scheme["first_content"], scheme["frames"][0], "first content")
            if scheme["kl_bf16"] < 0 or scheme["nll"] < 0:
                raise RuntimeError("NLL 或 KL 为负")
    summary = validation_summary(gate["rows"], allocation["budget_groups"], seeds)
    for key, value in summary.items():
        compare(gate[key], value, "gate summary")
    compare(files["mapping"]["n_pairs"], 8, "fixed mapping pairs")
    mapping = files["mapping"]
    compare(mapping["both_follow_mapping"], sum(row["both_follow_mapping"]
                                               for row in mapping["rows"]), "mapping pairs")
    for field, case in (("original_exact", "original"), ("swapped_exact", "swapped")):
        compare(mapping[field], sum(row["cases"][case]["exact"] for row in mapping["rows"]),
                "mapping total exact")
    compare(mapping["legacy_target_removed_copies"], sum(
        row["cases"]["legacy_target_removed"]["copies_old_answer"] for row in mapping["rows"]
    ), "legacy copy count")
    for row in mapping["rows"]:
        item = identities[row["id"]]
        for case, answer in (("original", item["value"]), ("swapped", item["paired_value"])):
            measured = row["cases"][case]
            compare(measured["exact"], measured["prediction"] == answer, "mapping exact")
        compare(row["cases"]["original"]["tokens"], row["cases"]["swapped"]["tokens"],
                "same length mapping pair")
        compare(row["both_follow_mapping"], row["cases"]["original"]["exact"]
                and row["cases"]["swapped"]["exact"], "mapping follows both values")
        control = row["cases"]["legacy_target_removed"]
        compare(control["copies_old_answer"], control["prediction"] == control["original_answer"],
                "legacy copy classification")
        if check_logits:
            compare(hashlib.sha256((root / control["path"]).read_bytes()).hexdigest(),
                    control["sha256"], "legacy control prompt")
    trace_path = folder / model / "free_token_diagnostic.json"
    trace_verified = False
    if trace_path.exists():
        trace = read(trace_path)
        question = next(row for row in gate["rows"] if row["key"] == trace["key"])
        compare(trace["input_sha256"], question["input_sha256"], "free trace input")
        compare(trace["main_execution_commit"], gate["commit"], "free trace main source")
        for name, measured in trace["schemes"].items():
            compare(measured["prediction"], question["schemes"][name]["prediction"],
                    "free trace reproduced answer")
            frames = measured["teacher_forced"]
            nll = float(np.mean([frames[i]["target"]["nll"]
                                 for i in trace["content_indices"]]))
            compare(nll, question["schemes"][name]["nll"], "free trace content loss")
            for i, current in enumerate(measured["frames"]):
                compare(measured["token_ids"][i], current["top"][0]["id"], "greedy trace")
                compare(current["correct_token_prefix"],
                        measured["token_ids"][:i] == trace["answer_ids"][:i], "trace prefix")
        source = root / "research/phase1/experiments/static_layer_allocation/code"
        source = source / "free_token_diagnostic.py"
        if not source.exists():
            source = root / "research/.runtime/static-key-content/free_token_diagnostic.py"
        compare(hashlib.sha256(source.read_bytes()).hexdigest(), trace["diagnostic_code_sha256"],
                "free trace source")
        trace_verified = True
    verified_logits = 0
    if model == plan["single_layer_diagnostic"]["model"]:
        diagnostic = read(folder / model / "single_layer.json")
        code_check(root, diagnostic)
        compare(diagnostic["plan_sha256"], digest(plan), "diagnostic plan")
        compare([row["id"] for row in diagnostic["rows"]], val, "single layer validation")
        layer_rows = []
        for layer in range(layers):
            selected = [row["single_layers"][layer] for row in diagnostic["rows"]]
            matched = [float(np.mean([frame["layers"][layer]["sensitivity"]
                                      for frame in row["matched_local_frames"]]))
                       for row in diagnostic["rows"]]
            layer_rows.append({"layer": layer, "invalid_groups": 0,
                               "matched_local_sensitivity": float(np.mean(matched)),
                               **{key: float(np.mean([row[key] for row in selected]))
                                  for key in ("delta_nll", "kl_baseline", "kl_bf16")}})
        compare(diagnostic["layers"], layer_rows, "single layer question mean")
        format_name = diagnostic["format"]
        local_rank = probe["formats"][format_name]["rankings"]["sensitivity"]
        agreement = {"calibration_local_vs_" + key: rank_agreement(
            local_rank, ranking(layer_rows, key, descending=True), layers // 2
        ) for key in ("delta_nll", "kl_baseline", "matched_local_sensitivity")}
        matched_rank = ranking(layer_rows, "matched_local_sensitivity", descending=True)
        agreement.update({"matched_local_vs_" + key: rank_agreement(
            matched_rank, ranking(layer_rows, key, descending=True), layers // 2
        ) for key in ("delta_nll", "kl_baseline")})
        compare(diagnostic["rank_agreement"], agreement, "single layer ranks")
        additive_means = {name: {key: float(np.mean([
            row["additive"][name][key] for row in diagnostic["rows"]
        ])) for key in ("actual_to_additive_kl", "sum_single_layer_kl",
                        "kl_sum_minus_actual", "additive_nll_minus_actual",
                        "centered_logit_residual_energy_fraction")}
                          for name in diagnostic["rows"][0]["additive"]}
        compare(diagnostic["additive_means"], additive_means, "additive question mean")
        if check_logits:
            import torch

            for row in diagnostic["rows"]:
                source = row["logits_source"]
                path = root / source["path"]
                compare(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"], "logits")
                if not recompute_logits:
                    verified_logits += 1
                    continue
                packed = torch.load(path, weights_only=False, map_location="cpu")
                record = identities[row["id"]]["models"][model]
                compare(packed["input_sha256"], record["sha256"], "logits input")
                compare(packed["content_indices"], record["content_indices"], "logits positions")
                compare(packed["targets"], [record["answer_ids"][i]
                                           for i in record["content_indices"]], "logits targets")
                values = {name: tensor.numpy() for name, tensor in packed["logits"].items()}
                baseline = values[diagnostic["baseline"]]
                reference = values["bf16"]
                singles = {i: values[f"single_layer_{i}"] for i in range(layers)}
                for layer, z in singles.items():
                    expected = {"layer": layer, **question_metrics(
                        z, packed["targets"], reference, baseline
                    )}
                    compare(row["single_layers"][layer], expected, "single logits")
                for name, measured in row["additive"].items():
                    spec = allocation["allocations"][name]
                    compare(measured, additive_metrics(baseline, singles, values[name],
                                                       spec["k4_layers"], packed["targets"],
                                                       reference), "additive logits")
                verified_logits += 1
    return {"status": "passed", "calibration_n": len(cal), "validation_n": len(val),
            "budget_workpoints": 9, "strict_random_wins": sum(
                result["sensitivity_strictly_below_all_random"]
                for budgets in gate["budget_comparisons"].values() for result in budgets.values()
            ), "full_logits_questions_verified": verified_logits,
            "mapping_both_follow": mapping["both_follow_mapping"],
            "free_token_trace_verified": trace_verified,
            "legacy_complete_answer_matches": mapping["legacy_target_removed_copies"],
            "legacy_shared_suffix_questions": sum(
                row["cases"]["legacy_target_removed"]["original_answer"][4:]
                in row["cases"]["legacy_target_removed"]["prediction"]
                for row in mapping["rows"]
            )}


def run(folder: Path, check_logits: bool) -> None:
    """保存审核状态及源文件校验，供审核后作图和报告引用。"""
    root = project_root()
    if not folder.is_absolute():
        folder = root / folder
    paths = [p for model in ("qwen", "llama") for p in sorted((folder / model).glob('*.json'))
             if not p.name.endswith('_progress.json')]
    sources = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in paths}
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    previous_path = folder / "review.json"
    previous = read(previous_path) if previous_path.exists() else {}
    reuse = (check_logits and previous.get("status") == "passed"
             and previous.get("logits_checked") is True
             and previous.get("review_code_sha256") == code_sha
             and previous.get("models", {}).get("llama", {}).get(
                 "full_logits_questions_verified") == 16
             and all(sources.get(path) == checksum
                     for path, checksum in previous.get("source_files", {}).items())
             and set(sources) - set(previous.get("source_files", {}))
             <= {"qwen/free_token_diagnostic.json"})
    models = {name: check_model(root, folder, name, check_logits, not reuse)
              for name in ("qwen", "llama")}
    payload = {"status": "passed", "date": "2026-10-09", "models": models,
               "statistical_unit": "question", "logits_checked": check_logits,
               "source_files": sources, "review_code_sha256": code_sha,
               "logit_recomputation_reused": bool(reuse)}
    (folder / "review.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)
                                         + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path,
                        default=Path('research/phase1/experiments/static_layer_allocation/results'))
    parser.add_argument('--check-logits', action='store_true')
    args = parser.parse_args()
    run(args.folder, args.check_logits)
