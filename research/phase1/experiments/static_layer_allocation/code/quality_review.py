"""从完整输出复核收尾质量、累计前缀和最终候选停止，不生成报告。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.static_layer_allocation.code.protocol import digest, load_plan, write_json
from experiments.static_layer_allocation.code.quality import (
    RESULTS,
    RUNTIME,
    config_for,
    judge,
    next_round,
    result_path,
    score_rows,
    reference_path,
    reference_result,
)
from experiments.static_layer_allocation.code.quality_data import read, samples, totals
from kvstudy.paths import project_root
from phase1.execution import current_publication, validate_segments, verify_publication


def _outputs(root: Path, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """读取唯一累计原始输出，并核对实体校验。"""
    source = payload["raw_source"]
    path = root / source["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise RuntimeError("累计原始输出与来源校验不同")
    raw = read(path)
    ids = payload.get("raw_selection", {}).get("ids")
    selected = [row for row in raw["predictions"] if ids is None or row["id"] in ids]
    predictions = {row["id"]: row for row in selected}
    if len(selected) != payload["n"] or len(predictions) != payload["n"]:
        raise RuntimeError("累计原始输出包含缺失或重复身份")
    if payload.get("reused_prefix", {}).get("n", 0):
        raise RuntimeError("新增预测实体不应混入未审核的历史输出前缀")
    if set(predictions) != {row["id"] for row in payload["rows"]}:
        raise RuntimeError("原始输出缺失、重复或不符合累计身份")
    return predictions


def _diagnostics(value: dict[str, Any], task: str) -> dict[str, Any]:
    """保存分项、停止和缺答计数供结果审核；不增设质量阈值。"""
    rows = value["rows"]
    result = {
        "generated_n": len(rows), "all_request_score": value["score"],
        "invalid_samples": sum(row["invalid_groups"] > 0 for row in rows),
        "stops": dict(Counter(row["stop"] for row in rows)),
        "peak_reserved_bytes": max(
            (row["peak_reserved_bytes"] for row in rows if row.get("peak_reserved_bytes")),
            default=None,
        ),
    }
    if task == "retrieval_ruler":
        groups = {}
        for row in rows:
            key = f"depth={row['depth']},interference={row['interference']}"
            groups.setdefault(key, []).append(row)
        result["strata"] = {key: {"n": len(group), "em": score_rows(group, task)}
                            for key, group in groups.items()}
    if task == "hotpot_four":
        result["below_256_tokens"] = sum(row["new_tokens"] < 256 for row in rows)
        result["missing_answers_by_position"] = [
            sum(not row["extracted_answers"][i] for row in rows) for i in range(4)
        ]
        result["mean_f1_by_position"] = [
            sum(row["answer_f1"][i] for row in rows) / len(rows) for i in range(4)
        ]
    return result


def audit(root: Path) -> dict[str, Any]:
    """独立重算评分及各固定档位判断，核对全部正式来源后登记审核。"""
    from transformers import AutoTokenizer
    from phase1.score import answer_exact, gsm8k_exact, token_f1
    from phase1.models import MODELS, _eos_ids
    from phase1.inputs import _ids_and_source, _score_new_input

    plan = load_plan(root)["v_plan"]
    seeds = read(root / "research/phase1/splits/seeds.json")
    publication = current_publication(root)
    verify_publication(root, publication)
    admission = read(root / RUNTIME / "admission.json")
    if admission["status"] != "passed" or not plan["frozen"]:
        raise RuntimeError("缺少已冻结计划或候选准入")
    reuse = read(root / RESULTS / "reference_reuse.json")
    cleanup = read(root / RESULTS / "reference_cleanup.json")
    if reuse["status"] != "passed" or cleanup["status"] != "passed":
        raise RuntimeError("参考复用或重复实体清理尚未审核")
    for model in plan["candidates"]:
        for task in plan["tasks"]:
            for condition in ("bf16", "kv8", "k8v4"):
                duplicate = root / RESULTS / "final_quality" / model / f"{task}_{condition}.json"
                if duplicate.exists():
                    raise RuntimeError("静态分配结果仍含重复参考实体")
    all_results = {}
    source_index = {}
    diagnostics = {}
    counts = Counter()
    execution_device = None
    input_cache = {}
    for model in plan["candidates"]:
        tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model])
        eos = _eos_ids(tokenizer, model == "llama")
        all_results[model] = {}
        diagnostics[model] = {}
        for task, task_spec in plan["tasks"].items():
            all_results[model][task] = {}
            diagnostics[model][task] = {}
            for condition in ("bf16", "kv8", "k8v4", *plan["candidates"][model]):
                reference = condition in ("bf16", "kv8", "k8v4")
                outputs = {}
                base_n = 0
                path = result_path(root, model, task, condition)
                generated = read(path) if path.exists() else None
                if reference:
                    base_path = reference_path(root, model, task, condition)
                    base = read(base_path)
                    source = reuse["inventory"][f"{model}/{task}/{condition}"]["source"]
                    if source["sha256"] != hashlib.sha256(base_path.read_bytes()).hexdigest():
                        raise RuntimeError("复用首轮参考的实体发生改变")
                    base_n = len(base["rows"])
                    total = generated["total_n_with_reference"] if generated else base_n
                    value = reference_result(root, model, task, condition, total)
                    counts["reused_reference_requests"] += base_n
                    source_index.update(value["sources"])
                else:
                    if generated is None:
                        raise RuntimeError("缺少必要候选实体")
                    value = generated
                if generated is not None:
                    if (generated["protocol"] != plan["protocol"]
                            or generated["model"] != model or generated["task"] != task
                            or generated["condition"] != condition):
                        raise RuntimeError("正式结果的模型、任务或配置身份不同")
                    device = {key: item for key, item in generated["device"].items()
                              if key not in ("slurm_job", "visible_device")}
                    if execution_device is None:
                        execution_device = device
                    if device != execution_device or device["tf32"] or device["autocast"]:
                        raise RuntimeError("候选与新增参考的设备或数值执行条件不同")
                    if len(generated["rows"]) != generated["n"]:
                        raise RuntimeError("实体样本数与行数不同")
                    if generated["config_digest"] != digest(config_for(root, model, condition)):
                        raise RuntimeError("实际位宽配置与冻结名单不同")
                    validate_segments(root, generated)
                    if generated["output_limit"] != task_spec["max_new_tokens"]:
                        raise RuntimeError("输出预算发生改变")
                    outputs = _outputs(root, generated)
                    source_index[path.relative_to(root).as_posix()] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
                    counts["new_reference_requests" if reference
                           else "candidate_requests"] += generated["n"]
                if value["n"] not in totals(root, task):
                    raise RuntimeError("累计范围并非预先规定的档位")
                items = samples(root, model, task, value["n"])
                if task == "retrieval_ruler":
                    value["rows"] = [dict(row, depth=item["depth"],
                                          interference=item["interference"])
                                     for row, item in zip(value["rows"], items)]
                if [row["id"] for row in value["rows"]] != [item["id"] for item in items]:
                    raise RuntimeError("正式结果并非原冻结的累计前缀")
                for position, (item, row) in enumerate(zip(items, value["rows"])):
                    cache_key = (model, task, item["id"])
                    if cache_key not in input_cache:
                        input_cache[cache_key] = _ids_and_source(root, tokenizer, model, task, item)
                    ids, source_hash = input_cache[cache_key]
                    if len(ids) != row["input_tokens"] or source_hash != row["input_sha256"]:
                        raise RuntimeError("实际完整输入或分词与冻结值不同")
                    if position < base_n:
                        continue
                    output = outputs[item["id"]]
                    text = tokenizer.decode(output["token_ids"], skip_special_tokens=True)
                    if text != output["text"] or hashlib.sha256(text.encode()).hexdigest() != (
                        row["prediction_sha256"]
                    ):
                        raise RuntimeError("原始 token、文本和输出校验不一致")
                    if len(ids) != row["input_tokens"] or source_hash != row["input_sha256"]:
                        raise RuntimeError("实际完整输入或分词与冻结值不同")
                    if row["new_tokens"] != len(output["token_ids"]) or not (
                        1 <= row["new_tokens"] <= task_spec["max_new_tokens"]
                    ):
                        raise RuntimeError("实际输出长度无效")
                    if len(ids) + task_spec["max_new_tokens"] > 32768:
                        raise RuntimeError("输入加完整输出预算超过上下文限制")
                    if row["stop"] == "eos" and output["token_ids"][-1] not in eos:
                        raise RuntimeError("EOS 停止记录无效")
                    if (row["stop"] == "limit"
                            and row["new_tokens"] != task_spec["max_new_tokens"]):
                        raise RuntimeError("触及输出上限记录无效")
                    if row["stop"] not in ("eos", "limit", "invalid"):
                        raise RuntimeError("未识别的生成停止原因")
                    invalid = row["invalid_groups"]
                    if task == "gsm8k":
                        scoring = {"correct": not invalid and gsm8k_exact(text, item["answer"])}
                    elif task == "hotpot":
                        scoring = {"em": not invalid and answer_exact(text, item["answer"]),
                                   "f1": 0.0 if invalid else token_f1(text, item["answer"])}
                    else:
                        scoring = _score_new_input(task, text, item, invalid)
                    if any(row[key] != score for key, score in scoring.items()):
                        raise RuntimeError("独立重评分与正式结果不同")
                if generated is not None and score_rows(
                    generated["rows"], task
                ) != generated["score"]:
                    raise RuntimeError("独立汇总得分与正式结果不同")
                value["score"] = score_rows(value["rows"], task)
                all_results[model][task][condition] = value
                diagnostics[model][task][condition] = _diagnostics(value, task)
        selection_path = root / RESULTS / "expanded_quality" / model / "selection.json"
        selection = read(selection_path)
        if selection["status"] != "complete":
            raise RuntimeError("尚有必要扩样，不能完成收尾审核")
        for history in selection["history"]:
            for condition, tasks in history["cells"].items():
                for task, recorded in tasks.items():
                    count = recorded["requested_n"]
                    prefix = {}
                    for name in ("bf16", "kv8", condition):
                        value = all_results[model][task][name]
                        prefix[name] = {"n": count, "rows": value["rows"][:count]}
                    recomputed = judge(prefix["bf16"], prefix["kv8"], prefix[condition], task,
                                       plan["tasks"][task], seeds["bootstrap"],
                                       seeds["bootstrap_draws"], True)
                    if recomputed != recorded:
                        raise RuntimeError("固定档位质量区间或状态与独立复算不同")
            requests, stops = next_round(
                list(plan["candidates"][model]), history["cells"],
                {task: totals(root, task) for task in plan["tasks"]},
                plan["llama_priority"] if model == "llama" else None,
            )
            if requests != history["next_expansion"] or stops != history["stops"]:
                raise RuntimeError("固定档位扩样或停止决定不符合预登记规则")
        source_index[selection_path.relative_to(root).as_posix()] = hashlib.sha256(
            selection_path.read_bytes()).hexdigest()
    payload = {
        "status": "passed", "protocol": plan["protocol"], "counts": dict(counts),
        "execution_device": execution_device,
        "reference_cleanup_sha256": hashlib.sha256(
            (root / RESULTS / "reference_cleanup.json").read_bytes()).hexdigest(),
        "reference_reuse_sha256": hashlib.sha256(
            (root / RESULTS / "reference_reuse.json").read_bytes()).hexdigest(),
        "diagnostics": diagnostics,
        "results": source_index, "review_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "checks": ["new_prediction_rescoring", "verified_historical_reference_prefixes",
                   "no_duplicate_reference_entities", "frozen_inputs_and_tokenization",
                   "candidate_layer_bits", "fixed_checkpoint_intervals_and_stops",
                   "output_budget_and_eos", "source_publication_unchanged"],
    }
    write_json(root / RESULTS / "final_quality_review.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(audit(project_root()), ensure_ascii=False), flush=True)
