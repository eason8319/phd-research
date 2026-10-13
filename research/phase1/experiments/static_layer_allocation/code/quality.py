"""按冻结三候选执行独立生成、配对质量判定与有上限扩样。"""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any

from experiments.static_layer_allocation.code.protocol import digest, load_plan, write_json
from experiments.static_layer_allocation.code.quality_data import read, samples, totals
from phase1.execution import current_publication, verify_publication, validate_segments
from phase1.results import ResultCatalog, quality_path, reference_view
from phase1.reuse import (
    assess_reuse,
    computation_signature,
    file_hash,
    reference_path as shared_reference_path,
)


RESULTS = "research/phase1/experiments/static_layer_allocation/results"
RUNTIME = "research/.runtime/static-final"


def config_for(root: Path, model: str, condition: str) -> dict[str, Any] | None:
    """从冻结候选生成层表，逐项核对原校准分配。"""
    from phase1.generation import _config

    plan = load_plan(root)
    v = plan["v_plan"]
    if condition == "bf16":
        return None
    if condition == "kv8":
        return _config(v["references"]["kv8"])
    if condition == "k8v4":
        return _config(v["references"]["same_format_k8v4"][model])
    spec = v["candidates"][model][condition]
    config = _config(spec["representation"])
    k4 = spec["k4_layers"]
    if k4 is not None:
        allocation = read(root / RESULTS / model / "allocations.json")["allocations"][condition]
        if sorted(allocation["k4_layers"]) != k4:
            raise RuntimeError("候选层名单与原校准冻结名单不同")
        config["layer_bits"] = allocation["layer_bits"]
        if any(row["v"] != 4 for row in config["layer_bits"]):
            raise RuntimeError("候选 Value 并非全层 INT4")
        expected = [i for i, row in enumerate(config["layer_bits"]) if row["k"] == 4]
        if expected != k4 or len(config["layer_bits"]) != plan["models"][model]["layers"]:
            raise RuntimeError("候选实际位宽表不一致")
    return config


def score_rows(rows: list[dict[str, Any]], task: str) -> float:
    """题目或请求等权平均，统一返回 0–100 分。"""
    if not rows:
        raise ValueError("有效统计单位为空")
    if task in ("hotpot", "hotpot_four"):
        return sum(row["f1"] for row in rows) / len(rows)
    return 100 * sum(row["correct"] for row in rows) / len(rows)


def judge(
    reference: dict[str, Any], kv8: dict[str, Any], candidate: dict[str, Any],
    task: str, spec: dict[str, Any], seed: int, draws: int, final_stage: bool,
) -> dict[str, Any]:
    """按共同 BF16 准入子集计算两参考下降；缺失区间记证据不足。"""
    from phase1.quality_stats import _interval, _paired
    from kvstudy.intervals import class_unit_passes, quality_state

    ids = [row["id"] for row in reference["rows"]]
    if task == "hotpot_four":
        ids = [row["id"] for row in reference["rows"]
               if row["new_tokens"] >= spec["baseline_min_tokens"]]
    views = []
    for result in (reference, kv8, candidate):
        indexed = {row["id"]: row for row in result["rows"]}
        if len(indexed) != len(reference["rows"]) or set(indexed) != {
            row["id"] for row in reference["rows"]
        }:
            raise RuntimeError("质量配对样本身份不一致")
        views.append({"n": len(ids), "rows": [indexed[item_id] for item_id in ids]})
    if not ids or score_rows(views[0]["rows"], task) < spec["baseline_minimum"]:
        return {"state": "基线缺口", "n": len(ids), "requested_n": reference["n"],
                "reason": "BF16 长度准入为空或任务得分低于冻结下限"}
    interval_task = "hotpot" if task in ("hotpot", "hotpot_four") else task
    b, k, a = views
    versus_bf16 = _paired(b, a, interval_task, seed, draws)
    versus_kv8 = _paired(k, a, interval_task, seed, draws)
    invalid = sum(row.get("invalid_groups", 0) > 0 for row in candidate["rows"])
    state = quality_state(_interval(versus_bf16), _interval(versus_kv8),
                          format_ok=invalid < 2)
    if state != "淘汰" and any(value["upper"] is None
                              for value in (versus_bf16, versus_kv8)):
        state = "证据不足"
    return {
        "state": state, "n": len(ids), "requested_n": reference["n"], "paired_ids": ids,
        "score": score_rows(a["rows"], task), "bf16_score": score_rows(b["rows"], task),
        "kv8_score": score_rows(k["rows"], task), "versus_bf16": versus_bf16,
        "versus_kv8": versus_kv8, "invalid_samples": invalid,
        "class_unit_passes": class_unit_passes(
            _interval(versus_bf16), _interval(versus_kv8), format_ok=invalid < 2),
        "stops": dict(Counter(row["stop"] for row in candidate["rows"])),
        "missing_answers": sum(row.get("missing_answers", 0) for row in candidate["rows"]),
        "below_256_tokens": sum(row["new_tokens"] < 256 for row in candidate["rows"]),
    }


def next_round(
    candidates: list[str], cells: dict[str, dict[str, Any]], sizes: dict[str, list[int]],
    priority: list[str] | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    """首轮之后只扩仍有价值的证据不足单元；候选收敛与淘汰分别登记。"""
    stops = {}
    eligible = []
    for name in candidates:
        values = cells[name].values()
        if any(value["state"] == "淘汰" for value in values):
            stops[name] = "必要任务淘汰，停止覆盖全部任务的最终资格扩样"
        elif any(value["state"] == "基线缺口" for value in values):
            stops[name] = "必要任务基线缺口，无法授予全部任务保留"
        elif all(value["state"] == "保留" for value in values):
            stops[name] = "全部必要任务满足最终保留条件"
        else:
            eligible.append(name)
    if priority and stops.get(priority[0]) == "全部必要任务满足最终保留条件":
        for name in eligible:
            stops[name] = "低成本主候选已满足要求，停止备选扩样"
        eligible = []
    elif priority and priority[0] in eligible:
        primary = cells[priority[0]]
        primary_can_expand = any(
            value["state"] == "证据不足" and value["requested_n"] < sizes[task][-1]
            for task, value in primary.items()
        )
        if primary_can_expand:
            eligible = [priority[0]]
    requests = {}
    for name in eligible:
        for task, value in cells[name].items():
            if value["state"] != "证据不足":
                continue
            stages = sizes[task]
            index = stages.index(value["requested_n"])
            if index + 1 < len(stages):
                requests[f"{name}/{task}"] = stages[index + 1]
        if not any(pair.startswith(name + "/") for pair in requests):
            stops[name] = "必要任务达到样本上限仍证据不足"
    return requests, stops


def result_path(root: Path, model: str, task: str, condition: str) -> Path:
    """通过阶段唯一规则定位累计候选与追加参考。"""
    return quality_path(root, model, task, condition)


def reference_path(root: Path, model: str, task: str, condition: str) -> Path:
    """通过统一规则定位真实生成包的唯一首轮参考。"""
    return shared_reference_path(root, model, task, condition,
                                 load_plan(root)["v_plan"]["references"])


def reference_result(
    root: Path, model: str, task: str, condition: str, total: int,
) -> dict[str, Any]:
    """共用只读参考视图，不在消费者目录复制首轮记录。"""
    return reference_view(root, model, task, condition, total,
                          load_plan(root)["v_plan"]["references"])


def _device() -> dict[str, Any]:
    """记录实际模型计算设备与软件，避免按节点名推断精度。"""
    import torch
    import transformers

    return {
        "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "weights": "bfloat16", "attention": "float32",
        "autocast": torch.is_autocast_enabled(),
    }


def _reuse(
    root: Path, model: str, task: str, condition: str, items: list[dict[str, Any]],
    device: dict[str, Any], tokenizer: Any,
) -> tuple[dict[str, Any] | None, str]:
    """所有参考按共同身份判定；缺少证据先查证，不自动重新生成。"""
    from phase1.inputs import _ids_and_source

    path = reference_path(root, model, task, condition)
    value = read(path) if path.exists() else None
    expected = []
    for item in items:
        ids, source_hash = _ids_and_source(root, tokenizer, model, task, item)
        expected.append({"id": item["id"], "input_sha256": source_hash,
                         "input_tokens": len(ids)})
    registry_path = root / RESULTS / "reference_reuse.json"
    registry = read(registry_path) if registry_path.exists() else {}
    entry = registry.get("inventory", {}).get(f"{model}/{task}/{condition}", {})
    name = "bf16" if condition == "bf16" else config_for(root, model, condition)["name"]
    decision = assess_reuse(
        value, expected, model=model, configuration=name, device=device,
        evidence=entry.get("validation"), source_sha256=file_hash(path) if value else None,
        signature=computation_signature(root),
    )
    if decision.action != "reuse":
        return None, decision.reason
    raw_source = value.get("raw_source", value.get("prediction_source"))
    if raw_source and file_hash(root / raw_source["path"]) != raw_source["sha256"]:
        raise RuntimeError("已登记的原始预测校验不同，先核查来源")
    value["reuse_source"] = {"path": path.relative_to(root).as_posix(),
                             "sha256": file_hash(path), "raw_source": raw_source}
    value["reuse_validation"] = entry["validation"]
    return value, decision.reason


def prepare(root: Path) -> None:
    """输入依赖与复用清单核对通过后，冻结质量计划。"""
    from transformers import AutoTokenizer
    from phase1.models import MODELS

    plan = load_plan(root)
    device = _device()
    inventory = {}
    for model, candidates in plan["v_plan"]["candidates"].items():
        tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model])
        for name in candidates:
            config_for(root, model, name)
        for task in plan["v_plan"]["tasks"]:
            count = totals(root, task)[0]
            items = samples(root, model, task, count)
            for name in ("bf16", "kv8", "k8v4"):
                value, reason = _reuse(root, model, task, name, items, device, tokenizer)
                inventory[f"{model}/{task}/{name}"] = {
                    "action": "reuse" if value is not None else "measure", "reason": reason,
                    "source": value.get("reuse_source") if value else None,
                    "validation": value.get("reuse_validation") if value else None,
                }
                if value is None:
                    raise RuntimeError(f"参考需查证或匹配执行条件：{model}/{task}/{name}：{reason}")
    write_json(root / RESULTS / "reference_reuse.json", {
        "device": device, "inventory": inventory, "plan_digest": digest(plan["v_plan"]),
        "status": "passed", "model_manifests": {
            model: hashlib.sha256((root / "research/manifests/models" /
                f"{Path(path).name}.source.json").read_bytes()).hexdigest()
            for model, path in MODELS.items()},
    })
    plan["v_plan"]["frozen"] = True
    plan["v_plan"]["dependency_review"] = "passed_reference_reuse_and_expansion_inputs"
    write_json(root / "research/phase1/configs/static_layer_allocation.json", plan)
    publication = current_publication(root)
    if file_hash(root / "research/phase1/configs/static_layer_allocation.json") != (
        publication["files"]["research/phase1/configs/static_layer_allocation.json"]
    ):
        raise RuntimeError("冻结计划已经改变，重新发布源码后再执行")


def _saved_predictions(root: Path, value: dict[str, Any]) -> list[dict[str, Any]]:
    """从不可变来源读取当前行对应预测，追加时保留已完成测量。"""
    if not value.get("rows"):
        return []
    source = value["raw_source"]
    path = root / source["path"]
    if file_hash(path) != source["sha256"]:
        raise RuntimeError("已有完整预测来源发生改变")
    indexed = {row["id"]: row for row in read(path)["predictions"]}
    return [indexed[row["id"]] for row in value["rows"]]


def measure(root: Path, model_key: str, task: str, condition: str, total: int) -> None:
    """恢复唯一累计前缀，保存完整预测和逐请求峰值，不缩短输出预算。"""
    import torch
    from transformers import AutoTokenizer
    from phase1.runner import load_model
    from phase1.score import answer_exact, gsm8k_exact, token_f1
    from phase1.models import MODELS, _eos_ids
    from phase1.runner import generate_bf16 as bf16_generate
    from phase1.generation import _generate
    from phase1.inputs import _ids_and_source, _score_new_input

    plan = load_plan(root)
    if not plan["v_plan"]["frozen"] or total not in totals(root, task):
        raise RuntimeError("质量计划未冻结或判断档位非法")
    items = samples(root, model_key, task, total)
    reference = condition in ("bf16", "kv8", "k8v4")
    base_path = reference_path(root, model_key, task, condition) if reference else None
    base = read(base_path) if reference else {"rows": []}
    offset = len(base["rows"])
    if reference:
        tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
        reused, reason = _reuse(root, model_key, task, condition,
                                samples(root, model_key, task, offset), _device(), tokenizer)
        if reused is None:
            raise RuntimeError(f"参考尚未通过统一复用核验：{reason}")
    if reference and total <= offset:
        reference_result(root, model_key, task, condition, total)
        return
    if condition == "k8v4":
        raise RuntimeError("K8V4 只用于已有首轮对照，不强制扩样")
    path = result_path(root, model_key, task, condition)
    existing = read(path) if path.exists() else {"rows": []}
    rows = existing["rows"]
    if offset + len(rows) >= total:
        return
    config = config_for(root, model_key, condition)
    token_limit = plan["v_plan"]["tasks"][task]["max_new_tokens"]
    config_digest = digest(config)
    progress_folder = "shared_reference_progress" if reference else "shared_progress"
    progress_path = root / RUNTIME / progress_folder / model_key / f"{task}_{condition}.json"
    progress = read(progress_path) if progress_path.exists() else {
        "rows": rows, "predictions": _saved_predictions(root, existing),
        "source_segments": existing.get("source_segments", []),
        "config_digest": existing.get("config_digest", config_digest),
        "device": existing.get("device") or _device(),
        "publication": existing.get("publication") or current_publication(root),
        "reused_prefix": {"n": 0, "source": None},
    }
    device_identity = {key: value for key, value in _device().items()
                       if key not in ("slurm_job", "visible_device")}
    previous_device = {key: value for key, value in progress["device"].items()
                       if key not in ("slurm_job", "visible_device")}
    if progress["config_digest"] != config_digest or previous_device != device_identity:
        raise RuntimeError("恢复配置或执行设备已经改变")
    rows = progress["rows"]
    if rows and not progress.get("source_segments"):
        progress["source_segments"] = [{"start": 0, "end": len(rows),
                                         "publication": progress["publication"]}]
    if rows:
        validate_segments(root, {**progress, "n": len(rows)})
    publication = current_publication(root)
    verify_publication(root, publication)
    if [row["id"] for row in rows] != [
        item["id"] for item in items[offset:offset + len(rows)]
    ]:
        raise RuntimeError("保存进度不符合冻结顺序")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    free, _ = torch.cuda.mem_get_info()
    required_gib = 30 if model_key == "qwen" else 36
    if free < required_gib * 1024 ** 3:
        raise RuntimeError(f"实时空闲显存不足本次含余量估算 {required_gib} GiB")
    model = load_model(str(root / MODELS[model_key]), materialize_host_weights=True)
    eos = _eos_ids(tokenizer, model_key == "llama")
    for item in items[offset + len(rows):total]:
        ids, source_hash = _ids_and_source(root, tokenizer, model_key, task, item)
        if len(ids) + token_limit > 32768:
            raise RuntimeError("完整输入加输出预算超过 32K")
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            if config is None:
                produced, stop = bf16_generate(
                    model, torch.tensor([ids], device="cuda"), eos, token_limit,
                    consume_cache=True,
                )
                invalid = 0
            else:
                produced, stop, invalid = _generate(model, ids, eos, token_limit, config, {})
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row = {
            "id": item["id"], "input_tokens": len(ids), "input_sha256": source_hash,
            "new_tokens": len(produced), "stop": stop, "invalid_groups": invalid,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
        }
        if task == "gsm8k":
            row["correct"] = not invalid and gsm8k_exact(prediction, item["answer"])
        elif task == "hotpot":
            row.update(em=not invalid and answer_exact(prediction, item["answer"]),
                       f1=0.0 if invalid else token_f1(prediction, item["answer"]))
        else:
            row.update(_score_new_input(task, prediction, item, invalid))
        if task == "retrieval_ruler":
            row.update(depth=item["depth"], interference=item["interference"])
        rows.append(row)
        segments = progress.setdefault("source_segments", [])
        if segments and segments[-1]["publication"] == publication:
            segments[-1]["end"] = len(rows)
        else:
            segments.append({"start": len(rows) - 1, "end": len(rows),
                             "publication": publication})
        progress["predictions"].append({"id": item["id"], "text": prediction,
                                        "token_ids": produced})
        write_json(progress_path, progress)
        print("QUALITY", model_key, task, condition, offset + len(rows), total,
              row.get("correct", row.get("f1")), row["new_tokens"], flush=True)
    from phase1.result_format import write_quality_result

    payload = {
        "protocol": plan["v_plan"]["protocol"], "model": model_key, "task": task,
        "condition": condition, "config": config, "config_digest": config_digest,
        "device": progress["device"], "n": len(rows), "rows": rows,
        "output_limit": token_limit, "score": score_rows(rows, task),
        "raw_source": {"path": progress_path.relative_to(root).as_posix(),
                       "sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest()},
        "publication": progress["publication"],
        "source_segments": progress["source_segments"],
        "reused_prefix": progress["reused_prefix"],
    }
    if reference:
        payload.update(reference_source={
            "path": base_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(), "n": offset,
        }, total_n_with_reference=offset + len(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_quality_result(path, payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize(root: Path, model: str) -> dict[str, Any]:
    """复算当前档位的逐任务状态，保留所有既往档位的判断。"""
    plan = load_plan(root)["v_plan"]
    seeds = read(root / "research/phase1/splits/seeds.json")
    out = root / RESULTS / "expanded_quality" / model / "selection.json"
    old = read(out) if out.exists() else {"history": []}
    cells = {}
    for condition in plan["candidates"][model]:
        cells[condition] = {}
        for task, spec in plan["tasks"].items():
            candidate = read(result_path(root, model, task, condition))
            count = candidate["n"]
            values = []
            for name in ("bf16", "kv8"):
                values.append(reference_result(root, model, task, name, count))
            cells[condition][task] = judge(
                *values, candidate, task, spec, seeds["bootstrap"], seeds["bootstrap_draws"],
                final_stage=True,
            )
    request, stops = next_round(list(plan["candidates"][model]), cells,
                               {task: totals(root, task) for task in plan["tasks"]},
                               plan["llama_priority"] if model == "llama" else None)
    snapshot = {"cells": cells, "next_expansion": request, "stops": stops}
    history = old["history"]
    if not history or history[-1] != snapshot:
        history.append(snapshot)
    payload = {"protocol": plan["protocol"], "model": model, **snapshot,
               "history": history, "status": "complete" if not request else "expansion_required"}
    write_json(out, payload)
    return payload


def execution_plan(root: Path, model: str) -> dict[str, Any]:
    """只读列出参考复用与候选缺样；不会加载模型或默认重跑已有实体。"""
    plan = load_plan(root)["v_plan"]
    registry_path = root / RESULTS / "reference_reuse.json"
    registry = read(registry_path) if registry_path.exists() else {}
    records = {}
    catalog = ResultCatalog(root)
    completed = catalog.decision("static_layer_allocation")
    if completed["action"] == "reuse":
        selection = read(root / RESULTS / "expanded_quality" / model / "selection.json")
        if selection["status"] == "complete" and not selection["next_expansion"]:
            return {**completed, "model": model, "cells": {
                f"{task}/{condition}": {"action": "reuse", "completed_n": cell["requested_n"],
                                       "missing_n": 0, "state": cell["state"]}
                for condition, tasks in selection["cells"].items()
                for task, cell in tasks.items()
            }, "stops": selection["stops"]}
    signature = computation_signature(root)
    for task in plan["tasks"]:
        count = totals(root, task)[0]
        for condition in ("bf16", "kv8", "k8v4"):
            path = reference_path(root, model, task, condition)
            value = read(path) if path.exists() else None
            entry = registry.get("inventory", {}).get(f"{model}/{task}/{condition}", {})
            label = "bf16" if condition == "bf16" else (
                plan["references"]["kv8"] if condition == "kv8" else
                plan["references"]["same_format_k8v4"][model]
            )
            decision = assess_reuse(
                value, value["rows"][:count] if value else [], model=model,
                configuration=label, evidence=entry.get("validation"),
                source_sha256=file_hash(path) if value else None, signature=signature,
            )
            record = decision.record()
            if value is None:
                record["missing_n"] = count
            elif decision.action == "reuse" and value["n"] < count:
                record.update(action="append", missing_n=count - value["n"])
            records[f"{task}/{condition}"] = record
        for condition in plan["candidates"][model]:
            path = result_path(root, model, task, condition)
            value = read(path) if path.exists() else None
            have = value["n"] if value else 0
            records[f"{task}/{condition}"] = {
                "action": "review" if value else "measure",
                "completed_n": have, "missing_n": max(0, count - have),
                "reason": "已有候选先核对当前来源与冻结输入；扩样由原判定决定",
            }
    return {"policy": "verified-prefix-append-missing", "cells": records}


def run(root: Path, model: str) -> None:
    """三个候选四单元首轮完整执行，再按固定优先级逐档扩样。"""
    completed = ResultCatalog(root).decision("static_layer_allocation")
    if completed["action"] == "reuse":
        print("FINAL_QUALITY_REUSED", model, flush=True)
        return
    if root.joinpath("research/phase1/source_index.json").exists() and (
        completed["action"] == "review"
    ):
        raise RuntimeError(f"已有质量结果先核验，不自动补跑：{completed['reason']}")
    plan = load_plan(root)["v_plan"]
    for task in plan["tasks"]:
        total = totals(root, task)[0]
        for condition in ("bf16", "kv8", "k8v4", *plan["candidates"][model]):
            measure(root, model, task, condition, total)
    while True:
        selection = summarize(root, model)
        if not selection["next_expansion"]:
            break
        for pair, total in selection["next_expansion"].items():
            condition, task = pair.split("/")
            for reference in ("bf16", "kv8"):
                measure(root, model, task, reference, total)
            measure(root, model, task, condition, total)
    print("FINAL_QUALITY_COMPLETE", model, selection["stops"], flush=True)


if __name__ == "__main__":
    import argparse
    from kvstudy.paths import project_root

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--model", choices=("qwen", "llama"))
    args = parser.parse_args()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.action == "prepare":
        prepare(project_root())
    elif args.action == "run":
        run(project_root(), args.model)
    else:
        print(json.dumps(summarize(project_root(), args.model), ensure_ascii=False), flush=True)
