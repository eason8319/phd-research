"""真实生成与误差来源的 H=0 假量化质量运行；缓存反量化为 FP32，不代表压缩后的成本。"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.generation import (
    G_C,
    MAX_CONTEXT,
    RESIDUAL_TOKENS,
    _config,
    _calibration,
    _layer_params,
    _rotate_key,
    _quantize,
    _recent,
    _layer_bits,
    _prefill_cache,
    _write_residual_token,
    _decode_step,
    _generate,
)
from phase1.inputs import _inputs, _ids_and_source, _task_protocol, _score_new_input
from phase1.inputs import _samples
from phase1.models import MODELS, _eos_ids
from phase1.result_format import write_quality_result
from phase1.reuse import require_missing_result
from phase1.runner import (
    attention_from_batch,
    cache_lists,
    load_model,
    modeling_module,
    prefill,
    project_qkv,
)
from phase1.runtime_sources import _protocol_matches, _runtime_path
from phase1.score import (
    answer_exact,
    gsm8k_exact,
    retrieval_exact,
    retrieval_number_exact,
    retrieval_numbers,
    token_f1,
)


def run(
    model_key: str, task: str, split: str, name: str, commit: str, h_bits: int = 0,
    *, save_predictions: bool = False, guard_severe_failure: bool = False,
    execution_source: dict[str, Any] | None = None,
) -> None:
    """按一个配置和任务逐题运行；只回传小型评分与输入来源索引。"""
    if task not in (
        "gsm8k", "hotpot", "retrieval_ruler", "hotpot_four",
    ) or split not in ("v", "expand"):
        raise ValueError(f"不支持的任务与划分：{task}/{split}")
    if task in ("retrieval_ruler", "hotpot_four") and split != "v":
        raise ValueError("长任务只使用 V")
    if h_bits not in (0, 8, 16) or (h_bits and split != "v"):
        raise ValueError(f"残留条件不支持：H={h_bits}，{split}")
    root = project_root()
    config = _config(name)
    items = _inputs(root, model_key, task, split)
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    ids_order = [item["id"] for item in items]
    if task == "gsm8k":
        template = json.loads(
            (root / "research/phase1/splits/templates.json").read_text("utf-8")
        )["gsm8k"]
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        source_index = [template_hash + item["question_sha256"] for item in items]
    else:
        source_index = [item["sha256"] for item in items]
    suffix = f"-residual{h_bits}" if h_bits else ""
    legacy_suffix = f"-h{h_bits}" if h_bits else ""
    progress = _runtime_path(
        root,
        f"generation-{model_key}-{task}-{split}-{name}{suffix}.json",
        f"e2-{model_key}-{task}-{split}-{name}{legacy_suffix}.json",
    )
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ids_order
                or state.get("source_index") != source_index):
            raise RuntimeError("已有量化进度与代码或冻结输入不一致")
        rows = state["rows"]
        predictions = state.get("predictions", [])
        if save_predictions and (len(predictions) != len(rows)
                                 or state.get("execution_source") != execution_source):
            raise RuntimeError("恢复预测缺失或执行来源变化，不能静默重新生成")
    else:
        rows = []
        predictions = []
    out = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    require_missing_result(out / f"{split}_{task}_{name}{suffix}.json")
    if guard_severe_failure:
        from experiments.generation_error_sources.code.early_stop import severe_failure
        references = [json.loads((out / f"{split}_{task}_{ref}.json").read_text("utf-8"))
                      for ref in ("bf16", "kv8-per_token-post")]
        failure = severe_failure(rows, references, task)
        if failure is not None:
            raise RuntimeError(f"已有进度已触发停止：{failure}")
    calibration = _calibration(root, model_key, config)
    model = load_model(str(root / MODELS[model_key]))
    if save_predictions:
        execution_device = {
            "name": torch.cuda.get_device_name(),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        }
        torch.cuda.reset_peak_memory_stats()
    eos = _eos_ids(tokenizer, model_key == "llama")
    limit = 128 if task in ("retrieval", "retrieval_ruler") else 1024
    for item in items[len(rows):]:
        ids, source_hash = _ids_and_source(root, tokenizer, model_key, task, item)
        if len(ids) + limit > MAX_CONTEXT:
            raise RuntimeError(f"输入超出总窗口：{item['id']}")
        with torch.inference_mode():
            produced, stop, invalid = _generate(
                model, ids, eos, limit, config, calibration, h_bits
            )
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row: dict[str, Any] = {
            "id": item["id"], "input_tokens": len(ids), "input_sha256": source_hash,
            "new_tokens": len(produced), "stop": stop, "invalid_groups": invalid,
        }
        if save_predictions:
            row["memory_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["memory_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if task == "gsm8k":
            row["correct"] = not invalid and gsm8k_exact(prediction, item["answer"])
        elif task == "retrieval":
            row["correct"] = not invalid and retrieval_exact(prediction, item["answer"])
        elif task in ("retrieval_ruler", "hotpot_four"):
            row.update(_score_new_input(task, prediction, item, invalid))
        else:
            row["em"] = not invalid and answer_exact(prediction, item["answer"])
            row["f1"] = 0.0 if invalid else token_f1(prediction, item["answer"])
        if save_predictions:
            predictions.append({"id": item["id"], "text": prediction, "token_ids": produced})
            row["prediction_source"] = {
                "path": progress.relative_to(root).as_posix(), "index": len(rows),
                "text_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
            }
        rows.append(row)
        state = {
            "commit": commit, "ids": ids_order, "source_index": source_index,
            "rows": rows,
        }
        if save_predictions:
            state.update(predictions=predictions, execution_source=execution_source,
                         execution_device=execution_device,
                         model=model_key, config=name, residual_bits=h_bits,
                         residual_tokens=RESIDUAL_TOKENS if h_bits else 0,
                         task=f"{split}-{task}", generation={
                             "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                             "max_new_tokens": limit, "eos_token_ids": sorted(eos),
                         })
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(progress)
        print(model_key, task, split, name, len(rows), row, flush=True)
        if task in ("retrieval_ruler", "hotpot_four"):
            torch.cuda.empty_cache()
        if guard_severe_failure:
            failure = severe_failure(rows, references, task)
            if failure is not None:
                partial = {
                    "protocol": "phase1-full-key-retrieval", "commit": commit,
                    "model": model_key, "config": name, "task": f"{split}-{task}",
                    "planned_n": len(items), "observed_n": len(rows), "rows": rows,
                    "residual_bits": h_bits, "residual_tokens": RESIDUAL_TOKENS if h_bits else 0,
                    "stopped_early": True, "reason": failure,
                    "calibration_commit": calibration.get("commit"),
                    "execution_source": execution_source,
                }
                write_quality_result(
                    out / f"{split}_{task}_{name}{suffix}_early_stop.json", partial,
                )
                raise RuntimeError(f"配置触发严重失效停止：{failure}")
    payload: dict[str, Any] = {
        "protocol": _task_protocol(root, task), "commit": commit, "model": model_key,
        "config": name, "task": f"{split}-{task}", "n": len(rows), "rows": rows,
        "calibration_commit": calibration.get("commit"),
        "invalid_samples": sum(row["invalid_groups"] > 0 for row in rows),
    }
    if h_bits:
        payload["residual_bits"] = h_bits
        payload["residual_tokens"] = RESIDUAL_TOKENS
    if save_predictions:
        payload["execution_source"] = execution_source
        payload["execution_device"] = execution_device
        payload["prediction_source"] = {
            "path": progress.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
        }
    if task == "hotpot_four":
        payload.update(
            statistical_unit="request", questions_per_request=4, n_questions=4 * len(rows),
            em=100 * sum(row["em"] for row in rows) / len(rows),
            f1=sum(row["f1"] for row in rows) / len(rows),
            answer_em=sum(row["answer_em_percent"] for row in rows) / len(rows),
            missing_answers=sum(row["missing_answers"] for row in rows),
        )
    elif task.startswith("hotpot"):
        payload["em"] = 100 * sum(row["em"] for row in rows) / len(rows)
        payload["f1"] = sum(row["f1"] for row in rows) / len(rows)
    else:
        payload["correct"] = sum(row["correct"] for row in rows)
        payload["em"] = 100 * payload["correct"] / len(rows)
        if task == "retrieval_ruler":
            payload["strict_correct"] = sum(row["strict_correct"] for row in rows)
    out = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    write_quality_result(out / f"{split}_{task}_{name}{suffix}.json", payload)
    print(model_key, task, split, name, payload.get("f1", payload.get("em")), flush=True)


if __name__ == "__main__":
    run(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
        int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    )
