"""按冻结顺序把验证候选及其参照扩到 128、最多 256 题。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.select_final import (
    _canonical_condition, result_file_name,
)
from experiments.generation_error_sources.code.short_baseline import MODELS, _eos_ids
from experiments.generation_error_sources.code.long_baseline import _generate as _generate_bf16
from experiments.generation_error_sources.code.quantized_generation import (
    MAX_CONTEXT, _config, _generate, _ids_and_source, _inputs,
)
from phase1.runner import load_model
from phase1.score import answer_exact, gsm8k_exact, token_f1


def _condition(
    root: Path, model_key: str, name: str
) -> tuple[dict[str, Any] | None, int, str]:
    """读取冻结简单表示或静态分配；内容条件名与历史身份指向同一配置。"""
    name = _canonical_condition(name)
    if name == "bf16":
        return None, 0, "bf16"
    if name == "kv8":
        return _config("kv8-per_token-post"), 0, "kv8-per_token-post"
    e2_folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    if name == "e2":
        selected = json.loads((e2_folder / "residual_selection.json").read_text("utf-8"))
        return (
            _config(selected["e3_representation"]),
            int(selected["e3_residual_bits"]),
            selected["e3_representation"],
        )
    if name.startswith("e3-"):
        allocation_name = name[3:]
        folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
        selection = json.loads((folder / "selection.json").read_text("utf-8"))
        gate = json.loads((folder / "gate.json").read_text("utf-8"))
        allowed = [selection["f_protocol_candidate"], *gate["random_for_v"]]
        if allocation_name not in allowed:
            raise RuntimeError("该静态层分配未被选中进入扩样")
        allocations = json.loads((folder / "allocations.json").read_text("utf-8"))
        spec = allocations["allocations"][allocation_name]
        config = {
            **_config(allocations["representation"]),
            "layer_bits": spec["layer_bits"],
        }
        return config, int(allocations["residual_bits"]), allocation_name
    raise ValueError(f"未知扩样条件：{name}")


def run(
    model_key: str, task: str, condition_name: str,
    extra_n: int, commit: str,
) -> None:
    """追加冻结顺序的 64 或 192 题，保存逐题评分及可继续的进度。"""
    if task not in ("gsm8k", "hotpot") or extra_n not in (64, 192):
        raise ValueError("扩样只接受冻结任务和 64→192 的追加题数")
    condition_name = _canonical_condition(condition_name)
    root = project_root()
    config, residual_h, source_name = _condition(root, model_key, condition_name)
    if config is not None and (
        config["key_mode"] != "per_token" or config["value_mode"] != "per_token"
    ):
        raise RuntimeError("当前扩样实现需要在线逐 token 的 K/V 轴")
    items = _inputs(root, model_key, task, "expand")
    if len(items) != 192:
        raise RuntimeError("扩样池不是冻结的 192 题")
    ids_order = [item["id"] for item in items]
    if task == "gsm8k":
        template = json.loads((
            root / "research/phase1/splits/templates.json"
        ).read_text("utf-8"))["gsm8k"]
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        source_index = [template_hash + item["question_sha256"] for item in items]
    else:
        source_index = [item["sha256"] for item in items]
    config_digest = hashlib.sha256(json.dumps({
        "name": condition_name, "source": source_name,
        "config": config, "residual_h": residual_h,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    file_name = result_file_name(task, condition_name, 64 + extra_n)
    file_condition = file_name[len(task) + 1:].rsplit("_n", 1)[0]
    progress = root / "research/.runtime" / (
        f"expanded-quality-{model_key}-{task}-{file_condition}.json"
    )
    historical_progress = root / "research/.runtime" / (
        f"f-{model_key}-{task}-{condition_name}.json"
    )
    progress_source = progress if progress.exists() else historical_progress
    if progress_source.exists():
        state = json.loads(progress_source.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ids_order
                or state["source_index"] != source_index
                or state["config_digest"] != config_digest):
            raise RuntimeError("保存的进度与冻结扩样输入或条件不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ids_order[:len(rows)]:
            raise RuntimeError("保存的扩样顺序不一致")
    else:
        rows = []
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    limit = 1024
    for item in items[len(rows):extra_n]:
        ids, source_hash = _ids_and_source(root, tokenizer, model_key, task, item)
        if len(ids) + limit > MAX_CONTEXT:
            raise RuntimeError(f"扩样输入超出窗口：{item['id']}")
        with torch.inference_mode():
            if config is None:
                produced, stop = _generate_bf16(
                    model, torch.tensor([ids], device="cuda"), eos, limit
                )
                invalid = 0
            else:
                produced, stop, invalid = _generate(
                    model, ids, eos, limit, config, {}, residual_h
                )
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row: dict[str, Any] = {
            "id": item["id"], "input_tokens": len(ids),
            "input_sha256": source_hash,
            "new_tokens": len(produced), "stop": stop,
            "invalid_groups": invalid,
        }
        if task == "gsm8k":
            row["correct"] = not invalid and gsm8k_exact(prediction, item["answer"])
        else:
            row["em"] = not invalid and answer_exact(prediction, item["answer"])
            row["f1"] = 0.0 if invalid else token_f1(prediction, item["answer"])
        rows.append(row)
        progress.write_text(json.dumps({
            "commit": commit, "ids": ids_order,
            "source_index": source_index,
            "config_digest": config_digest, "rows": rows,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(model_key, task, condition_name, len(rows), row, flush=True)
    if len(rows) < extra_n:
        raise RuntimeError("扩样未达到本次要求的题数")
    used = rows[:extra_n]
    payload: dict[str, Any] = {
        "protocol": "phase1-full-key-retrieval", "commit": commit,
        "model": model_key, "task": task,
        "condition": condition_name, "source": source_name,
        "residual_bits": residual_h,
        "config_digest": config_digest,
        "extra_n": extra_n, "total_n_with_v": 64 + extra_n,
        "rows": used,
        "invalid_samples": sum(row["invalid_groups"] > 0 for row in used),
    }
    if task == "gsm8k":
        payload["correct"] = sum(row["correct"] for row in used)
        payload["em"] = 100 * payload["correct"] / extra_n
    else:
        payload["em"] = 100 * sum(row["em"] for row in used) / extra_n
        payload["f1"] = sum(row["f1"] for row in used) / extra_n
    out = (
        root / "research/phase1/experiments/static_layer_allocation/results/expanded_quality"
        / model_key
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / file_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, task, condition_name, "total", 64 + extra_n, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
