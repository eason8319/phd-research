"""静态层分配：冻结静态层分配在 V 的真实生成质量。"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.generation_error_sources.code.short_baseline import MODELS, _eos_ids
from experiments.generation_error_sources.code.quantized_generation import MAX_CONTEXT, _config, _generate, _ids_and_source, _inputs
from phase1.runner import load_model
from phase1.score import answer_exact, gsm8k_exact, token_f1


def run(model_key: str, task: str, allocation_name: str, commit: str) -> None:
    if task not in ("gsm8k", "hotpot"):
        raise ValueError("静态层分配 V 只运行基线达标的短题和长问答单元")
    root = project_root()
    folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    allocations = json.loads((folder / "allocations.json").read_text("utf-8"))
    gate = json.loads((folder / "gate.json").read_text("utf-8"))
    allowed = gate["selected_nonrandom_for_v"] + gate["random_for_v"]
    if allocation_name not in allowed:
        raise RuntimeError("分配未通过静态层分配的 V 准入清单")
    spec = allocations["allocations"][allocation_name]
    config = {
        **_config(allocations["representation"]),
        "layer_bits": spec["layer_bits"],
    }
    if config["key_mode"] != "per_token" or config["value_mode"] != "per_token":
        raise RuntimeError("当前静态层生成实现需要在线逐 token 的 K/V 轴")
    residual_h = int(allocations["residual_bits"])
    items = _inputs(root, model_key, task, "v")
    if len(items) != 64:
        raise RuntimeError("静态层分配 V 不是冻结的 64 题")
    ids_order = [item["id"] for item in items]
    if task == "gsm8k":
        template = json.loads((
            root / "research/phase1/splits/templates.json"
        ).read_text("utf-8"))["gsm8k"]
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        source_index = [template_hash + item["question_sha256"] for item in items]
    else:
        source_index = [item["sha256"] for item in items]
    digest = hashlib.sha256(json.dumps(spec["layer_bits"], sort_keys=True).encode(
        "utf-8"
    )).hexdigest()
    progress = root / "research/.runtime" / (
        f"layer_allocation-{model_key}-{task}-v-{allocation_name}.json"
    )
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ids_order
                or state["source_index"] != source_index
                or state["allocation_digest"] != digest):
            raise RuntimeError("静态层分配 V 进度与冻结输入或分配不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ids_order[:len(rows)]:
            raise RuntimeError("静态层分配 V 逐题进度顺序不一致")
    else:
        rows = []
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    limit = 1024
    for item in items[len(rows):]:
        ids, source_hash = _ids_and_source(root, tokenizer, model_key, task, item)
        if len(ids) + limit > MAX_CONTEXT:
            raise RuntimeError(f"输入超出窗口：{item['id']}")
        with torch.inference_mode():
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
            "allocation_digest": digest, "rows": rows,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(model_key, task, allocation_name, len(rows), row, flush=True)
    payload: dict[str, Any] = {
        "protocol": "phase1-full-key-retrieval", "commit": commit,
        "model": model_key, "allocation": allocation_name,
        "allocation_commit": allocations["commit"],
        "gate_commit": gate["commit"],
        "allocation_digest": digest,
        "representation": allocations["representation"],
        "residual_bits": residual_h,
        "task": f"v-{task}", "n": len(rows), "rows": rows,
        "invalid_samples": sum(row["invalid_groups"] > 0 for row in rows),
    }
    if task == "gsm8k":
        payload["correct"] = sum(row["correct"] for row in rows)
        payload["em"] = 100 * payload["correct"] / len(rows)
    else:
        payload["em"] = 100 * sum(row["em"] for row in rows) / len(rows)
        payload["f1"] = sum(row["f1"] for row in rows) / len(rows)
    (folder / f"v_{task}_{allocation_name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, task, allocation_name, payload.get("f1", payload.get("em")), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
