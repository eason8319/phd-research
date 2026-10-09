"""真实生成与误差来源检索与多证据问答的高精度生成基线，可从逐题进度恢复。"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.result_format import write_quality_result
from experiments.generation_error_sources.code.short_baseline import MODELS, _eos_ids
from experiments.generation_error_sources.code.sources import _runtime_path
from phase1.runner import cache_lists, decode_step, load_model, prefill
from phase1.score import (
    answer_exact, retrieval_exact, retrieval_number_exact, retrieval_numbers, token_f1,
)
from phase1.ruler_retrieval import CONFIG, MANIFEST, file_hash, verify_publication


def _samples(root: Path, model_key: str, task: str, split: str) -> list[dict[str, Any]]:
    """按冻结顺序返回题目、全文位置和校验值。"""
    if task == "retrieval_ruler":
        frozen = json.loads((root / MANIFEST).read_text("utf-8"))
        if split != "v" or frozen["n"] != 64:
            raise RuntimeError("RULER 改编检索仅冻结完整 V 64 题")
        if file_hash(root / CONFIG) != frozen["config_sha256"]:
            raise RuntimeError("冻结后配置发生变化")
        return [
            {"id": item["id"], "answer": item["models"][model_key]["value"],
             "tokens": item["models"][model_key]["tokens"],
             "sha256": item["models"][model_key]["sha256"],
             "path": root / item["models"][model_key]["path"]}
            for item in frozen["items"]
        ]
    if task == "retrieval":
        items = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
        prefix = f"synth-{split}-"
        return [
            {
                "id": item["id"],
                "answer": item["models"][model_key]["value"],
                "tokens": item["models"][model_key]["tokens"],
                "sha256": item["models"][model_key]["sha256"],
                "path": root / "research/.runtime/data/phase1-text/retrieval"
                / model_key / f"{item['id']}.txt",
            }
            for item in items if item["id"].startswith(prefix)
        ]
    if task in ("hotpot", "hotpot_long"):
        frozen = json.loads((root / "research/phase1/splits/hotpot.json").read_text("utf-8"))
        if task == "hotpot_long":
            if split != "v":
                raise ValueError("长解码诊断只使用 V")
            indexed = {item["id"]: item for item in frozen["v"]}
            items = [indexed[item_id] for item_id in frozen["long_decode"]]
            variant = "long"
        else:
            items = frozen[split]
            variant = "main"
        return [
            {
                "id": item["id"],
                "answer": item["answer"],
                "tokens": item["models"][model_key][f"{variant}_tokens"],
                "sha256": item["models"][model_key][f"{variant}_sha256"],
                "path": root / "research/.runtime/data/phase1-text/hotpot"
                / model_key / split / f"{item['id']}.{variant}.txt",
            }
            for item in items
        ]
    raise ValueError(f"未知任务：{task}")


def _generate(
    model: Any, input_ids: torch.Tensor, eos: set[int], limit: int,
    *, consume_cache: bool = False,
) -> tuple[list[int], str]:
    """生成至 EOS 或上限；返回 token 列表及停止原因。"""
    outputs = prefill(model, input_ids)
    keys, values = cache_lists(outputs.past_key_values)
    token = int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())
    del outputs
    produced = [token]
    if token in eos:
        return produced, "eos"
    for step in range(limit - 1):
        logits, keys, values = decode_step(
            model, keys, values, token, input_ids.shape[1] + step,
            consume_cache=consume_cache,
        )
        token = int(logits.reshape(-1, logits.shape[-1])[0].argmax())
        produced.append(token)
        if token in eos:
            return produced, "eos"
    return produced, "limit"


def run(model_key: str, task: str, split: str, commit: str) -> None:
    """逐样本核对输入并运行高精度生成；中断后从已完成样本继续。"""
    if (task, split) not in {
        ("retrieval", "c"), ("retrieval", "v"), ("retrieval", "expand"),
        ("hotpot", "v"), ("hotpot", "expand"),
        ("hotpot_long", "v"),
        ("retrieval_ruler", "v"),
    }:
        raise ValueError(f"不支持的任务与划分：{task}/{split}")
    root = project_root()
    samples = _samples(root, model_key, task, split)
    if not samples:
        raise RuntimeError("冻结样本为空")
    sample_ids = [item["id"] for item in samples]
    input_hashes = [item["sha256"] for item in samples]
    ruler = task == "retrieval_ruler"
    if ruler:
        verify_publication(root)
    frozen_hash = file_hash(root / MANIFEST) if ruler else None
    if ruler:
        progress = root / "research/.runtime/retrieval-ruler" / f"{model_key}.json"
    else:
        progress = _runtime_path(
            root, f"generation-{model_key}-{task}-{split}.json",
            f"e2-{model_key}-{task}-{split}.json",
        )
    progress.parent.mkdir(parents=True, exist_ok=True)
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != sample_ids
                or state.get("input_hashes") != input_hashes
                or ruler and state.get("manifest_sha256") != frozen_hash):
            raise RuntimeError("已有进度与当前代码或冻结输入不一致")
        rows = state["rows"]
    else:
        rows = []
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    limit = 128 if task in ("retrieval", "retrieval_ruler") else 1024
    config = json.loads((root / CONFIG).read_text("utf-8")) if ruler else None
    device = {"node": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for item in samples[len(rows):]:
        source = item["path"].read_text("utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != item["sha256"]:
            raise RuntimeError(f"输入全文校验失败：{item['id']}")
        input_ids = tokenizer(source, add_special_tokens=False)["input_ids"]
        if len(input_ids) != item["tokens"] or len(input_ids) + limit > 32768:
            raise RuntimeError(f"输入长度不符合冻结记录：{item['id']}，{len(input_ids)}")
        with torch.inference_mode():
            produced, stopped = _generate(
                model, torch.tensor([input_ids], device="cuda"), eos, limit,
                consume_cache=ruler,
            )
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row: dict[str, Any] = {
            "id": item["id"],
            "input_tokens": len(input_ids),
            "input_sha256": item["sha256"],
            "new_tokens": len(produced),
            "stop": stopped,
        }
        if ruler:
            row["correct"] = retrieval_number_exact(prediction, item["answer"])
            row["strict_correct"] = retrieval_exact(prediction, item["answer"])
            row["extracted_values"] = retrieval_numbers(prediction)
            row["target"] = item["answer"]
            row["prediction"] = prediction
            row["produced_ids"] = produced
        elif task == "retrieval":
            row["correct"] = retrieval_exact(prediction, item["answer"])
        else:
            row["em"] = answer_exact(prediction, item["answer"])
            row["f1"] = token_f1(prediction, item["answer"])
        rows.append(row)
        progress.write_text(json.dumps({
            "commit": commit,
            "ids": sample_ids,
            "input_hashes": input_hashes,
            "manifest_sha256": frozen_hash,
            "device": device,
            "rows": rows,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(model_key, task, split, len(rows), row["id"], row.get("correct"), flush=True)
        if ruler:
            torch.cuda.empty_cache()
    payload: dict[str, Any] = {
        "protocol": (
            config["protocol"] if ruler else
            "phase1-hotpot-direct-answer" if task.startswith("hotpot")
            else "phase1-full-key-retrieval"
        ),
        "commit": commit,
        "model": model_key,
        "config": "bf16-h0",
        "task": f"{split}-{task}",
        "n": len(rows),
        "rows": [
            {k: v for k, v in row.items() if k not in ("prediction", "produced_ids")}
            for row in rows
        ] if ruler else rows,
    }
    if task in ("retrieval", "retrieval_ruler"):
        payload["correct"] = sum(row["correct"] for row in rows)
        payload["em"] = 100 * payload["correct"] / len(rows)
        if ruler:
            payload.update(
                strict_correct=sum(row["strict_correct"] for row in rows),
                manifest_sha256=frozen_hash, config_sha256=file_hash(root / CONFIG),
                device=device, generation="bf16_weights_sdpa_prefill_fp32_attention_decode_greedy",
                raw_source={"path": progress.relative_to(root).as_posix(),
                            "sha256": file_hash(progress)},
                execution_source={
                    "path": "research/.runtime/retrieval-ruler/publication.json",
                    "sha256": file_hash(
                        root / "research/.runtime/retrieval-ruler/publication.json"
                    ),
                },
            )
    else:
        payload["em"] = 100 * sum(row["em"] for row in rows) / len(rows)
        payload["f1"] = sum(row["f1"] for row in rows) / len(rows)
    out = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    write_quality_result(out / f"{split}_{task}_bf16.json", payload)
    print(model_key, task, split, payload.get("f1", payload.get("em")), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
