"""E2 检索与多证据问答的高精度生成基线，可从逐题进度恢复。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.e2_gsm import MODELS, _eos_ids
from phase1.runner import cache_lists, decode_step, load_model, prefill
from phase1.score import answer_exact, retrieval_exact, token_f1


def _samples(root: Path, model_key: str, task: str, split: str) -> list[dict[str, Any]]:
    """按冻结顺序返回题目、全文位置和校验值。"""
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
    if task == "hotpot":
        items = json.loads((root / "research/phase1/splits/hotpot.json").read_text("utf-8"))
        return [
            {
                "id": item["id"],
                "answer": item["answer"],
                "tokens": item["models"][model_key]["main_tokens"],
                "sha256": item["models"][model_key]["main_sha256"],
                "path": root / "research/.runtime/data/phase1-text/hotpot"
                / model_key / split / f"{item['id']}.main.txt",
            }
            for item in items[split]
        ]
    raise ValueError(f"未知任务：{task}")


def _generate(
    model: Any, input_ids: torch.Tensor, eos: set[int], limit: int
) -> tuple[list[int], str]:
    """生成至 EOS 或上限；返回 token 列表及停止原因。"""
    outputs = prefill(model, input_ids)
    keys, values = cache_lists(outputs.past_key_values)
    token = int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())
    produced = [token]
    if token in eos:
        return produced, "eos"
    for step in range(limit - 1):
        logits, keys, values = decode_step(model, keys, values, token, input_ids.shape[1] + step)
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
    }:
        raise ValueError(f"不支持的任务与划分：{task}/{split}")
    root = project_root()
    samples = _samples(root, model_key, task, split)
    if not samples:
        raise RuntimeError("冻结样本为空")
    ids = [item["id"] for item in samples]
    input_hashes = [item["sha256"] for item in samples]
    progress = root / "research/.runtime" / f"e2-{model_key}-{task}-{split}.json"
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ids
                or state.get("input_hashes") != input_hashes):
            raise RuntimeError("已有进度与当前代码或冻结输入不一致")
        rows = state["rows"]
    else:
        rows = []
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    limit = 128 if task == "retrieval" else 1024
    for item in samples[len(rows):]:
        source = item["path"].read_text("utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != item["sha256"]:
            raise RuntimeError(f"输入全文校验失败：{item['id']}")
        ids = tokenizer(source, add_special_tokens=False)["input_ids"]
        if len(ids) != item["tokens"] or len(ids) + limit > 32768:
            raise RuntimeError(f"输入长度不符合冻结记录：{item['id']}，{len(ids)}")
        with torch.inference_mode():
            produced, stopped = _generate(
                model, torch.tensor([ids], device="cuda"), eos, limit
            )
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row: dict[str, Any] = {
            "id": item["id"],
            "input_tokens": len(ids),
            "input_sha256": item["sha256"],
            "new_tokens": len(produced),
            "stop": stopped,
        }
        if task == "retrieval":
            row["correct"] = retrieval_exact(prediction, item["answer"])
        else:
            row["em"] = answer_exact(prediction, item["answer"])
            row["f1"] = token_f1(prediction, item["answer"])
        rows.append(row)
        progress.write_text(json.dumps({
            "commit": commit,
            "ids": ids,
            "input_hashes": input_hashes,
            "rows": rows,
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        print(model_key, task, split, len(rows), row, flush=True)
    payload: dict[str, Any] = {
        "protocol": "phase1-protocol v6" if task == "hotpot" else "phase1-protocol v7",
        "commit": commit,
        "model": model_key,
        "config": "bf16-h0",
        "task": f"{split}-{task}",
        "n": len(rows),
        "rows": rows,
    }
    if task == "retrieval":
        payload["correct"] = sum(row["correct"] for row in rows)
        payload["em"] = 100 * payload["correct"] / len(rows)
    else:
        payload["em"] = 100 * sum(row["em"] for row in rows) / len(rows)
        payload["f1"] = sum(row["f1"] for row in rows) / len(rows)
    out = root / "research/phase1/results/e2" / model_key
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}_{task}_bf16.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, task, split, payload.get("f1", payload.get("em")), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
