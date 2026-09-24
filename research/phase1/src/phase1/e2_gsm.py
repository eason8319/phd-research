"""E2 短生成的高精度基线。贪心解码，遇 EOS 或 1024 个新 token 停止。"""

from __future__ import annotations

import json
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.runner import cache_lists, decode_step, load_model, prefill
from phase1.score import gsm8k_exact

MODELS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "llama": "models/Llama-3.1-8B-Instruct",
}


def _eos_ids(tokenizer: Any, llama: bool) -> set[int]:
    if llama:
        return {128001, 128008, 128009}
    values = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        values.append(int(tokenizer.convert_tokens_to_ids(token)))
    return set(values)


def _generate(model: Any, input_ids: torch.Tensor, eos: set[int]) -> list[int]:
    outputs = prefill(model, input_ids)
    keys, values = cache_lists(outputs.past_key_values)
    first = int(outputs.logits.view(-1, outputs.logits.shape[-1])[0].argmax())
    produced = [first]
    if first in eos:
        return produced
    token = first
    for step in range(1023):
        logits, keys, values = decode_step(model, keys, values, token, int(input_ids.shape[1]) + step)
        token = int(logits.view(-1)[0].argmax()) if logits.ndim == 1 else int(logits.view(-1, logits.shape[-1])[0].argmax())
        produced.append(token)
        if token in eos:
            break
    return produced


def run(model_key: str, commit: str) -> None:
    root = project_root()
    llama = model_key == "llama"
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, llama)
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text(encoding="utf-8"))
    templates = json.loads((root / "research/phase1/splits/templates.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "research/manifests/data/gsm8k.source.json").read_text(encoding="utf-8"))
    table = pq.read_table(meta["destination"])
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    out = root / "research/phase1/results/e2" / model_key
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    correct = 0
    for item in gsm["v"]:
        user = templates["gsm8k"].format(question=questions[item["row"]])
        kwargs = {"date_string": "26 Jul 2024"} if llama else {}
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            **kwargs,
        )
        produced = _generate(model, torch.tensor([ids], device="cuda"), eos)
        text = tokenizer.decode(produced, skip_special_tokens=True)
        ok = gsm8k_exact(text, answers[item["row"]])
        correct += int(ok)
        rows.append({"id": item["id"], "correct": ok, "new_tokens": len(produced)})
        print(item["id"], ok, len(produced), flush=True)
    payload = {
        "protocol": "phase1-protocol v5",
        "commit": commit,
        "model": model_key,
        "config": "bf16-h0",
        "task": "v-gsm8k",
        "n": len(rows),
        "correct": correct,
        "em": 100.0 * correct / len(rows),
        "rows": rows,
    }
    (out / "v_gsm8k_bf16.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(model_key, "em", payload["em"], flush=True)


if __name__ == "__main__":
    import sys

    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unknown")
