"""M0：核对 GPU 上的 BF16，并测量短输入、4K 和 32K 的高精度冒烟。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from kvstudy.paths import project_root
from phase1.runner import cache_lists, decode_step, load_model, prefill, timed

LENGTHS = (32, 4096, 32768)
MODELS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "llama": "models/Llama-3.1-8B-Instruct",
}


def smoke_length(model: torch.nn.Module, length: int, vocab: int) -> dict[str, float | int | str]:
    """跑一次 prefill 和一步 decode。显存按本次重置后的峰值字节记录。"""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(length)
    tokens = torch.randint(10, vocab - 10, (1, length), generator=generator)
    input_ids = tokens.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        outputs, prefill_s = timed(lambda: prefill(model, input_ids))
        past_k, past_v = cache_lists(outputs.past_key_values)
        logits = outputs.logits
        next_id = int(logits.view(-1, logits.shape[-1])[0].argmax())
        _, decode_s = timed(
            lambda: decode_step(model, past_k, past_v, next_id, length)
        )
        peak = int(torch.cuda.max_memory_allocated())
        return {
            "length": length,
            "status": "ok",
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "peak_bytes": peak,
            "cache_tokens": int(past_k[0].shape[2]),
            "logits_shape": list(logits.shape),
        }
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return {"length": length, "status": "oom", "error": str(exc)}


def run(model_key: str, commit: str) -> None:
    """对一个模型写 results/m0/<model>.json。"""
    root = project_root()
    os.environ.setdefault("HF_HOME", str(root / "research/.runtime/cache/hf"))
    os.environ.setdefault("TORCH_HOME", str(root / "research/.runtime/cache/torch"))
    os.environ.setdefault("TMPDIR", str(root / "research/.runtime/tmp"))
    model = load_model(str(root / MODELS[model_key]))
    bf16_ok = bool(torch.cuda.is_bf16_supported())
    probe = torch.zeros(1, device="cuda", dtype=torch.bfloat16)
    del probe
    rows = [smoke_length(model, length, int(model.config.vocab_size)) for length in LENGTHS]
    payload = {
        "protocol": "phase1-protocol v4",
        "commit": commit,
        "model": model_key,
        "model_dir": MODELS[model_key],
        "bf16": bf16_ok,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "lengths": rows,
    }
    out = root / "research/phase1/results/m0"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{model_key}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unknown")
