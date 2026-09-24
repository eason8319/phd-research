"""E0 的模型侧核对：prefill/decode 交界、全历史掩码，以及 pre-RoPE 与缓存一致。"""

from __future__ import annotations

import json
from typing import Any

import torch

from kvstudy.attention import full_history_mask
from kvstudy.paths import project_root
from phase1.e0_quant import write_report
from phase1.runner import cache_lists, decode_step, load_model, modeling_module, prefill

MODELS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "llama": "models/Llama-3.1-8B-Instruct",
}


def _capture_pre_rope(model: Any, input_ids: torch.Tensor) -> tuple[Any, list[torch.Tensor]]:
    module = modeling_module(model)
    original = module.apply_rotary_pos_emb
    saved: list[torch.Tensor] = []

    def wrapped(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        saved.append(key.detach().to("cpu"))
        return original(query, key, cos, sin, *args, **kwargs)

    module.apply_rotary_pos_emb = wrapped
    try:
        outputs = prefill(model, input_ids)
    finally:
        module.apply_rotary_pos_emb = original
    return outputs, saved


def _rope_error(model: Any, pre_key: torch.Tensor, post_key: torch.Tensor) -> float:
    """同一输入上，旋转前的 Key 再施加 RoPE，应接近缓存里的 post-RoPE Key。"""
    module = modeling_module(model)
    seq = pre_key.shape[2]
    hidden = torch.zeros(1, seq, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(seq, device="cuda").view(1, -1)
    cos, sin = model.model.rotary_emb(hidden, positions)
    _query, rotated = module.apply_rotary_pos_emb(pre_key.to("cuda"), pre_key.to("cuda"), cos, sin)
    delta = (rotated.float() - post_key.float()).abs()
    return float(delta.max())


def check_length(model: Any, length: int, vocab: int) -> dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(length)
    input_ids = torch.randint(10, vocab - 10, (1, length), generator=generator, device="cuda")
    outputs, saved = _capture_pre_rope(model, input_ids)
    keys, _values = cache_lists(outputs.past_key_values)
    logits = outputs.logits.view(-1, outputs.logits.shape[-1])
    next_id = int(logits[0].argmax())
    cache_len = int(keys[0].shape[2])
    mask = full_history_mask(cache_len + 1, device="cuda")
    _logits, updated_k, _updated_v = decode_step(model, keys, _values, next_id, length)
    rope_err = _rope_error(model, saved[0], keys[0])
    return {
        "length": length,
        "cache_tokens": cache_len,
        "decode_tokens": int(updated_k[0].shape[2]),
        "layers_captured": len(saved),
        "mask_ones": int(mask.sum()),
        "logits_shape": list(outputs.logits.shape),
        "rope_max_abs": rope_err,
        "ok": cache_len == length and int(updated_k[0].shape[2]) == length + 1 and int(mask.sum()) == length + 1 and outputs.logits.shape[1] == 1,
    }


def main(commit: str) -> None:
    quant_path = write_report(commit)
    root = project_root()
    out = root / "research/phase1/results/e0"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"protocol": "phase1-protocol v4", "commit": commit, "quant": str(quant_path.relative_to(root)).replace("\\", "/"), "models": {}}
    passed = True
    for name, path in MODELS.items():
        model = load_model(str(root / path))
        rows = [check_length(model, length, int(model.config.vocab_size)) for length in (32, 32768)]
        passed = passed and all(bool(row["ok"]) and float(row["rope_max_abs"]) < 0.05 for row in rows)
        report["models"][name] = rows
        del model
        torch.cuda.empty_cache()
        print(name, rows, flush=True)
    report["passed"] = passed
    (out / "model.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("E0 模型核对未通过")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "unknown")
