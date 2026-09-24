"""E1：每层用高精度隐藏状态单独算注意力误差，不向后传播。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from kvstudy.attention import attention_fp32, centered_logit_rms, probability_kl, relative_output_error
from kvstudy.groups import fit_per_channel, quantize_per_channel, quantize_per_token
from kvstudy.ledger import account_pair
from kvstudy.paths import project_root
from phase1.runner import cache_lists, load_model, modeling_module, prefill
from phase1.select_e1 import E1Cell, select_e1

MODELS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "llama": "models/Llama-3.1-8B-Instruct",
}
G_C = 32


def _render_gsm(tokenizer: Any, question: str, llama: bool) -> torch.Tensor:
    templates = json.loads((project_root() / "research/phase1/splits/templates.json").read_text(encoding="utf-8"))
    user = templates["gsm8k"].format(question=question)
    kwargs = {"date_string": "26 Jul 2024"} if llama else {}
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    return torch.tensor([ids], device="cuda")


def _load_c_items(model_key: str) -> list[dict[str, Any]]:
    root = project_root()
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "research/manifests/data/gsm8k.source.json").read_text(encoding="utf-8"))
    questions = pq.read_table(meta["destination"]).column("question").to_pylist()
    items = []
    for row in gsm["c"]:
        items.append({"id": row["id"], "fold": int(row["fold"]), "kind": "math", "question": questions[row["row"]]})
    retrieval = json.loads((root / "research/phase1/splits/retrieval.json").read_text(encoding="utf-8"))
    for row in retrieval:
        if not row["id"].startswith("synth-c-"):
            continue
        path = root / "research/.runtime/data/phase1-text/retrieval" / model_key / f"{row['id']}.txt"
        items.append({"id": row["id"], "fold": int(row["fold"]), "kind": "retrieval", "text": path.read_text(encoding="utf-8")})
    del questions
    return items


def _capture(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    module = modeling_module(model)
    original = module.apply_rotary_pos_emb
    pre: list[torch.Tensor] = []

    def wrapped(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        pre.append(key.detach())
        return original(query, key, cos, sin, *args, **kwargs)

    module.apply_rotary_pos_emb = wrapped
    try:
        outputs = prefill(model, input_ids)
        token = int(outputs.logits.view(-1, outputs.logits.shape[-1])[0].argmax())
        query_pre: list[torch.Tensor] = []

        def wrapped_query(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            query_pre.append(query.detach())
            pre.append(key.detach())
            return original(query, key, cos, sin, *args, **kwargs)

        module.apply_rotary_pos_emb = wrapped_query
        step = model(
            input_ids=torch.tensor([[token]], device=input_ids.device),
            past_key_values=outputs.past_key_values,
            use_cache=True,
            num_logits_to_keep=1,
        )
    finally:
        module.apply_rotary_pos_emb = original
    keys, values = cache_lists(step.past_key_values)
    n_layers = len(query_pre)
    history_pre = pre[:n_layers]
    new_pre = pre[n_layers:]
    return {
        "pre": [torch.cat([old[0], new[0]], dim=1).float().cpu() for old, new in zip(history_pre, new_pre)],
        "post": [item[0].float().cpu() for item in keys],
        "value": [item[0].float().cpu() for item in values],
        "query": [item[0, :, 0, :].float().cpu() for item in query_pre],
        "token": token,
    }


def _configs() -> list[dict[str, Any]]:
    rows = []
    for key_mode in ("per_channel", "per_token"):
        for rope in ("pre", "post"):
            for name, k_bits, v_bits in (("kv8", 8, 8), ("k8v4", 8, 4), ("k4v8", 4, 8), ("kv4", 4, 4)):
                tier = "6" if name in ("k8v4", "k4v8") else ("4" if name == "kv4" else "base")
                rows.append({
                    "name": f"{name}-{key_mode}-{rope}",
                    "key_mode": key_mode,
                    "rope": rope,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "v_mode": "per_token",
                    "tier": tier,
                })
    rows.append({
        "name": "kv4-per_channel-pre-vchannel",
        "key_mode": "per_channel",
        "rope": "pre",
        "k_bits": 4,
        "v_bits": 4,
        "v_mode": "per_channel",
        "tier": "4",
    })
    return rows


def _quantize_side(values: torch.Tensor, bits: int, mode: str, params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, int]:
    if mode == "per_token":
        restored, stats = quantize_per_token(values, bits, G_C)
        return restored, stats.invalid_groups
    assert params is not None
    restored, stats = quantize_per_channel(values, bits, *params)
    return restored, stats.invalid_groups


def _layer_error(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_hat: torch.Tensor, value_hat: torch.Tensor) -> tuple[float, bool]:
    reference, _, _ = attention_fp32(query, key, value)
    candidate, scores_hat, probs_hat = attention_fp32(query, key_hat, value_hat)
    _reference_scores = None
    error, zero = relative_output_error(reference, candidate)
    del scores_hat, probs_hat, _reference_scores
    return error, zero


def run_model(model_key: str, commit: str) -> None:
    root = project_root()
    from transformers import AutoTokenizer

    llama = model_key == "llama"
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    items = _load_c_items(model_key)
    configs = _configs()
    print(f"{model_key} samples {len(items)}", flush=True)
    captured = []
    for item in items:
        if item["kind"] == "math":
            ids = _render_gsm(tokenizer, item["question"], llama)
        else:
            ids = tokenizer(item["text"], add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        blob = _capture(model, ids)
        blob["id"] = item["id"]
        blob["fold"] = item["fold"]
        blob["seq"] = int(ids.shape[1])
        captured.append(blob)
        print(item["id"], blob["seq"], flush=True)
    n_layers = len(captured[0]["pre"])
    kv = captured[0]["pre"][0].shape[0]
    dim = captured[0]["pre"][0].shape[-1]
    scales: dict[int, dict[str, Any]] = {}
    for fold in (0, 1):
        acc = {
            "k_pre": _empty_minmax(n_layers, kv, dim),
            "k_post": _empty_minmax(n_layers, kv, dim),
            "v": _empty_minmax(n_layers, kv, dim),
        }
        for blob in captured:
            if blob["fold"] != fold:
                continue
            _update_minmax(acc["k_pre"], blob["pre"])
            _update_minmax(acc["k_post"], blob["post"])
            _update_minmax(acc["v"], blob["value"])
        scales[fold] = {
            "k_pre": _fit_minmax(acc["k_pre"], 8),
            "k_pre4": _fit_minmax(acc["k_pre"], 4),
            "k_post": _fit_minmax(acc["k_post"], 8),
            "k_post4": _fit_minmax(acc["k_post"], 4),
            "v4": _fit_minmax(acc["v"], 4),
        }
    totals: dict[str, dict[str, float]] = {
        cfg["name"]: {"err0": 0.0, "n0": 0.0, "err1": 0.0, "n1": 0.0, "invalid": 0.0, "bytes": 0.0} for cfg in configs
    }
    for blob in captured:
        other = 1 - blob["fold"]
        seq = blob["seq"]
        for cfg in configs:
            errors = []
            invalid = 0
            for layer in range(n_layers):
                key_src = blob["pre"][layer] if cfg["rope"] == "pre" else blob["post"][layer]
                key_params = _key_params(scales[other], cfg, layer)
                key_hat, bad_k = _quantize_side(key_src, cfg["k_bits"], cfg["key_mode"], key_params)
                key_ref = blob["post"][layer]
                if cfg["rope"] == "pre":
                    key_hat = _rotate_key(model, key_hat)
                value = blob["value"][layer]
                if cfg["v_mode"] == "per_channel":
                    value_hat, bad_v = _quantize_side(value, cfg["v_bits"], "per_channel", scales[other]["v4"][layer])
                else:
                    value_hat, bad_v = _quantize_side(value, cfg["v_bits"], "per_token", None)
                invalid += bad_k + bad_v
                query = _rotate_query(model, blob["query"][layer], seq)
                error, zero = _layer_error(query, key_ref, value, key_hat, value_hat)
                if not zero:
                    errors.append(error)
            mean = sum(errors) / len(errors) if errors else 0.0
            slot = totals[cfg["name"]]
            slot[f"err{blob['fold']}"] += mean
            slot[f"n{blob['fold']}"] += 1
            slot["invalid"] += invalid
            slot["bytes"] += _bytes(cfg, n_layers, kv, seq, dim)
        print("scored", blob["id"], flush=True)
    del model
    torch.cuda.empty_cache()
    cells = []
    kv8_name = "kv8-per_token-post"
    kv4_name = "kv4-per_token-post"
    summary = {}
    for cfg in configs:
        slot = totals[cfg["name"]]
        fold_mean = 0.5 * (
            slot["err0"] / max(slot["n0"], 1) + slot["err1"] / max(slot["n1"], 1)
        )
        summary[cfg["name"]] = {"error": fold_mean, "invalid": slot["invalid"], "bytes": slot["bytes"]}
        if cfg["tier"] == "base":
            continue
        cells.append(E1Cell(cfg["name"], cfg["tier"], fold_mean, _bits(cfg), int(slot["bytes"]), slot["invalid"] > 0))
    kv8_bits = _bits(next(cfg for cfg in configs if cfg["name"] == kv8_name))
    chosen = select_e1(cells, kv8_bits, summary[kv4_name]["error"])
    out = root / "research/phase1/results/e1" / model_key
    out.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": "phase1-protocol v4", "commit": commit, "summary": summary, "kept": list(chosen.kept), "reason": chosen.reason}
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(model_key, chosen, flush=True)


def _empty_minmax(layers: int, kv: int, dim: int) -> dict[str, torch.Tensor]:
    return {
        "min": torch.full((layers, kv, dim), float("inf")),
        "max": torch.full((layers, kv, dim), float("-inf")),
    }


def _update_minmax(acc: dict[str, torch.Tensor], tensors: list[torch.Tensor]) -> None:
    for layer, values in enumerate(tensors):
        acc["min"][layer] = torch.minimum(acc["min"][layer], values.amin(dim=1))
        acc["max"][layer] = torch.maximum(acc["max"][layer], values.amax(dim=1))


def _fit_minmax(acc: dict[str, torch.Tensor], bits: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    fitted = []
    for layer in range(acc["min"].shape[0]):
        # 用最小/最大两个值代表该通道的校准范围。
        pair = torch.stack([acc["min"][layer], acc["max"][layer]], dim=1)
        scale, offset, status = fit_per_channel(pair, bits)
        fitted.append((scale, offset, status))
    return fitted


def _key_params(scale_pack: dict[str, Any], cfg: dict[str, Any], layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if cfg["key_mode"] != "per_channel":
        return None
    kind = "k_pre" if cfg["rope"] == "pre" else "k_post"
    if cfg["k_bits"] == 4:
        kind = kind + "4"
    return scale_pack[kind][layer]


def _rotate_key(model: Any, key: torch.Tensor) -> torch.Tensor:
    """对 [kv, seq, dim] 的未旋转 Key 施加与模型相同的 RoPE。"""
    module = modeling_module(model)
    seq = key.shape[1]
    hidden = torch.zeros(1, seq, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(seq, device="cuda").view(1, -1)
    cos, sin = model.model.rotary_emb(hidden, positions)
    packed = key.unsqueeze(0).to("cuda")
    _query, rotated = module.apply_rotary_pos_emb(packed, packed, cos, sin)
    return rotated[0].float().cpu()


def _rotate_query(model: Any, query: torch.Tensor, position: int) -> torch.Tensor:
    """旋转第一个生成 token 的 query。query 形状为 [heads, dim]。"""
    module = modeling_module(model)
    hidden = torch.zeros(1, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([[position]], device="cuda")
    cos, sin = model.model.rotary_emb(hidden, positions)
    packed = query.view(1, query.shape[0], 1, query.shape[1]).to("cuda")
    rotated, _key = module.apply_rotary_pos_emb(packed, packed[:, :1], cos, sin)
    return rotated[0, :, 0, :].float().cpu()


def _bytes(cfg: dict[str, Any], layers: int, kv: int, seq: int, dim: int) -> float:
    ledger = account_pair(
        layers=layers,
        kv_heads=kv,
        seq=seq,
        dim=dim,
        k_bits=cfg["k_bits"],
        v_bits=cfg["v_bits"],
        k_mode=cfg["key_mode"] if cfg["key_mode"] != "per_channel" else "per_channel",
        v_mode="per_channel" if cfg["v_mode"] == "per_channel" else "per_token",
        g_c=G_C,
        g_t=32,
        block_tokens=64,
    )
    return float(ledger.resident_bytes)


def _bits(cfg: dict[str, Any]) -> float:
    return (cfg["k_bits"] + cfg["v_bits"]) / 2 + (0 if cfg["key_mode"] == "per_channel" else 1)


def main() -> None:
    import sys

    commit = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    run_model(sys.argv[1], commit)


if __name__ == "__main__":
    main()
