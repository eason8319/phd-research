"""E3：在 C 的高精度轨迹上，逐个（层，K/V）探测 INT4 敏感度。"""

from __future__ import annotations

import json
import sys
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.attention import attention_fp32
from kvstudy.paths import project_root
from phase1.e1 import MODELS, _capture, _load_c_items, _render_gsm, _rotate_query
from phase1.e2_quant import _calibration, _config, _quantize, _rotate_key
from phase1.runner import load_model


def _relative_by_kv_head(
    reference: torch.Tensor, candidate: torch.Tensor, kv_heads: int
) -> tuple[float, int]:
    """每个 KV head 对应的 query head 相对误差，再取可定义组的均值。"""
    repeats = reference.shape[0] // kv_heads
    values: list[float] = []
    zero = 0
    for head in range(kv_heads):
        sl = slice(head * repeats, (head + 1) * repeats)
        denominator = torch.sum(reference[sl] ** 2)
        if float(denominator) == 0.0:
            zero += 1
            continue
        numerator = torch.sum((reference[sl] - candidate[sl]) ** 2)
        values.append(float(numerator / denominator))
    return (sum(values) / len(values) if values else 0.0), zero


def _reconstruction(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """按 KV head 计算相对重建误差，再取均值。"""
    numerator = torch.sum((reference - candidate) ** 2, dim=(1, 2))
    denominator = torch.sum(reference ** 2, dim=(1, 2))
    valid = denominator > 0
    if not bool(valid.any()):
        return 0.0
    return float((numerator[valid] / denominator[valid]).mean())


def _empty(n_layers: int) -> list[dict[str, float | int]]:
    return [
        {"sensitivity_sum": 0.0, "reconstruction_sum": 0.0,
         "samples": 0, "zero_heads": 0, "invalid_groups": 0}
        for _ in range(n_layers * 2)
    ]


def _summarize(rows: list[dict[str, float | int]]) -> list[dict[str, Any]]:
    result = []
    for unit, row in enumerate(rows):
        n = int(row["samples"])
        result.append({
            "layer": unit // 2,
            "side": "k" if unit % 2 == 0 else "v",
            "sensitivity": float(row["sensitivity_sum"]) / n if n else None,
            "reconstruction": float(row["reconstruction_sum"]) / n if n else None,
            "samples": n,
            "zero_heads": int(row["zero_heads"]),
            "invalid_groups": int(row["invalid_groups"]),
        })
    return result


def _ranking(rows: list[dict[str, Any]], metric: str) -> list[int]:
    if any(row[metric] is None for row in rows):
        raise RuntimeError("至少一个层单元没有可用探针")
    return sorted(range(len(rows)), key=lambda index: (-rows[index][metric], index))


def run(model_key: str, selected_name: str, commit: str) -> None:
    """用全部冻结 C 探测；不根据 V 的评分改动量化轴或 scale。"""
    root = project_root()
    config = _config(selected_name)
    four = {**config, "k_bits": 4, "v_bits": 4}
    calibration = _calibration(root, model_key, four)
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    items = _load_c_items(model_key)
    counts = {"math": 0, "retrieval": 0}
    totals: dict[str, list[dict[str, float | int]]] = {}
    lengths: dict[str, list[int]] = {"math": [], "retrieval": []}
    with torch.inference_mode():
        for item in items:
            if item["kind"] == "math":
                ids = _render_gsm(tokenizer, item["question"], model_key == "llama")
            else:
                ids = tokenizer(
                    item["text"], add_special_tokens=False, return_tensors="pt"
                ).input_ids.to("cuda")
                if ids.shape[1] != item["tokens"]:
                    raise RuntimeError(f"冻结检索长度不符：{item['id']}")
            blob = _capture(model, ids)
            sequence = int(ids.shape[1])
            n_layers = len(blob["post"])
            if not totals:
                totals = {"math": _empty(n_layers), "retrieval": _empty(n_layers)}
            kind = item["kind"]
            counts[kind] += 1
            lengths[kind].append(sequence)
            positions = torch.arange(sequence + 1, device="cuda").view(1, -1)
            for layer in range(n_layers):
                key_ref = blob["post"][layer].to("cuda")
                value_ref = blob["value"][layer].to("cuda")
                query = _rotate_query(model, blob["query"][layer], sequence).to("cuda")
                key_source = blob["pre"][layer] if config["rope"] == "pre" else blob["post"][layer]
                key_params = calibration.get("key", [None] * n_layers)[layer]
                key_hat, bad_k = _quantize(
                    key_source.to("cuda"), 4, config["key_mode"], key_params
                )
                if config["rope"] == "pre":
                    key_hat = _rotate_key(model, key_hat, positions)
                value_params = calibration.get("value", [None] * n_layers)[layer]
                value_hat, bad_v = _quantize(
                    value_ref, 4, config["value_mode"], value_params
                )
                reference, _, _ = attention_fp32(query, key_ref, value_ref)
                k_candidate, _, _ = attention_fp32(query, key_hat, value_ref)
                v_candidate, _, _ = attention_fp32(query, key_ref, value_hat)
                for side, restored, output, bad in (
                    (0, key_hat, k_candidate, bad_k),
                    (1, value_hat, v_candidate, bad_v),
                ):
                    row = totals[kind][2 * layer + side]
                    error, zeros = _relative_by_kv_head(reference, output, key_ref.shape[0])
                    row["sensitivity_sum"] += error
                    row["reconstruction_sum"] += _reconstruction(
                        key_ref if side == 0 else value_ref, restored
                    )
                    row["samples"] += 1
                    row["zero_heads"] += zeros
                    row["invalid_groups"] += int(bad)
            print(model_key, item["id"], kind, sequence, flush=True)
    if counts != {"math": 64, "retrieval": 32}:
        raise RuntimeError(f"冻结 C 数量不符：{counts}")
    summaries = {kind: _summarize(totals[kind]) for kind in counts}
    ranking_math = _ranking(summaries["math"], "sensitivity")
    ranking_retrieval = _ranking(summaries["retrieval"], "sensitivity")
    n_units = len(ranking_math)
    top = n_units // 2
    overlap = len(set(ranking_math[:top]) & set(ranking_retrieval[:top]))
    payload = {
        "protocol": "phase1-protocol v7", "commit": commit,
        "model": model_key, "selected_representation": selected_name,
        "probe_bits": 4, "counts": counts,
        "input_token_range": {
            kind: [min(lengths[kind]), max(lengths[kind])] for kind in lengths
        },
        "units": summaries,
        "rankings": {
            kind: {
                "sensitivity": _ranking(summaries[kind], "sensitivity"),
                "reconstruction": _ranking(summaries[kind], "reconstruction"),
            }
            for kind in counts
        },
        "top_half_overlap": overlap,
        "top_half_size": top,
        "ranking_transfer_warning": overlap < top / 2,
        "calibration_commit": calibration.get("commit"),
    }
    out = root / "research/phase1/results/e3" / model_key
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, "probe done", overlap, top, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
