"""真实生成与误差来源：用冻结 C/V 快照检查离线 Key 范围漂移与注意力误差。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.attention import attention_fp32
from kvstudy.paths import project_root
from kvstudy.quantize import STATUS_INVALID, STATUS_NORMAL
from experiments.generation_error_sources.code.long_baseline import _samples
from experiments.generation_error_sources.code.quantized_generation import (
    _calibration,
    _config,
    _quantize,
    _rotate_key,
)
from experiments.generation_error_sources.code.sources import _runtime_path
from phase1.representation import MODELS, _capture, _load_c_items, _render_gsm, _rotate_query
from phase1.runner import load_model


def _head_errors(
    reference: torch.Tensor, candidate: torch.Tensor, kv_heads: int
) -> list[float | None]:
    repeats = reference.shape[0] // kv_heads
    result = []
    for head in range(kv_heads):
        sl = slice(head * repeats, (head + 1) * repeats)
        denominator = torch.sum(reference[sl] ** 2)
        result.append(
            None if float(denominator) == 0.0
            else float(torch.sum((reference[sl] - candidate[sl]) ** 2) / denominator)
        )
    return result


def _reconstruction(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    denominator = torch.sum(reference ** 2)
    return None if float(denominator) == 0.0 else float(
        torch.sum((reference - candidate) ** 2) / denominator
    )


def _load_items(root: Path, model_key: str) -> list[dict[str, Any]]:
    frozen = json.loads((root / "research/phase1/splits/trajectories.json").read_text("utf-8"))
    c_items = {item["id"]: item for item in _load_c_items(model_key)}
    result = []
    for subset, kind in (("c_math", "math"), ("c_retrieval", "retrieval")):
        for item_id in frozen[subset]:
            result.append({**c_items[item_id], "subset": subset, "kind": kind})
    for subset, task in (("v_retrieval_32k", "retrieval"), ("v_hotpot", "hotpot")):
        indexed = {item["id"]: item for item in _samples(root, model_key, task, "v")}
        for item_id in frozen[subset]:
            result.append({**indexed[item_id], "subset": subset, "kind": task})
    return result


def _input_ids(
    tokenizer: Any, model_key: str, item: dict[str, Any]
) -> torch.Tensor:
    if item["kind"] == "math":
        return _render_gsm(tokenizer, item["question"], model_key == "llama")
    if "text" in item:
        source = item["text"]
    else:
        source = item["path"].read_text("utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != item["sha256"]:
            raise RuntimeError(f"快照全文校验失败：{item['id']}")
    ids = tokenizer(source, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    if ids.shape[1] != item["tokens"]:
        raise RuntimeError(f"快照 token 长度不符：{item['id']}")
    return ids


def run(model_key: str, name: str, commit: str) -> None:
    """只记录机制快照；不以 V 的范围重新拟合任何参数。"""
    root = project_root()
    config = _config(name)
    if config["key_mode"] != "per_channel":
        raise ValueError("范围漂移诊断只用于离线逐通道 Key")
    calibration = _calibration(root, model_key, config)
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    items = _load_items(root, model_key)
    ordered_ids = [item["id"] for item in items]
    source_index = [
        item.get("sha256") or hashlib.sha256(
            (item.get("text") or item["question"]).encode("utf-8")
        ).hexdigest()
        for item in items
    ]
    progress = _runtime_path(
        root,
        f"generation-scale-drift-{model_key}-{name}.json",
        f"e2-scale-drift-{model_key}-{name}.json",
    )
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ordered_ids
                or state["source_index"] != source_index):
            raise RuntimeError("范围漂移进度与当前代码或冻结输入不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ordered_ids[:len(rows)]:
            raise RuntimeError("范围漂移逐题进度顺序不符")
    else:
        rows = []
    with torch.inference_mode():
        for item in items[len(rows):]:
            ids = _input_ids(tokenizer, model_key, item)
            blob = _capture(model, ids)
            seq = int(ids.shape[1]) + 1
            positions = torch.arange(seq, device="cuda").view(1, -1)
            layers = []
            for layer in range(len(blob["post"])):
                source = (
                    blob["pre"][layer] if config["rope"] == "pre"
                    else blob["post"][layer]
                ).to("cuda")
                key_ref = blob["post"][layer].to("cuda")
                value_ref = blob["value"][layer].to("cuda")
                scale, offset, status = calibration["key"][layer]
                low = offset.float().unsqueeze(1)
                high = low + scale.float().unsqueeze(1) * (2 ** config["k_bits"] - 1)
                outside = ((source < low) | (source > high)) & (
                    status == STATUS_NORMAL
                ).unsqueeze(1)
                key_hat, bad_k = _quantize(
                    source, config["k_bits"], config["key_mode"],
                    calibration["key"][layer],
                )
                if config["rope"] == "pre":
                    key_hat = _rotate_key(model, key_hat, positions)
                value_params = calibration.get("value", [None] * len(blob["post"]))[layer]
                value_hat, bad_v = _quantize(
                    value_ref, config["v_bits"], config["value_mode"], value_params
                )
                query = _rotate_query(model, blob["query"][layer], seq - 1).to("cuda")
                reference, _, _ = attention_fp32(query, key_ref, value_ref)
                candidate, _, _ = attention_fp32(query, key_hat, value_hat)
                layers.append({
                    "layer": layer,
                    "key_outside_fraction": float(outside.float().mean()),
                    "key_outside_by_kv_head": [
                        float(outside[head].float().mean()) for head in range(source.shape[0])
                    ],
                    "key_reconstruction": _reconstruction(key_ref, key_hat),
                    "value_reconstruction": _reconstruction(value_ref, value_hat),
                    "attention_output_error_by_kv_head": _head_errors(
                        reference, candidate, key_ref.shape[0]
                    ),
                    "invalid_groups": int(bad_k + bad_v),
                    "invalid_calibrated_channels": int((status == STATUS_INVALID).sum()),
                })
            rows.append({
                "id": item["id"], "subset": item["subset"],
                "input_tokens": int(ids.shape[1]), "layers": layers,
            })
            progress.write_text(json.dumps({
                "commit": commit, "ids": ordered_ids,
                "source_index": source_index, "rows": rows,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            print(model_key, name, item["subset"], item["id"], flush=True)
            del blob
            torch.cuda.empty_cache()
    payload = {
        "protocol": "phase1-full-key-retrieval", "commit": commit,
        "model": model_key, "config": name,
        "source": "pre_registered_c_v_trajectories",
        "calibration_commit": calibration["commit"],
        "rows": rows,
    }
    out = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scale_drift_{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
