"""核查近零 KL 的实际词表分布及直接高精度计算，固定首末及最大 KL 检索题复算全部配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.generation_error_sources.code.quantized_generation import _calibration, _config
from experiments.static_layer_allocation.code.gate import (
    _candidate_logits, _kl, _prefill_source,
)
from experiments.static_layer_allocation.code.protocol import (
    digest, input_identity, load_plan, write_json,
)
from phase1.representation import MODELS, _load_c_items
from phase1.runner import decode_step, load_model


def _direct_kl(reference: torch.Tensor, candidate: torch.Tensor, dtype: Any) -> float:
    left = reference.detach().double().cpu().numpy().reshape(-1).astype(dtype)
    right = candidate.detach().double().cpu().numpy().reshape(-1).astype(dtype)
    left -= np.max(left)
    right -= np.max(right)
    old = left - np.log(np.sum(np.exp(left)))
    new = right - np.log(np.sum(np.exp(right)))
    return float(np.sum(np.exp(old) * (old - new)))


def _reference_stats(logits: torch.Tensor, tokenizer: Any) -> dict[str, Any]:
    values = logits.detach().double().cpu().numpy().reshape(-1)
    shifted = values - np.max(values)
    logp = shifted - np.log(np.sum(np.exp(shifted)))
    probabilities = np.exp(logp)
    return {
        "maximum_probability": float(np.max(probabilities)),
        "entropy_nat": float(-np.sum(probabilities * logp)),
        "argmax_token": int(np.argmax(probabilities)),
        "argmax_text": tokenizer.decode([int(np.argmax(probabilities))]),
    }


def run(model_key: str, commit: str) -> None:
    """保存全检索参考分布与预先固定三类题的数值核验，不更改预算或策略。"""
    root = project_root()
    plan = load_plan(root)
    folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    allocation = json.loads((folder / "allocations.json").read_text("utf-8"))
    gate_path = folder / "gate.json"
    gate = json.loads(gate_path.read_text("utf-8"))
    configs = {name: _config(spec["representation"])
               for name, spec in allocation["formats"].items()}
    calibrated = {
        name: _calibration(root, model_key, {
            **config, "layer_bits": [{"k": 4, "v": 4}, {"k": 8, "v": 4}],
        }) for name, config in configs.items()
    }
    items = [item for item in _load_c_items(model_key) if item["kind"] == "retrieval"]
    expected = {row["id"]: row for row in gate["rows"] if row["kind"] == "retrieval"}
    if len(items) != plan["counts"]["retrieval"] or len(expected) != len(items):
        raise RuntimeError("数值核验检索样本数不符")
    largest = max(range(len(items)), key=lambda i: max(expected[items[i]["id"]]["kl"].values()))
    selected = {0, len(items) - 1, largest}
    precision = float(np.finfo(np.longdouble).eps)
    if precision >= np.finfo(np.float64).eps:
        raise RuntimeError("直接复算环境的 longdouble 精度未高于 FP64")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    references = []
    checked = []
    with torch.inference_mode():
        for index, item in enumerate(items):
            ids = tokenizer(item["text"], add_special_tokens=False, return_tensors="pt") \
                .input_ids.to("cuda")
            identity = input_identity(item, ids[0].tolist())
            original = expected[item["id"]]
            if {key: original[key] for key in identity} != identity:
                raise RuntimeError("数值核验的实际输入与正式 KL 不一致")
            source = _prefill_source(model, ids)
            if source["token"] != original["query_token"]:
                raise RuntimeError("数值核验的共同首 token 改变")
            reference, _, _ = decode_step(
                model, source["keys"], source["values"], source["token"], int(ids.shape[1])
            )
            references.append({
                **identity, "query_token": source["token"],
                "query_token_text": tokenizer.decode([source["token"]]),
                **_reference_stats(reference, tokenizer),
            })
            if index in selected:
                measurements = {}
                logits, invalid = _candidate_logits(
                    model, ids, source["token"], _config(plan["kv8_reference"]), 0, {}, source
                )
                candidates = [("bf16", reference, 0), ("kv8", logits, invalid)]
                for name, spec in allocation["allocations"].items():
                    if name != spec["measurement_alias"]:
                        continue
                    fmt = spec["format"]
                    logits, invalid = _candidate_logits(
                        model, ids, source["token"],
                        {**configs[fmt], "layer_bits": spec["layer_bits"]},
                        allocation["formats"][fmt]["residual_bits"], calibrated[fmt], source,
                    )
                    candidates.append((name, logits, invalid))
                for name, logits, invalid in candidates:
                    if invalid or logits is None:
                        raise RuntimeError("数值核验出现无效候选")
                    stable = _kl(reference, logits)
                    if stable != original["kl"][name]:
                        raise RuntimeError(f"同输入复算的 KL 改变：{model_key} {name}")
                    direct = _direct_kl(reference, logits, np.longdouble)
                    direct64 = _direct_kl(reference, logits, np.float64)
                    error = abs(stable - direct)
                    if error > 100 * np.finfo(np.float64).eps * max(1.0, stable):
                        raise RuntimeError("稳定 KL 与直接扩展精度复算不一致")
                    measurements[name] = {
                        "stable_float64_kl": stable, "direct_longdouble_kl": direct,
                        "direct_float64_kl": direct64, "absolute_error_longdouble": error,
                    }
                checked.append({**identity, "measurements": measurements})
            del source
            print("numerical", model_key, index + 1, len(items), flush=True)
    write_json(folder / "numerical_check.json", {
        "status": "passed", "model": model_key, "commit": commit,
        "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        "allocation_digest": allocation["allocation_digest"],
        "input_digest": gate["input_digest"],
        "selection": "first_last_and_largest_full_matrix_KL_frozen_C_retrieval_questions",
        "reference_n": len(references), "checked_n": len(checked),
        "longdouble_epsilon": precision,
        "reference_distributions": references, "checked_questions": checked,
        "max_absolute_error_longdouble": max(
            row["absolute_error_longdouble"] for sample in checked
            for row in sample["measurements"].values()
        ),
        "numerical_code_sha256": hashlib.sha256(
            (root / "research/phase1/experiments/static_layer_allocation/code/numerical_check.py")
            .read_bytes()
        ).hexdigest(),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("qwen", "llama"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.model, args.commit)

