"""E2 机制核对：只在冻结 V 快照里重估 Key 范围，不生成候选结果。"""

from __future__ import annotations

import json
import sys

import torch
from transformers import AutoTokenizer

from kvstudy.attention import attention_fp32
from kvstudy.groups import fit_per_channel
from kvstudy.paths import project_root
from phase1.e1 import MODELS, _capture, _rotate_query
from phase1.e2_quant import _calibration, _config, _quantize, _rotate_key
from phase1.e2_scale_drift import _head_errors, _input_ids, _load_items, _reconstruction
from phase1.runner import load_model


def run(model_key: str, name: str, commit: str) -> None:
    """保持 Value 和 query 相同，仅替换 Key 的诊断参数；不用于配置选择。"""
    root = project_root()
    config = _config(name)
    if config["key_mode"] != "per_channel":
        raise ValueError("仅离线逐通道 Key 可做该诊断")
    calibration = _calibration(root, model_key, config)
    items = [
        item for item in _load_items(root, model_key)
        if item["subset"] in ("v_retrieval_32k", "v_hotpot")
    ]
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    ordered_ids = [item["id"] for item in items]
    source_index = [item["sha256"] for item in items]
    progress = root / "research/.runtime" / f"e2-range-refit-{model_key}-{name}.json"
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ordered_ids
                or state["source_index"] != source_index):
            raise RuntimeError("重估诊断进度与冻结输入不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ordered_ids[:len(rows)]:
            raise RuntimeError("重估诊断进度顺序不一致")
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
                frozen_key, frozen_bad = _quantize(
                    source, config["k_bits"], "per_channel", calibration["key"][layer]
                )
                refit_params = fit_per_channel(source, config["k_bits"])
                refit_key, refit_bad = _quantize(
                    source, config["k_bits"], "per_channel", refit_params
                )
                if config["rope"] == "pre":
                    frozen_key = _rotate_key(model, frozen_key, positions)
                    refit_key = _rotate_key(model, refit_key, positions)
                value_params = calibration.get("value", [None] * len(blob["post"]))[layer]
                value_hat, value_bad = _quantize(
                    value_ref, config["v_bits"], config["value_mode"], value_params
                )
                query = _rotate_query(model, blob["query"][layer], seq - 1).to("cuda")
                reference, _, _ = attention_fp32(query, key_ref, value_ref)
                frozen_output, _, _ = attention_fp32(query, frozen_key, value_hat)
                refit_output, _, _ = attention_fp32(query, refit_key, value_hat)
                layers.append({
                    "layer": layer,
                    "frozen_key_reconstruction": _reconstruction(key_ref, frozen_key),
                    "refit_key_reconstruction": _reconstruction(key_ref, refit_key),
                    "frozen_output_error_by_kv_head": _head_errors(
                        reference, frozen_output, key_ref.shape[0]
                    ),
                    "refit_output_error_by_kv_head": _head_errors(
                        reference, refit_output, key_ref.shape[0]
                    ),
                    "invalid_groups": int(frozen_bad + refit_bad + value_bad),
                })
            rows.append({
                "id": item["id"], "subset": item["subset"],
                "input_tokens": int(ids.shape[1]),
                "input_sha256": item["sha256"], "layers": layers,
            })
            progress.write_text(json.dumps({
                "commit": commit, "ids": ordered_ids,
                "source_index": source_index, "rows": rows,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            print(model_key, name, item["subset"], item["id"], flush=True)
            del blob
            torch.cuda.empty_cache()
    payload = {
        "protocol": "phase1-protocol v7", "commit": commit,
        "model": model_key, "config": name,
        "diagnostic_only": True,
        "v_refit_uses_evaluation_values": True,
        "candidate_unchanged": True,
        "calibration_commit": calibration["commit"],
        "rows": rows,
    }
    out = root / "research/phase1/results/e2" / model_key
    (out / f"range_refit_{name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
