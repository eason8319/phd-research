"""E2：在冻结轨迹的闭合组上记录在线跨 token Key 的局部误差。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.attention import attention_fp32
from kvstudy.groups import quantize_cross_token
from kvstudy.paths import project_root
from phase1.e1 import MODELS, _capture, _rotate_query
from phase1.e2_quant import _calibration, _config, _inputs, _quantize, _rotate_key
from phase1.e2_scale_drift import _head_errors, _input_ids, _load_items
from phase1.runner import load_model

GROUP_TOKENS = 32
BLOCK_TOKENS = 64


def _group_errors(reference: torch.Tensor, candidate: torch.Tensor) -> list[float | None]:
    """按组内位置计算相对平方重建误差，输入形状为 [KV head, 32, dim]。"""
    errors = []
    for offset in range(GROUP_TOKENS):
        original = reference[:, offset, :]
        restored = candidate[:, offset, :]
        denominator = torch.sum(original ** 2)
        errors.append(
            None if float(denominator) == 0.0
            else float(torch.sum((original - restored) ** 2) / denominator)
        )
    return errors


def _relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    """计算一组张量的相对平方误差，零范数时返回空值。"""
    denominator = torch.sum(reference ** 2)
    return None if float(denominator) == 0.0 else float(
        torch.sum((reference - candidate) ** 2) / denominator
    )


def _source_index(items: list[dict[str, Any]]) -> list[str]:
    """给逐题恢复进度保存冻结的源文本校验值。"""
    return [
        item.get("sha256") or hashlib.sha256(
            (item.get("text") or item["question"]).encode("utf-8")
        ).hexdigest()
        for item in items
    ]


def _items_with_v_math(root: Path, model_key: str) -> list[dict[str, Any]]:
    """把冻结的 V 数学轨迹插入 C 与 V 长任务之间。"""
    items = _load_items(root, model_key)
    frozen = json.loads((root / "research/phase1/splits/trajectories.json").read_text("utf-8"))
    indexed = {
        item["id"]: item for item in _inputs(root, model_key, "gsm8k", "v")
    }
    v_math = [
        {**indexed[item_id], "subset": "v_math", "kind": "math"}
        for item_id in frozen["v_math"]
    ]
    if any(
        hashlib.sha256(item["question"].encode("utf-8")).hexdigest()
        != item["question_sha256"] for item in v_math
    ):
        raise RuntimeError("跨 token 的 V 数学原题校验失败")
    c_count = len(frozen["c_math"]) + len(frozen["c_retrieval"])
    if [item["id"] for item in items[:c_count]] != (
        frozen["c_math"] + frozen["c_retrieval"]
    ):
        raise RuntimeError("跨 token 轨迹的 C 冻结顺序不一致")
    return items[:c_count] + v_math + items[c_count:]


def run(model_key: str, commit: str) -> None:
    """量化各预登记输入末尾的一个完整组，并记录闭合前后差异。"""
    root = project_root()
    e2 = root / "research/phase1/results/e2" / model_key
    selection = json.loads((e2 / "h0_selection.json").read_text("utf-8"))
    name = selection["selected_low_bit"] or "kv4-per_token-post"
    config = _config(name)
    calibration = _calibration(root, model_key, config)
    if BLOCK_TOKENS % GROUP_TOKENS:
        raise RuntimeError("跨 token 组长度必须整除物理块长")
    items = _items_with_v_math(root, model_key)
    ordered_ids = [item["id"] for item in items]
    sources = _source_index(items)
    progress = root / "research/.runtime" / f"e2-cross-token-{model_key}.json"
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ordered_ids
                or state["source_index"] != sources or state["representation"] != name):
            raise RuntimeError("跨 token 轨迹进度与冻结输入或表示不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ordered_ids[:len(rows)]:
            raise RuntimeError("跨 token 逐题进度顺序不符")
    else:
        rows = []
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    with torch.inference_mode():
        for item in items[len(rows):]:
            ids = _input_ids(tokenizer, model_key, item)
            blob = _capture(model, ids)
            seq = int(ids.shape[1]) + 1
            boundary = ((seq - 1) // GROUP_TOKENS) * GROUP_TOKENS
            if boundary < GROUP_TOKENS:
                raise RuntimeError(f"轨迹不足以跨组边界：{item['id']}")
            start = boundary - GROUP_TOKENS
            positions = torch.arange(start, boundary, device="cuda").view(1, -1)
            layers = []
            for layer in range(len(blob["post"])):
                source = (
                    blob["pre"][layer] if config["rope"] == "pre"
                    else blob["post"][layer]
                )[:, start:boundary, :].to("cuda")
                original = blob["post"][layer][:, start:boundary, :].to("cuda")
                value = blob["value"][layer][:, start:boundary, :].to("cuda")
                restored, stats = quantize_cross_token(
                    source, config["k_bits"], GROUP_TOKENS
                )
                if config["rope"] == "pre":
                    restored = _rotate_key(model, restored, positions)
                value_params = calibration.get("value", [None] * len(blob["value"]))[layer]
                value_restored, value_invalid = _quantize(
                    value, config["v_bits"], config["value_mode"], value_params
                )
                query = _rotate_query(model, blob["query"][layer], seq - 1).to("cuda")
                reference_output, _, _ = attention_fp32(query, original, value)
                key_only_output, _, _ = attention_fp32(query, restored, value)
                closed_output, _, _ = attention_fp32(query, restored, value_restored)
                layers.append({
                    "layer": layer,
                    "key_invalid_groups": stats.invalid_groups,
                    "key_constant_groups": stats.constant_groups,
                    "value_invalid_groups": int(value_invalid),
                    "group_start_token": start,
                    "closure_token_count": boundary,
                    "before_closure_error": 0.0,
                    "after_closure_position_error": _group_errors(original, restored),
                    "next_open_token_error": 0.0,
                    "value_relative_error": _relative_error(value, value_restored),
                    "local_32_key_only_error_by_kv_head": _head_errors(
                        reference_output, key_only_output, original.shape[0]
                    ),
                    "local_32_attention_error_by_kv_head": _head_errors(
                        reference_output, closed_output, original.shape[0]
                    ),
                })
            row = {
                "id": item["id"], "subset": item["subset"],
                "input_tokens": int(ids.shape[1]),
                "source_sha256": sources[len(rows)],
                "boundary": boundary,
                "open_tokens_at_query": seq - boundary,
                "layers": layers,
            }
            rows.append(row)
            progress.write_text(json.dumps({
                "commit": commit, "ids": ordered_ids,
                "source_index": sources, "representation": name,
                "rows": rows,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            print(model_key, item["subset"], item["id"], flush=True)
            del blob
            torch.cuda.empty_cache()
    payload = {
        "protocol": "phase1-protocol v7", "commit": commit,
        "model": model_key, "source": "pre_registered_c_v_trajectories",
        "representation": name, "key_mode": "cross_token",
        "key_bits": config["k_bits"], "value_mode": config["value_mode"],
        "value_bits": config["v_bits"], "rope": config["rope"],
        "group_tokens": GROUP_TOKENS, "block_tokens": BLOCK_TOKENS,
        "boundary_method": "last_complete_group_before_first_generated_query",
        "attention_scope": "local_closed_group_key_and_selected_value",
        "calibration_commit": calibration.get("commit"),
        "rows": rows,
    }
    e2.mkdir(parents=True, exist_ok=True)
    (e2 / "cross_token_gt32.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
