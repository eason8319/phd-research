"""真实生成与误差来源：在冻结轨迹上区分单层误差、固定 token 累积误差与自由生成分歧。"""

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
from experiments.generation_error_sources.code import quantized_generation as quant_module
from experiments.generation_error_sources.code.cross_token_groups import (
    _items_with_v_math,
    _source_index,
)
from experiments.generation_error_sources.code.short_baseline import _eos_ids
from experiments.generation_error_sources.code.long_baseline import _generate as generate_bf16
from experiments.generation_error_sources.code.quantized_generation import (
    MAX_CONTEXT,
    _calibration,
    _config,
    _decode_step,
    _generate as generate_quantized,
    _prefill_cache,
    _quantize,
    _rotate_key,
)
from experiments.generation_error_sources.code.scale_drift import _head_errors, _input_ids
from experiments.generation_error_sources.code.sources import _runtime_path
from phase1.representation import MODELS, _capture, _rotate_query
from phase1.runner import load_model


def _frozen_sources(root: Path, items: list[dict[str, Any]]) -> list[str]:
    """核对数学题原文哈希，并返回全部轨迹的冻结来源索引。"""
    split = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    indexed = {
        row["id"]: row["question_sha256"] for group in ("c", "v")
        for row in split[group]
    }
    for item in items:
        if item["kind"] != "math":
            continue
        digest = hashlib.sha256(item["question"].encode("utf-8")).hexdigest()
        if digest != indexed[item["id"]]:
            raise RuntimeError(f"冻结数学题原文不一致：{item['id']}")
    return _source_index(items)


def _input_hash(
    root: Path, tokenizer: Any, model_key: str, item: dict[str, Any]
) -> str:
    """计算实际送入 tokenizer 的完整题面 SHA256。"""
    if item["kind"] == "math":
        template = json.loads(
            (root / "research/phase1/splits/templates.json").read_text("utf-8")
        )["gsm8k"]
        user = template.format(question=item["question"])
        kwargs = {"date_string": "26 Jul 2024"} if model_key == "llama" else {}
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, **kwargs,
        )
    else:
        source = item.get("text") or item["path"].read_text("utf-8")
        expected = item.get("sha256")
        if expected and hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise RuntimeError(f"轨迹全文校验失败：{item['id']}")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _concentration(errors: list[float | None]) -> dict[str, float | int | None]:
    """记录最高误差 KV head 的占比与非零 head 数，不把零范数当作误差。"""
    positive = [max(0.0, value) for value in errors if value is not None]
    total = sum(positive)
    return {
        "largest_head_share": max(positive) / total if total > 0 else None,
        "nonzero_heads": sum(value > 0 for value in positive),
        "defined_heads": len(positive),
    }


def _local_layers(
    model: Any,
    blob: dict[str, Any],
    config: dict[str, Any],
    calibration: dict[str, Any],
    input_tokens: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], list[torch.Tensor]]:
    """在相同 BF16 隐藏状态上逐层单独干预，并保存首个生成 query 的参考输出。"""
    n_layers = len(blob["post"])
    positions = torch.arange(input_tokens + 1, device="cuda").view(1, -1)
    layers: list[dict[str, Any]] = []
    reference_outputs: list[torch.Tensor] = []
    reference_queries: list[torch.Tensor] = []
    for layer in range(n_layers):
        key_ref = blob["post"][layer].to("cuda")
        value_ref = blob["value"][layer].to("cuda")
        query = _rotate_query(model, blob["query"][layer], input_tokens).to("cuda")
        source = blob["pre"][layer] if config["rope"] == "pre" else blob["post"][layer]
        key_params = calibration.get("key", [None] * n_layers)[layer]
        key_hat, bad_key = _quantize(
            source.to("cuda"), config["k_bits"], config["key_mode"], key_params
        )
        if config["rope"] == "pre":
            key_hat = _rotate_key(model, key_hat, positions)
        value_params = calibration.get("value", [None] * n_layers)[layer]
        value_hat, bad_value = _quantize(
            value_ref, config["v_bits"], config["value_mode"], value_params
        )
        reference, _, _ = attention_fp32(query, key_ref, value_ref)
        key_only, _, _ = attention_fp32(query, key_hat, value_ref)
        value_only, _, _ = attention_fp32(query, key_ref, value_hat)
        combined, _, _ = attention_fp32(query, key_hat, value_hat)
        kv_heads = key_ref.shape[0]
        combined_errors = _head_errors(reference, combined, kv_heads)
        layers.append({
            "layer": layer,
            "invalid_key_groups": int(bad_key),
            "invalid_value_groups": int(bad_value),
            "key_only_error_by_kv_head": _head_errors(reference, key_only, kv_heads),
            "value_only_error_by_kv_head": _head_errors(
                reference, value_only, kv_heads
            ),
            "combined_error_by_kv_head": combined_errors,
            "combined_concentration": _concentration(combined_errors),
        })
        reference_outputs.append(reference.float().cpu())
        reference_queries.append(query.float().cpu())
    return layers, reference_outputs, reference_queries


def _fixed_replay(
    model: Any,
    ids: torch.Tensor,
    first_token: int,
    config: dict[str, Any],
    calibration: dict[str, Any],
    reference_outputs: list[torch.Tensor],
    reference_queries: list[torch.Tensor],
) -> tuple[list[dict[str, Any]], int | None, int]:
    """固定 BF16 首 token，全层量化一步并与同位置 BF16 轨迹比较。"""
    first, keys, values, read_keys, read_values, invalid = _prefill_cache(
        model, ids, config, calibration, 2, 0
    )
    if first != first_token:
        raise RuntimeError("固定回放的 BF16 首 token 与参考轨迹不同")
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_attention = quant_module.attention_from_batch

    def capture(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """记录全层回放的 query 与注意力输出，不改变解码计算。"""
        attended = original_attention(query, key, value)
        captured.append((
            query[0, :, 0, :].detach().float().cpu(),
            attended[0, 0].detach().float().cpu().reshape(
                query.shape[1], query.shape[-1]
            ),
        ))
        return attended

    quant_module.attention_from_batch = capture
    try:
        next_token, bad = _decode_step(
            model, keys, values, read_keys, read_values,
            first_token, int(ids.shape[1]), config, calibration, 0,
        )
    finally:
        quant_module.attention_from_batch = original_attention
    if len(captured) != len(reference_outputs):
        raise RuntimeError("固定回放未经过全部层")
    layers = []
    kv_heads = keys[0].shape[1]
    for layer, ((query, output), reference, ref_query) in enumerate(zip(
        captured, reference_outputs, reference_queries
    )):
        errors = _head_errors(reference, output, kv_heads)
        layers.append({
            "layer": layer,
            "query_error_by_kv_head": _head_errors(ref_query, query, kv_heads),
            "attention_output_error_by_kv_head": errors,
            "attention_concentration": _concentration(errors),
        })
    del keys, values, read_keys, read_values
    return layers, next_token, invalid + bad


def _first_divergence(reference: list[int], candidate: list[int]) -> int | None:
    """返回从零开始的首个不同位置；共同前缀后长度不同时返回其长度。"""
    for index, (old, new) in enumerate(zip(reference, candidate)):
        if old != new:
            return index
    return min(len(reference), len(candidate)) if len(reference) != len(candidate) else None


def _free_generation(
    model: Any,
    ids: torch.Tensor,
    eos: set[int],
    limit: int,
    config: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """分别自由生成，只记录分歧位置与停止信息，不保存完整预测。"""
    reference, reference_stop = generate_bf16(model, ids, eos, limit)
    candidate, candidate_stop, invalid = generate_quantized(
        model, ids[0].tolist(), eos, limit, config, calibration, 0
    )
    return {
        "bf16_new_tokens": len(reference),
        "candidate_new_tokens": len(candidate),
        "bf16_stop": reference_stop,
        "candidate_stop": candidate_stop,
        "candidate_invalid_groups": invalid,
        "first_divergence_index": _first_divergence(reference, candidate),
    }


def run(model_key: str, commit: str) -> None:
    """对真实生成与误差来源选定的低位宽表示运行预登记轨迹，并逐题保存可恢复进度。"""
    root = project_root()
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    selection = json.loads((folder / "no_residual_selection.json").read_text("utf-8"))
    name = selection["selected_low_bit"]
    if name is None:
        raise RuntimeError("低位宽全部淘汰时不运行所选表示的自由生成轨迹")
    config = _config(name)
    calibration = _calibration(root, model_key, config)
    items = _items_with_v_math(root, model_key)
    ordered_ids = [item["id"] for item in items]
    sources = _frozen_sources(root, items)
    if len(items) != 32:
        raise RuntimeError("真实生成与误差来源预登记轨迹数量不是 32")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    input_hashes = [_input_hash(root, tokenizer, model_key, item) for item in items]
    progress = _runtime_path(
        root, f"generation-trajectories-{model_key}.json", f"e2-trajectories-{model_key}.json"
    )
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ordered_ids
                or state["source_index"] != sources
                or state["input_hashes"] != input_hashes
                or state["representation"] != name):
            raise RuntimeError("真实生成与误差来源轨迹进度与代码、输入或表示不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != ordered_ids[:len(rows)]:
            raise RuntimeError("真实生成与误差来源轨迹逐题顺序不一致")
    else:
        rows = []
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    with torch.inference_mode():
        for item in items[len(rows):]:
            ids = _input_ids(tokenizer, model_key, item)
            limit = 128 if item["kind"] == "retrieval" else 1024
            if ids.shape[1] + limit > MAX_CONTEXT:
                raise RuntimeError(f"轨迹输入超出总窗口：{item['id']}")
            torch.cuda.reset_peak_memory_stats()
            blob = _capture(model, ids)
            local, reference_outputs, reference_queries = _local_layers(
                model, blob, config, calibration, int(ids.shape[1])
            )
            fixed, next_token, fixed_invalid = _fixed_replay(
                model, ids, blob["token"], config, calibration,
                reference_outputs, reference_queries,
            )
            del blob, reference_outputs, reference_queries
            torch.cuda.empty_cache()
            free = _free_generation(model, ids, eos, limit, config, calibration)
            row = {
                "id": item["id"],
                "subset": item["subset"],
                "input_tokens": int(ids.shape[1]),
                "source_sha256": sources[len(rows)],
                "input_sha256": input_hashes[len(rows)],
                "query_position": int(ids.shape[1]),
                "device_name": torch.cuda.get_device_name(),
                "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "local_layers": local,
                "fixed_replay_layers": fixed,
                "fixed_next_token": next_token,
                "fixed_invalid_groups": fixed_invalid,
                "free_generation": free,
            }
            rows.append(row)
            progress.write_text(json.dumps({
                "commit": commit, "ids": ordered_ids, "source_index": sources,
                "input_hashes": input_hashes, "representation": name, "rows": rows,
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            print(model_key, item["subset"], len(rows), item["id"], flush=True)
            del ids
            torch.cuda.empty_cache()
    payload = {
        "protocol": "phase1-full-key-retrieval",
        "commit": commit,
        "model": model_key,
        "representation": name,
        "selection_commit": selection["commit"],
        "h": 0,
        "n": len(rows),
        "subset_counts": {
            subset: sum(row["subset"] == subset for row in rows)
            for subset in (
                "c_math", "c_retrieval", "v_math", "v_retrieval_32k", "v_hotpot"
            )
        },
        "source": "pre_registered_c_v_trajectories",
        "calibration_commit": calibration.get("commit"),
        "rows": rows,
    }
    (folder / "trajectories.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
