"""全层应用三格式、三预算的冻结 Key 位宽表，比较共同首 token 后的词表 KL。"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.generation_error_sources.code.quantized_generation import (
    _calibration, _config, _decode_step, _prefill_cache,
)
from experiments.static_layer_allocation.code.metrics import distribution_summary, vocabulary_kl
from experiments.static_layer_allocation.code.protocol import (
    SCHEMA, calibration_source, code_sources, digest, frozen_sources, input_identity,
    load_plan, write_json,
)
from phase1.representation import MODELS, _load_c_items, _render_gsm
from phase1.runner import cache_lists, decode_step, load_model, modeling_module, prefill


def _prefill_source(model: Any, ids: torch.Tensor) -> dict[str, Any]:
    """捕获一次共同高精度 prefill，候选各自量化，原始张量不被候选改写。"""
    module = modeling_module(model)
    original = module.apply_rotary_pos_emb
    pre_keys = []

    def capture(query: torch.Tensor, key: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        pre_keys.append(key.detach())
        return original(query, key, *args, **kwargs)

    module.apply_rotary_pos_emb = capture
    try:
        outputs = prefill(model, ids)
    finally:
        module.apply_rotary_pos_emb = original
    keys, values = cache_lists(outputs.past_key_values)
    if len(pre_keys) != len(keys):
        raise RuntimeError("共同 prefill 未捕获全部层旋转前 Key")
    return {"input_ids": ids, "keys": keys, "values": values, "pre_keys": pre_keys,
            "token": int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())}


def _kl(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """全部对照统一用 FP64 稳定 KL，保留同词表且同首 token 的分布。"""
    return vocabulary_kl(
        reference.detach().double().cpu().numpy(), candidate.detach().double().cpu().numpy()
    )


def _candidate_logits(
    model: Any, ids: torch.Tensor, token: int,
    config: dict[str, Any], residual_h: int, calibrated: dict[str, Any],
    prefill_source: dict[str, Any] | None = None,
) -> tuple[torch.Tensor | None, int]:
    """从高精度 prefill 构建全层缓存，固定首 token 后独立执行一步 decode。"""
    first, keys, values, read_keys, read_values, invalid = _prefill_cache(
        model, ids, config, calibrated, 1, residual_h, prefill_source=prefill_source
    )
    if first != token:
        raise RuntimeError("候选与高精度的第一个生成 token 不一致")
    if invalid:
        return None, invalid
    captured: list[torch.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured.append(output.detach())

    hook = model.lm_head.register_forward_hook(capture)
    try:
        _, bad = _decode_step(
            model, keys, values, read_keys, read_values,
            token, int(ids.shape[1]), config, calibrated, residual_h,
        )
    finally:
        hook.remove()
    if bad:
        return None, bad
    if len(captured) != 1:
        raise RuntimeError("未捕获唯一的 next-token logits")
    return captured[0], 0


def run(model_key: str, commit: str) -> None:
    """完整扫描全部预算；结果只给方案比较，不自动产生质量通过或 V 候选。"""
    root = project_root()
    plan = load_plan(root)
    folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    allocations = json.loads((folder / "allocations.json").read_text("utf-8"))
    probe = json.loads((folder / "probe.json").read_text("utf-8"))
    if (allocations.get("schema") != SCHEMA or allocations["model"] != model_key
            or allocations["plan_sha256"] != digest(plan)
            or allocations["probe_sha256"] != digest(probe)):
        raise RuntimeError("全层测量必须使用当前三格式探针和冻结 Key 分配")
    if digest({key: value for key, value in allocations.items()
               if key != "allocation_digest"}) != allocations["allocation_digest"]:
        raise RuntimeError("冻结位宽表内容与其登记校验值不一致")
    if probe["frozen_sources"] != frozen_sources(root, model_key):
        raise RuntimeError("探针与全层测量的输入、模板、种子或模型来源不一致")
    source = calibration_source(root, model_key)
    if source != allocations["calibration_source"]:
        raise RuntimeError("探针与全层测量的冻结 C 校准实体不一致")
    configs = {
        name: _config(spec["representation"]) for name, spec in allocations["formats"].items()
    }
    calibrated = {
        name: _calibration(root, model_key, {
            **config, "layer_bits": [{"k": 4, "v": 4}, {"k": 8, "v": 4}],
        }) for name, config in configs.items()
    }
    specs = allocations["allocations"]
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    if digest(model.config.to_dict()) != allocations["model_config_sha256"]:
        raise RuntimeError("探针与全层测量的模型配置不一致")
    items = _load_c_items(model_key)
    expected = probe["input_identity"]
    if [item["id"] for item in items] != [item["id"] for item in expected]:
        raise RuntimeError("探针与全层测量的冻结 C 顺序不一致")
    provenance = {
        "commit": commit, "allocation_digest": allocations["allocation_digest"],
        "input_digest": allocations["input_digest"], "plan_sha256": digest(plan),
        "code_sha256": code_sources(root), "calibration_source": source,
        "frozen_sources": probe["frozen_sources"],
    }
    progress = root / "research/.runtime" / f"key-allocation-kl-{model_key}.json"
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if state["provenance"] != provenance:
            raise RuntimeError("全层 KL 恢复点与输入、代码、校准或分配不一致")
        rows = state["rows"]
        if [row["id"] for row in rows] != [item["id"] for item in expected[:len(rows)]]:
            raise RuntimeError("全层 KL 恢复顺序不一致")
    else:
        rows = []
    with torch.inference_mode():
        for index, item in enumerate(items):
            ids = _render_gsm(tokenizer, item["question"], model_key == "llama") \
                if item["kind"] == "math" else tokenizer(
                    item["text"], add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(model.model.embed_tokens.weight.device)
            identity = input_identity(item, ids[0].tolist())
            if identity != expected[index]:
                raise RuntimeError(f"探针与全层 KL 的实际输入不一致：{item['id']}")
            if index < len(rows):
                continue
            source_cache = _prefill_source(model, ids)
            token = source_cache["token"]
            first_format = next(iter(probe["formats"].values()))
            if token != first_format["samples"][index]["query_token"]:
                raise RuntimeError("探针与全层测量的共同高精度首 token 不一致")
            reference, _, _ = decode_step(
                model, source_cache["keys"], source_cache["values"], token, int(ids.shape[1])
            )
            measured: dict[str, float | None] = {"bf16": _kl(reference, reference)}
            invalid = {"bf16": 0}
            kv8, bad = _candidate_logits(
                model, ids, token, _config(plan["kv8_reference"]), 0, {}, source_cache
            )
            measured["kv8"] = None if kv8 is None else _kl(reference, kv8)
            invalid["kv8"] = bad
            for name, spec in specs.items():
                canonical = spec["measurement_alias"]
                if canonical != name:
                    measured[name] = measured[canonical]
                    invalid[name] = invalid[canonical]
                    continue
                format_name = spec["format"]
                candidate, bad = _candidate_logits(
                    model, ids, token,
                    {**configs[format_name], "layer_bits": spec["layer_bits"]},
                    allocations["formats"][format_name]["residual_bits"],
                    calibrated[format_name],
                    source_cache,
                )
                measured[name] = None if candidate is None else _kl(reference, candidate)
                invalid[name] = bad
            del source_cache
            rows.append({**identity, "query_token": token, "query_position": int(ids.shape[1]),
                         "kl": measured, "invalid_groups": invalid})
            write_json(progress, {"commit": commit, "provenance": provenance, "rows": rows})
            print(model_key, item["id"], item["kind"], len(rows), flush=True)
    counts = {kind: sum(row["kind"] == kind for row in rows) for kind in plan["counts"]}
    if counts != plan["counts"]:
        raise RuntimeError("全层 KL 未覆盖完整冻结 C")
    payload = {
        "schema": SCHEMA, "protocol": plan["input_protocol"], "model": model_key,
        **provenance, "allocation_commit": allocations["commit"], "counts": counts,
        "kl_unit": "nat", "kl_precision": "float64_stable_normalized_identity",
        "logits_source": "common_bf16_first_token_then_one_decode",
        "statistical_unit": "question", "main_subset": "retrieval",
        **distribution_summary(rows, allocations["formats"], allocations["random_seeds"]),
        "duplicate_configurations": allocations["duplicate_configurations"], "rows": rows,
        "existing_quality": {
            name: spec["existing_quality"] for name, spec in allocations["formats"].items()
        },
        "v_plan_frozen": False, "quality_threshold": None,
        "generation_screen": "not_frozen", "candidate_status": "await_separate_validation_plan",
        "endpoints": "same_format_one_step_only_no_full_generation_rerun",
    }
    write_json(folder / "gate.json", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.model, args.commit)
