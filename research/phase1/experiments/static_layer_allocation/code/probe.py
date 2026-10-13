"""在共同高精度轨迹上测量三种格式的逐层 K4V4 与 K8V4 输出变化。"""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.static_layer_allocation.code.metrics import grouped_rse, rank_agreement, ranking
from experiments.static_layer_allocation.code.protocol import (
    SCHEMA,
    calibration_source,
    code_sources,
    digest,
    existing_quality,
    frozen_sources,
    input_identity,
    load_plan,
    write_json,
)
from kvstudy.attention import attention_fp32
from kvstudy.paths import project_root
from phase1.generation import (
    RESIDUAL_TOKENS,
    _calibration,
    _config,
    _layer_params,
    _quantize,
    _recent,
    _rotate_key,
)
from phase1.representation import MODELS, _capture, _load_c_items, _render_gsm, _rotate_query
from phase1.runner import load_model


def _reconstruction(reference: torch.Tensor, candidate: torch.Tensor) -> float | None:
    """沿用逐 KV head 的 K4 相对重建误差；pre 条件在读时旋转后测量。"""
    numerator = torch.sum((reference.double() - candidate.double()) ** 2, dim=(1, 2))
    denominator = torch.sum(reference.double() ** 2, dim=(1, 2))
    valid = denominator > 0
    return float((numerator[valid] / denominator[valid]).mean()) if bool(valid.any()) else None


def _layer_measurement(
    model: Any, blob: dict[str, Any], layer: int, config: dict[str, Any],
    calibrated: dict[str, Any], residual_h: int, sequence: int,
) -> dict[str, Any]:
    """K8/K4 均从相同源值直接量化；读取残留时按现有实现替换同一末尾窗口。"""
    device = model.model.embed_tokens.weight.device
    key_ref = blob["post"][layer].to(device)
    value_ref = blob["value"][layer].to(device)
    key_source = blob[config["rope"]][layer].to(device)
    positions = torch.arange(sequence + 1, device=device).view(1, -1)
    query = _rotate_query(model, blob["query"][layer], sequence).to(device)
    keys: dict[int, torch.Tensor] = {}
    invalid = 0
    for bits in (8, 4):
        keys[bits], bad = _quantize(
            key_source, bits, config["key_mode"], _layer_params(calibrated, "key", layer, bits)
        )
        invalid += int(bad)
        if config["rope"] == "pre":
            keys[bits] = _rotate_key(model, keys[bits], positions)
    # 重建对照沿用历史 post-RoPE 主存张量口径，残留替换前计入全部位置。
    reconstruction = _reconstruction(key_ref, keys[4]) if not invalid else None
    value_four, bad = _quantize(value_ref, 4, "per_token", None)
    invalid += int(bad)
    if residual_h:
        start = max(0, sequence + 1 - RESIDUAL_TOKENS)
        recent_k, bad_k = _recent(key_ref[:, start:], residual_h)
        recent_v, bad_v = _recent(value_ref[:, start:], residual_h)
        invalid += int(bad_k) + int(bad_v)
        for key in keys.values():
            key[:, start:] = recent_k
        value_four[:, start:] = recent_v
    row: dict[str, Any] = {
        "layer": layer, "invalid_groups": invalid, "reconstruction": reconstruction,
        "sensitivity": None, "k8v4_error": None, "k4v4_error": None, "zero_heads": 0,
    }
    if invalid:
        return row
    reference = attention_fp32(query, key_ref, value_ref)[0].cpu().numpy()
    eight = attention_fp32(query, keys[8], value_four)[0].cpu().numpy()
    four = attention_fp32(query, keys[4], value_four)[0].cpu().numpy()
    row["sensitivity"], row["zero_heads"] = grouped_rse(
        reference, four, eight, key_ref.shape[0]
    )
    row["k8v4_error"] = grouped_rse(reference, eight, reference, key_ref.shape[0])[0]
    row["k4v4_error"] = grouped_rse(reference, four, reference, key_ref.shape[0])[0]
    return row


def _summarize(samples: list[dict[str, Any]], layers: int) -> dict[str, Any]:
    """先平均题内有效组，再等权平均题；无效题保留并阻止冻结排名。"""
    subsets: dict[str, Any] = {}
    for kind in ("math", "retrieval"):
        selected = [row for row in samples if row["kind"] == kind]
        rows = []
        for layer in range(layers):
            measured = [sample["layers"][layer] for sample in selected]
            row = {
                "layer": layer, "samples": len(measured),
                "zero_heads": sum(value["zero_heads"] for value in measured),
                "invalid_groups": sum(value["invalid_groups"] for value in measured),
            }
            for metric in ("sensitivity", "reconstruction", "k8v4_error", "k4v4_error"):
                values = [value[metric] for value in measured]
                defined = [value for value in values if value is not None]
                row[metric] = (
                    sum(defined) / len(defined) if defined and not row["invalid_groups"] else None
                )
                row[metric + "_valid_questions"] = len(defined)
            rows.append(row)
        subsets[kind] = rows
    ranks: dict[str, Any] = {}
    for kind, rows in subsets.items():
        try:
            ranks[kind] = {
                metric: ranking(rows, metric, descending=True)
                for metric in ("sensitivity", "reconstruction")
            }
        except RuntimeError as error:
            ranks[kind] = {"status": "unavailable", "reason": str(error)}
    agreement: dict[str, Any] = {}
    if all("sensitivity" in value for value in ranks.values()):
        agreement = {
            "math_vs_retrieval_sensitivity": rank_agreement(
                ranks["math"]["sensitivity"], ranks["retrieval"]["sensitivity"], layers // 2
            ),
            "retrieval_sensitivity_vs_reconstruction": rank_agreement(
                ranks["retrieval"]["sensitivity"], ranks["retrieval"]["reconstruction"], layers // 2
            ),
        }
    return {"layers": subsets, "rankings_descending": ranks, "rank_agreement": agreement}


def run(model_key: str, commit: str) -> None:
    """测量全部冻结 C；每个条件独立排名，不根据数学排名改动分配来源。"""
    root = project_root()
    plan = load_plan(root)
    if plan.get("measurement") == "teacher_forced_answer_content":
        raise RuntimeError("当前协议使用 content 入口测量正确答案内容，旧单步探针已停止")
    layers = plan["models"][model_key]["layers"]
    configs = {name: _config(spec["representation"]) for name, spec in plan["formats"].items()}
    source = calibration_source(root, model_key)
    calibration = {
        name: _calibration(root, model_key, {
            **config, "layer_bits": [{"k": 4, "v": 4}, {"k": 8, "v": 4}],
        }) for name, config in configs.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    if len(model.model.layers) != layers:
        raise RuntimeError("模型层数与冻结预算不一致")
    inputs = []
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in configs}
    with torch.inference_mode():
        for item in _load_c_items(model_key):
            ids = _render_gsm(tokenizer, item["question"], model_key == "llama") \
                if item["kind"] == "math" else tokenizer(
                    item["text"], add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(model.model.embed_tokens.weight.device)
            sequence = int(ids.shape[1])
            if item["kind"] == "retrieval" and sequence != item["tokens"]:
                raise RuntimeError(f"冻结检索长度不符：{item['id']}")
            identity = input_identity(item, ids[0].tolist())
            inputs.append(identity)
            blob = _capture(model, ids)
            for name, config in configs.items():
                samples[name].append({
                    **identity, "query_token": blob["token"], "query_position": sequence,
                    "layers": [
                        _layer_measurement(
                            model, blob, layer, config, calibration[name],
                            plan["formats"][name]["residual_bits"], sequence,
                        ) for layer in range(layers)
                    ],
                })
            print(model_key, item["id"], item["kind"], sequence, flush=True)
    counts = {kind: sum(row["kind"] == kind for row in inputs) for kind in plan["counts"]}
    if counts != plan["counts"]:
        raise RuntimeError(f"冻结 C 数量不符：{counts}")
    payload = {
        "schema": SCHEMA, "protocol": plan["input_protocol"], "commit": commit,
        "model": model_key, "plan_sha256": digest(plan), "code_sha256": code_sources(root),
        "counts": counts, "input_identity": inputs, "input_digest": digest(inputs),
        "model_config_sha256": digest(model.config.to_dict()),
        "frozen_sources": frozen_sources(root, model_key),
        "calibration_source": source,
        "reference": "common_high_precision_trajectory",
        "reconstruction_location": "post_rope_low_bit_main_before_residual_override",
        "attention_precision": "float32", "metric_accumulation": "float64",
        "aggregation": "valid_query_groups_then_questions_equal_weight",
        "formats": {
            name: {
                **plan["formats"][name], **_summarize(rows, layers), "samples": rows,
                "existing_quality": existing_quality(root, model_key, plan["formats"][name]),
            }
            for name, rows in samples.items()
        },
        "quality_status_source": "research/phase1/experiments/generation_error_sources/results",
        "quality_status": "retain_existing_per_condition_status_no_presumed_pass",
    }
    out = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    write_json(out / "probe.json", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.model, args.commit)
