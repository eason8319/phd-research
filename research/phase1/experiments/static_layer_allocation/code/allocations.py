"""按三种格式的检索排名冻结仅 K 侧的三档预算、六种策略与完整成本。"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

from kvstudy.ledger import account_pair
from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.metrics import ranking
from experiments.static_layer_allocation.code.protocol import (
    SCHEMA, code_sources, digest, frozen_sources, load_plan, write_json,
)


def layer_bits(layers: int, k4_layers: list[int]) -> list[dict[str, int]]:
    """只选择 K4 层，其他 K8，全部 V4。"""
    selected = set(k4_layers)
    if len(selected) != len(k4_layers) or not selected <= set(range(layers)):
        raise ValueError("K4 层名单无效")
    return [{"k": 4 if layer in selected else 8, "v": 4} for layer in range(layers)]


def cost(
    assignments: list[dict[str, int]], model: dict[str, Any],
    config: dict[str, Any], residual_h: int, sequence: int,
) -> dict[str, Any]:
    """复用冻结账本，分别记容量及残留替代尾部／全读的传输界，单位为字节。"""
    dim = model["hidden_size"] // model["num_attention_heads"]
    heads = model["num_key_value_heads"]
    totals = {key: 0 for key in ("k_payload", "v_payload", "k_params", "v_params", "residual")}
    padding = 0
    read_full = 0
    skipped = 0
    main_write = 0
    residual_write = 0
    for row in assignments:
        common = {
            "layers": 1, "kv_heads": heads, "dim": dim,
            "k_bits": row["k"], "v_bits": row["v"],
            "k_mode": config["key_mode"], "v_mode": "per_token",
            "g_c": 32, "g_t": 32, "block_tokens": 64,
            "residual_h": residual_h, "residual_r": 128 if residual_h else 0,
        }
        ledger = account_pair(seq=sequence, **common)
        next_ledger = account_pair(seq=sequence + 1, **common)
        for key in totals:
            region = getattr(ledger, key)
            totals[key] += region.raw_bytes
            padding += region.padding_bytes
            read_full += region.raw_bytes
        for key in ("k_payload", "v_payload", "k_params", "v_params"):
            main_write += getattr(next_ledger, key).raw_bytes - getattr(ledger, key).raw_bytes
        if residual_h:
            window = min(sequence, 128)
            for bits, mode in ((row["k"], config["key_mode"]), (row["v"], "per_token")):
                skipped += heads * window * dim * bits // 8
                if mode == "per_token":
                    skipped += heads * window * math.ceil(dim / 32) * 4
            residual_write += 2 * heads * dim * residual_h // 8
            if residual_h == 8:
                residual_write += 2 * heads * math.ceil(dim / 32) * 4
    return {
        "source": "layout_count", "length": sequence, "batch": 1,
        "components_raw_bytes": totals, "padding_bytes": padding,
        "resident_bytes": sum(totals.values()) + padding,
        "decode_read_lower_bytes": read_full - skipped,
        "decode_read_full_bytes": read_full,
        "decode_main_write_bytes": main_write,
        "decode_residual_write_bytes": residual_write,
        "decode_write_bytes": main_write + residual_write,
        "prefill_conversion_write_raw_bytes": sum(totals.values()),
        "pre_rope_mul_adds_per_read": 3 * len(assignments) * heads * sequence * dim
        if config["rope"] == "pre" else 0,
        "alignment_bytes": 128, "block_tokens": 64,
    }


def build_choices(
    rows: list[dict[str, Any]], budgets: list[int], seeds: list[int],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, list[int]]]:
    """低误差层降位宽，反向取高误差层；每个种子只生成一次排列。"""
    layers = len(rows)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("随机对照必须使用三个不同的冻结种子")
    if any(not 0 < r < layers for r in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("K4 预算无效")
    low = ranking(rows, "sensitivity", descending=False)
    high = ranking(rows, "sensitivity", descending=True)
    reconstruction = ranking(rows, "reconstruction", descending=False)
    permutations = {
        str(seed): np.random.default_rng(seed).permutation(layers).tolist() for seed in seeds
    }
    choices = {
        f"k4_layers_{r}": {
            "sensitivity": low[:r], "reconstruction": reconstruction[:r], "inverse": high[:r],
            **{f"random_{seed}": permutation[:r] for seed, permutation in permutations.items()},
        } for r in budgets
    }
    return choices, permutations


def run(model_key: str, commit: str) -> None:
    """拒绝旧 K/V 探针；所有位宽表在全层测量之前一次冻结。"""
    root = project_root()
    plan = load_plan(root)
    if plan.get("measurement") == "teacher_forced_answer_content":
        raise RuntimeError("当前协议在 content 入口按独立校准折冻结位宽表")
    folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
    probe = json.loads((folder / "probe.json").read_text("utf-8"))
    if (probe.get("schema") != SCHEMA or probe["model"] != model_key
            or probe["plan_sha256"] != digest(plan)):
        raise RuntimeError("必须重算三格式的 K4V4−K8V4 探针，旧探针不能生成新分配")
    if probe["frozen_sources"] != frozen_sources(root, model_key):
        raise RuntimeError("探针之后冻结输入、模板、种子或模型清单发生变化")
    from phase1.representation import MODELS
    from experiments.generation_error_sources.code.quantized_generation import _config

    model = json.loads((root / MODELS[model_key] / "config.json").read_text("utf-8"))
    layers = int(model["num_hidden_layers"])
    if layers != plan["models"][model_key]["layers"]:
        raise RuntimeError("模型层数与冻结预算不一致")
    seed_pack = json.loads((root / plan["random_seeds_source"]).read_text("utf-8"))
    seeds = seed_pack[plan["random_seeds_key"]]
    allocations: dict[str, Any] = {}
    formats: dict[str, Any] = {}
    for name, spec in plan["formats"].items():
        measured = probe["formats"][name]
        if any(measured[key] != spec[key] for key in spec):
            raise RuntimeError("探针的格式或残留与当前条件不一致")
        choices, permutations = build_choices(
            measured["layers"]["retrieval"], plan["models"][model_key]["k4_layers"], seeds
        )
        config = _config(spec["representation"])
        groups: dict[str, list[str]] = {}
        canonical: dict[str, str] = {}
        for budget, strategies in choices.items():
            groups[budget] = []
            for strategy, selected in strategies.items():
                key = f"{name}_{budget}_{strategy}"
                assignments = layer_bits(layers, selected)
                bit_digest = digest(assignments)
                canonical.setdefault(bit_digest, key)
                allocations[key] = {
                    "format": name, "strategy": strategy, "k4_layers": sorted(selected),
                    "r": len(selected), "actual_int8_units": layers - len(selected),
                    "p_int8": (layers - len(selected)) / (2 * layers),
                    "nominal_bits": 4 + 2 * (layers - len(selected)) / layers,
                    "layer_bits": assignments, "layer_bits_sha256": bit_digest,
                    "measurement_alias": canonical[bit_digest],
                    "cost": {
                        str(length): cost(assignments, model, config, spec["residual_bits"], length)
                        for length in plan["cost_lengths"]
                    },
                }
                groups[budget].append(key)
            for length in plan["cost_lengths"]:
                costs = {digest(allocations[key]["cost"][str(length)]) for key in groups[budget]}
                if len(costs) != 1:
                    raise RuntimeError("同格式同预算的六策略完整成本不一致")
        for control, selected in (("k8v4", []), ("kv4", list(range(layers)))):
            key = f"{name}_{control}"
            assignments = layer_bits(layers, selected)
            bit_digest = digest(assignments)
            canonical.setdefault(bit_digest, key)
            allocations[key] = {
                "format": name, "strategy": control, "r": len(selected),
                "k4_layers": selected, "layer_bits": assignments,
                "layer_bits_sha256": bit_digest, "measurement_alias": canonical[bit_digest],
                "cost": {
                    str(length): cost(assignments, model, config, spec["residual_bits"], length)
                    for length in plan["cost_lengths"]
                },
            }
        formats[name] = {
            **spec, "budget_groups": groups, "random_permutations": permutations,
            "existing_quality": measured["existing_quality"],
        }
    payload = {
        "schema": SCHEMA, "protocol": plan["input_protocol"], "commit": commit,
        "model": model_key, "n_layers": layers, "rank_source": "retrieval",
        "model_geometry": {key: model[key] for key in (
            "hidden_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers",
        )},
        "plan_sha256": digest(plan), "probe_sha256": digest(probe),
        "probe_commit": probe["commit"], "input_digest": probe["input_digest"],
        "model_config_sha256": probe["model_config_sha256"],
        "frozen_sources": probe["frozen_sources"],
        "calibration_source": probe["calibration_source"], "random_seeds": seeds,
        "code_sha256": code_sources(root), "formats": formats, "allocations": allocations,
        "duplicate_configurations": {
            key: row["measurement_alias"] for key, row in allocations.items()
            if key != row["measurement_alias"]
        },
        "v_plan_frozen": plan["v_plan"]["frozen"],
        "reference_cost": {
            "kv8": {
                str(length): cost(
                    [{"k": 8, "v": 8} for _ in range(layers)], model,
                    {"key_mode": "per_token", "rope": "post"}, 0, length,
                ) for length in plan["cost_lengths"]
            },
            "bf16": {
                str(length): {
                    "source": "layout_count", "resident_bytes":
                    4 * layers * model["num_key_value_heads"] * length
                    * (model["hidden_size"] // model["num_attention_heads"]),
                    "quantization_parameters_bytes": 0, "residual_bytes": 0,
                } for length in plan["cost_lengths"]
            },
        },
    }
    payload["allocation_digest"] = digest(payload)
    write_json(folder / "allocations.json", payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("qwen", "llama"))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.model, args.commit)
