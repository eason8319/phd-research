"""E3：从冻结 C 排名产生两档静态层分配和固定随机对照。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from kvstudy.ledger import account_pair
from kvstudy.paths import project_root
from phase1.e2_gsm import MODELS
from phase1.e2_quant import _config


def _bits(layers: int, selected: set[int]) -> list[dict[str, int]]:
    return [
        {"k": 8 if 2 * layer in selected else 4,
         "v": 8 if 2 * layer + 1 in selected else 4}
        for layer in range(layers)
    ]


def _bytes(
    assignments: list[dict[str, int]], model: dict[str, Any],
    config: dict[str, Any], residual_h: int, sequence: int,
) -> int:
    total = 0
    dim = model["hidden_size"] // model["num_attention_heads"]
    for row in assignments:
        ledger = account_pair(
            layers=1, kv_heads=model["num_key_value_heads"],
            seq=sequence, dim=dim,
            k_bits=row["k"], v_bits=row["v"],
            k_mode=config["key_mode"], v_mode=config["value_mode"],
            g_c=32, g_t=32, block_tokens=64,
            residual_h=residual_h, residual_r=128 if residual_h else 0,
        )
        total += ledger.resident_bytes
    return total


def _entry(
    name: str, assignments: list[dict[str, int]], model: dict[str, Any],
    config: dict[str, Any], residual_h: int, budget: int | None,
) -> dict[str, Any]:
    bits = [side for row in assignments for side in (row["k"], row["v"])]
    return {
        "name": name,
        "budget_int8_units": budget,
        "actual_int8_units": sum(value == 8 for value in bits),
        "nominal_bits": sum(bits) / len(bits),
        "layer_bits": assignments,
        "resident_bytes_4k": _bytes(assignments, model, config, residual_h, 4096),
        "resident_bytes_32k": _bytes(assignments, model, config, residual_h, 32768),
    }


def run(model_key: str, commit: str) -> None:
    root = project_root()
    folder = root / "research/phase1/results/e3" / model_key
    probe = json.loads((folder / "probe.json").read_text("utf-8"))
    selection = json.loads((
        root / "research/phase1/results/e2" / model_key / "residual_selection.json"
    ).read_text("utf-8"))
    selected_name = selection["e3_representation"]
    config = _config(selected_name)
    residual_h = int(selection["e3_residual_bits"])
    model = json.loads((root / MODELS[model_key] / "config.json").read_text("utf-8"))
    layers = int(model["num_hidden_layers"])
    units = 2 * layers
    if (probe["selected_representation"] != selected_name
            or len(probe["units"]["math"]) != units
            or len(probe["units"]["retrieval"]) != units):
        raise RuntimeError("层探针与 E2 选择或模型层数不一致")
    ranks = probe["rankings"]["retrieval"]
    for key in ("sensitivity", "reconstruction"):
        if sorted(ranks[key]) != list(range(units)):
            raise RuntimeError("C 的层排名不完整")
    seeds = json.loads((root / "research/phase1/splits/seeds.json").read_text("utf-8"))
    allocations: dict[str, dict[str, Any]] = {}
    for fraction in (0.25, 0.5):
        budget = int(units * fraction)
        prefix = f"p{int(fraction * 100)}"
        choices = {
            "sensitivity": set(ranks["sensitivity"][:budget]),
            "reconstruction": set(ranks["reconstruction"][:budget]),
            "inverse": set(ranks["sensitivity"][-budget:]),
        }
        for seed in seeds["e3_random"]:
            choices[f"random_{seed}"] = set(
                np.random.default_rng(seed).choice(units, budget, replace=False).tolist()
            )
        for label, selected in choices.items():
            name = f"{prefix}_{label}"
            allocations[name] = _entry(
                name, _bits(layers, selected), model, config, residual_h, budget
            )
    controls = {
        "kv8": [{"k": 8, "v": 8} for _ in range(layers)],
        "kv4": [{"k": 4, "v": 4} for _ in range(layers)],
        "k8v4": [{"k": 8, "v": 4} for _ in range(layers)],
        "k4v8": [{"k": 4, "v": 8} for _ in range(layers)],
    }
    for name, assignments in controls.items():
        allocations[name] = _entry(
            name, assignments, model, config, residual_h, None
        )
    payload = {
        "protocol": "phase1-protocol v7", "commit": commit,
        "model": model_key, "representation": selected_name,
        "residual_bits": residual_h, "n_layers": layers,
        "n_units": units, "probe_commit": probe["commit"],
        "selection_commit": selection["commit"],
        "rank_source": "c_retrieval", "random_seeds": seeds["e3_random"],
        "allocations": allocations,
    }
    (folder / "allocations.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(model_key, len(allocations), "allocations", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
