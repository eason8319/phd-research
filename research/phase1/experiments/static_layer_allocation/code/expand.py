"""按冻结顺序把验证候选及其参照扩到 128、最多 256 题。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch

from experiments.static_layer_allocation.code.protocol import require_v_plan
from experiments.static_layer_allocation.code.select_final import _canonical_condition
from kvstudy.paths import project_root
from phase1.generation import _config


def _condition(
    root: Path, model_key: str, name: str
) -> tuple[dict[str, Any] | None, int, str]:
    """读取冻结简单表示或静态分配；内容条件名与历史身份指向同一配置。"""
    name = _canonical_condition(name)
    if name == "bf16":
        return None, 0, "bf16"
    if name == "kv8":
        return _config("kv8-per_token-post"), 0, "kv8-per_token-post"
    e2_folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    if name == "e2":
        selected = json.loads((e2_folder / "residual_selection.json").read_text("utf-8"))
        return (
            _config(selected["e3_representation"]),
            int(selected["e3_residual_bits"]),
            selected["e3_representation"],
        )
    if name.startswith("e3-"):
        allocation_name = name[3:]
        folder = root / "research/phase1/experiments/static_layer_allocation/results" / model_key
        selection = json.loads((folder / "selection.json").read_text("utf-8"))
        gate = json.loads((folder / "gate.json").read_text("utf-8"))
        allowed = [selection["f_protocol_candidate"], *gate["random_for_v"]]
        if allocation_name not in allowed:
            raise RuntimeError("该静态层分配未被选中进入扩样")
        allocations = json.loads((folder / "allocations.json").read_text("utf-8"))
        spec = allocations["allocations"][allocation_name]
        config = {
            **_config(allocations["representation"]),
            "layer_bits": spec["layer_bits"],
        }
        return config, int(allocations["residual_bits"]), allocation_name
    raise ValueError(f"未知扩样条件：{name}")


def run(
    model_key: str, task: str, condition_name: str, extra_n: int, commit: str,
) -> None:
    """兼容既有扩样命令，只转入当前唯一的累计测量实现。"""
    from experiments.static_layer_allocation.code.quality import measure
    from experiments.static_layer_allocation.code.quality_data import totals

    root = project_root()
    require_v_plan(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    measure(root, model_key, task, condition_name, totals(root, task)[0] + extra_n)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
