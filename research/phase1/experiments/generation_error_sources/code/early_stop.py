"""把真实生成与误差来源短题的严重失效中止记录为可核验的部分观察。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import numpy as np

from kvstudy.intervals import TOLERANCE_POINTS, binary_drop_interval
from kvstudy.paths import project_root
from experiments.generation_error_sources.code.sources import _runtime_path


def run(model_key: str, name: str, generation_commit: str, record_commit: str) -> None:
    """只在配对下降下界已越过容差时登记提前停止，不外推为完整 V 分数。"""
    root = project_root()
    runtime = _runtime_path(
        root, f"generation-{model_key}-gsm8k-v-{name}.json", f"e2-{model_key}-gsm8k-v-{name}.json"
    )
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    progress = json.loads(runtime.read_text(encoding="utf-8"))
    reference = json.loads((folder / "v_gsm8k_bf16.json").read_text(encoding="utf-8"))
    kv8 = json.loads((folder / "v_gsm8k_kv8-per_token-post.json").read_text(encoding="utf-8"))
    rows = progress["rows"]
    n = len(rows)
    if progress["commit"] != generation_commit or not 8 <= n < 64:
        raise RuntimeError("中止进度的版本或观察数不符")
    if len(progress["ids"]) != 64 or len(progress["source_index"]) != 64:
        raise RuntimeError("中止进度不是完整冻结 V 列表")
    if reference["n"] != 64 or kv8["n"] != 64:
        raise RuntimeError("参照不是完整冻结 V 列表")
    ids = [row["id"] for row in rows]
    if (ids != progress["ids"][:n]
            or ids != [row["id"] for row in reference["rows"][:n]]
            or ids != [row["id"] for row in kv8["rows"][:n]]):
        raise RuntimeError("中止进度和参照的 ID 或顺序不符")
    for row, counterpart in zip(rows, kv8["rows"]):
        if (row["input_sha256"] != counterpart["input_sha256"]
                or row["input_tokens"] != counterpart["input_tokens"]):
            raise RuntimeError("中止进度和 KV8 的输入不一致")
    if any(row["stop"] != "limit" or row["correct"] or row["invalid_groups"] for row in rows):
        raise RuntimeError("当前进度不满足生成耗尽且失分的严重失效观察")
    candidate_scores = np.asarray([row["correct"] for row in rows], dtype=bool)
    versus_bf16 = binary_drop_interval(
        np.asarray([row["correct"] for row in reference["rows"][:n]], dtype=bool),
        candidate_scores,
    )
    versus_kv8 = binary_drop_interval(
        np.asarray([row["correct"] for row in kv8["rows"][:n]], dtype=bool),
        candidate_scores,
    )
    if not any(
        value.lower is not None and value.lower > TOLERANCE_POINTS
        for value in (versus_bf16, versus_kv8)
    ):
        raise RuntimeError("配对下降下界尚未越过预设容差，不能提前停止")
    payload = {
        "protocol": "phase1-full-key-retrieval",
        "commit": record_commit,
        "generation_commit": generation_commit,
        "model": model_key,
        "config": name,
        "task": "v-gsm8k",
        "planned_n": 64,
        "observed_n": n,
        "rows": rows,
        "stopped_early": True,
        "reason": "生成均达到上限且失分；配对下降区间下界越过 2 分容差",
        "versus_bf16": asdict(versus_bf16),
        "versus_kv8": asdict(versus_kv8),
        "invalid_samples": 0,
    }
    out = folder / f"v_gsm8k_{name}_early_stop.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(model_key, name, n, versus_bf16.lower, versus_kv8.lower, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
