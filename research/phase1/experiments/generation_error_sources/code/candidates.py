"""确定真实生成的共同候选范围，并检查重新选择所需的结果覆盖。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.representation_interactions.code.select import select_representations

KV8 = "kv8-per_token-post"
KV4 = "kv4-per_token-post"
TASKS = ("gsm8k", "hotpot")


def generation_candidates(matrix_payload: dict[str, Any]) -> tuple[str, ...]:
    """使用当前共同候选规则；历史 kept 字段仅属于原运行记录。"""
    retained = select_representations(matrix_payload["summary"])
    return (KV8, KV4, *retained.kept)


def require_generation_results(folder: Path, active_candidates: list[str]) -> None:
    """缺测时停止重新选择；active_candidates 已排除核验通过的提前停止配置。"""
    required_files = [folder / "v_retrieval_bf16.json"]
    for task in TASKS:
        required_files.append(folder / f"v_{task}_bf16.json")
        required_files.extend(folder / f"v_{task}_{name}.json" for name in active_candidates)
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise RuntimeError("四种 K8V4 的生成对照尚未齐备：" + ", ".join(missing_files))
