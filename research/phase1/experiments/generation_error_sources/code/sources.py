"""按内容命名读取运行缓存，并兼容整理前的路径与输入约定标识。"""

from pathlib import Path


PROTOCOL_ALIASES = {
    "phase1-gsm8k-answer-example": "phase1-protocol v5",
    "phase1-hotpot-direct-answer": "phase1-protocol v6",
    "phase1-full-key-retrieval": "phase1-protocol v7",
}


def _protocol_matches(actual: str, expected: str) -> bool:
    """接受同一输入约定的现名与历史标识，不重写源文件。"""
    return actual in (expected, PROTOCOL_ALIASES.get(expected))


def _runtime_path(root: Path, current_name: str, legacy_name: str) -> Path:
    """新名尚未迁移时继续读取已有历史缓存；新运行使用内容名。"""
    folder = root / "research/.runtime"
    current = folder / current_name
    legacy = folder / legacy_name
    return legacy if not current.exists() and legacy.exists() else current
