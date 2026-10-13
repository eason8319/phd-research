"""各阶段共用的小型 JSON 实体读写与稳定校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON，不修改原始测量。"""
    return json.loads(path.read_text("utf-8"))


def file_hash(path: Path) -> str:
    """返回实际文件字节的 SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    """对配置和身份记录生成稳定校验，拒绝非有限数值。"""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """原子保存小型实体；校验失败时保留旧文件。"""
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, "utf-8")
    temporary.replace(path)
