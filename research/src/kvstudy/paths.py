"""从项目根解析路径。"""

from pathlib import Path


def project_root() -> Path:
    """返回仓库根目录。本文件位于 research/src/kvstudy/。"""
    return Path(__file__).resolve().parents[3]
