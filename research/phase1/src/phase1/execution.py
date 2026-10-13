"""发布可核对的阶段执行快照，保留历史快照与真实测量来源。"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from phase1.reuse import _Computation, computation_signature, digest, file_hash, function_signature


CURRENT = "research/.runtime/phase1-execution/current.json"


def source_files(root: Path) -> list[Path]:
    """只发布代码、配置和冻结输入清单；测量结果独立管理。"""
    folders = (
        "research/src", "research/phase1/src", "research/tests", "research/phase1/configs",
        "research/phase1/splits", "research/manifests", "research/env",
    )
    files = []
    for folder in folders:
        files.extend(path for path in (root / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts
                     and path.suffix in (".py", ".json", ".md", ".txt", ".toml", ".yml"))
    files.extend(path for path in (root / "research/phase1/experiments").rglob("*.py")
                 if ("code" in path.parts or path.name == "__init__.py")
                 and "__pycache__" not in path.parts)
    files.append(root / "research/phase1/guide.md")
    return sorted(set(files))


def artifact_files(root: Path) -> list[Path]:
    """列出发布所需源码、当前结果、报告和报告直接引用的项目材料。"""
    files = set(source_files(root))
    markdown = []
    for experiment in (root / "research/phase1/experiments").iterdir():
        for folder in ("results", "report"):
            files.update(path for path in (experiment / folder).rglob("*") if path.is_file())
        markdown.extend((experiment / "report").glob("*.md"))
    index = root / "research/phase1/source_index.json"
    if index.exists():
        files.add(index)
    for document in markdown:
        references = re.findall(r"`((?:research|docs)/[^`\n]+)`", document.read_text("utf-8"))
        for relative in references:
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or not path.exists():
                raise RuntimeError(f"发布所需报告来源失效：{relative}")
            if path.is_file():
                files.add(path)
    return sorted(files)


def publish(root: Path, commit: str) -> dict[str, Any]:
    """按实际文件校验发布；保留旧运行并让后续入口选择新快照。"""
    if os.name == "nt":
        raise RuntimeError("执行快照只发布到服务器项目，不在本地创建运行目录")
    if (root / ".execution.json").exists():
        raise RuntimeError("只能从维护树发布，不能改写已有执行快照")
    files = {path.relative_to(root).as_posix(): file_hash(path) for path in source_files(root)}
    identity = digest(files)
    relative = f"research/.runtime/phase1-execution/{identity[:16]}/execution_root"
    target = root / relative
    publication = {
        "schema": "phase1-source-publication", "base_commit": commit,
        "identity": "base_commit_plus_actual_file_sha256", "files": files,
        "source_root": relative, "source_digest": identity,
        "computation_signature": computation_signature(root),
        "bf16_computation_signature": computation_signature(root, False),
    }
    if target.exists():
        existing = json.loads((target / ".execution.json").read_text("utf-8"))
        if existing != publication:
            raise RuntimeError("已有同名执行快照与本次发布不同")
        verify_publication(root, existing)
    else:
        for path in artifact_files(root):
            destination = target / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (target / "models").symlink_to(root / "models", target_is_directory=True)
        (target / "research/.runtime").symlink_to(root / "research/.runtime",
                                                target_is_directory=True)
        (target / ".execution.json").write_text(
            json.dumps(publication, ensure_ascii=False, indent=2) + "\n", "utf-8",
        )
    current = root / CURRENT
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_text(json.dumps(publication, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return publication


def workspace_root(root: Path) -> Path:
    """从项目内运行快照定位维护树；不读取项目上级或其他目录。"""
    parts = root.resolve().parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ("research", ".runtime"):
            return Path(*parts[:index])
    return root.resolve()


def current_publication(root: Path) -> dict[str, Any]:
    """执行快照使用自身来源；维护树读取当前发布，兼容旧收尾快照。"""
    if (root / ".execution.json").exists():
        return json.loads((root / ".execution.json").read_text("utf-8"))
    current = workspace_root(root) / CURRENT
    if current.exists():
        return json.loads(current.read_text("utf-8"))
    return json.loads((root / "research/.runtime/static-final/publication.json").read_text(
        "utf-8",
    ))


def verify_publication(root: Path, publication: dict[str, Any]) -> Path:
    """核对原执行快照；目录迁移不修改历史发布清单。"""
    workspace = workspace_root(root)
    relative = publication.get("source_root", "research/.runtime/static-final/execution_root")
    source = (workspace / relative).resolve()
    if not source.is_relative_to(workspace) or not source.is_dir():
        raise RuntimeError("执行来源不存在或超出本项目")
    files = publication["files"]
    if isinstance(files, list):
        files = {item["path"]: item["sha256"] for item in files}
    for name, expected in files.items():
        path = (source / name).resolve()
        if not path.is_relative_to(workspace) or file_hash(path) != expected:
            raise RuntimeError(f"冻结执行来源发生改变：{name}")
    return source


def verify_recorded_code(root: Path, relative: str, expected: str) -> Path:
    """代码提取后核对原快照与等价实现，返回真实测量源码，不改写来源校验。"""
    current = root / relative
    if file_hash(current) == expected:
        return current
    index = json.loads((root / "research/phase1/source_index.json").read_text("utf-8"))
    alignments = [index.get("execution_alignment", {}),
                  *index.get("execution_alignment_history", []),
                  index.get("previous_execution_alignment", {})]
    for alignment in alignments:
        publication = alignment.get("publication", {})
        if publication.get("files", {}).get(relative) != expected:
            continue
        source = verify_publication(root, publication)
        historical = source / relative
        old_tree = ast.parse(historical.read_text("utf-8-sig"))
        new_tree = ast.parse(current.read_text("utf-8-sig"))
        present = {node.name for node in new_tree.body if isinstance(node, ast.FunctionDef)}
        aliases = {item.asname or item.name: item.name for node in new_tree.body
                   if isinstance(node, ast.ImportFrom) and node.module == "kvstudy.artifacts"
                   for item in node.names}
        for node in old_tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in present:
                continue
            imported = aliases.get(node.name)
            shared = root / "research/src/kvstudy/artifacts.py"
            if not imported or function_signature(historical, node.name) != function_signature(
                shared, imported,
            ):
                raise RuntimeError(f"记录源码与当前实现不兼容：{relative}/{node.name}")
            new_tree.body.append(copy.deepcopy(node))
        normalized = []
        for tree in (old_tree, new_tree):
            tree = _Computation().visit(tree)
            functions = sorted((node for node in tree.body if isinstance(node, ast.FunctionDef)),
                               key=lambda node: node.name)
            tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
            tree.body.extend(functions)
            normalized.append(ast.dump(tree, include_attributes=False))
        if normalized[0] != normalized[1]:
            raise RuntimeError(f"记录源码与当前实现不兼容：{relative}")
        return historical
    raise RuntimeError(f"缺少可核对的真实执行源码：{relative}")


def validate_segments(root: Path, value: dict[str, Any]) -> None:
    """累计结果按实际源码段追溯；兼容恢复必须保持同一计算实现。"""
    segments = value.get("source_segments") or [{
        "start": 0, "end": value["n"], "publication": value["publication"],
    }]
    offset = 0
    for segment in segments:
        if segment["start"] != offset or segment["end"] <= offset:
            raise RuntimeError("测量源码分段缺失、重复或倒序")
        source = verify_publication(root, segment["publication"])
        if computation_signature(source) != computation_signature(root):
            raise RuntimeError("恢复段的数值实现不同，不能拼接已有测量")
        offset = segment["end"]
    if offset != value["n"]:
        raise RuntimeError("测量源码分段未覆盖全部记录")
