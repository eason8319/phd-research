"""静态 Key 降位宽实验的冻结配置、输入身份与实际代码来源。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kvstudy.paths import project_root

PLAN_PATH = "research/phase1/configs/static_layer_allocation.json"
SCHEMA = "key-downgrade-allocation"


def digest(value: Any) -> str:
    """对配置或输入身份生成稳定校验值。"""
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_plan(root: Path | None = None) -> dict[str, Any]:
    """读取阶段唯一的静态层分配配置。"""
    root = root or project_root()
    plan = json.loads((root / PLAN_PATH).read_text("utf-8"))
    if plan["schema"] != SCHEMA or plan["rank_source"] != "retrieval":
        raise RuntimeError("静态层分配配置或排名来源不符")
    return plan


def require_v_plan(root: Path) -> None:
    """后续真实生成的样本、格式、预算和扩样规则未冻结时停止运行。"""
    if not load_plan(root)["v_plan"]["frozen"]:
        raise RuntimeError("Key 分配的 V 与扩样规则尚未冻结；先完成 C 的预算—KL 比较")


def input_identity(item: dict[str, Any], token_ids: list[int]) -> dict[str, Any]:
    """同时记录题面源值与实际模型输入，避免同题号不同输入混用。"""
    return {
        "id": item["id"], "kind": item["kind"],
        "source_sha256": item["source_sha256"],
        "input_tokens": len(token_ids), "input_ids_sha256": digest(token_ids),
    }


def code_sources(root: Path) -> dict[str, str]:
    """登记实际工作区代码校验，不把未提交代码冒充执行提交。"""
    paths = list((root / "research/phase1/experiments/static_layer_allocation/code").glob("*.py"))
    paths.extend(root / path for path in (
        "research/phase1/experiments/generation_error_sources/code/quantized_generation.py",
        "research/phase1/src/phase1/representation.py", "research/phase1/src/phase1/runner.py",
        "research/src/kvstudy/attention.py", "research/src/kvstudy/quantize.py",
        "research/src/kvstudy/ledger.py",
        "research/src/kvstudy/groups.py", "research/src/kvstudy/paths.py",
        "research/phase1/src/phase1/score.py", "research/phase1/src/phase1/result_format.py",
        "research/phase1/experiments/generation_error_sources/code/sources.py",
        "research/phase1/experiments/generation_error_sources/code/short_baseline.py",
        "research/phase1/experiments/generation_error_sources/code/long_baseline.py",
    ))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def frozen_sources(root: Path, model_key: str) -> dict[str, str]:
    """登记划分、模板、种子与模型清单，捕获后续复用依赖的源值变化。"""
    model_name = {"qwen": "Qwen2.5-7B-Instruct", "llama": "Llama-3.1-8B-Instruct"}[model_key]
    paths = [
        "research/phase1/splits/gsm8k.json", "research/phase1/splits/retrieval.json",
        "research/phase1/splits/templates.json", "research/phase1/splits/seeds.json",
        f"research/manifests/models/{model_name}.source.json",
    ]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def calibration_source(root: Path, model_key: str) -> dict[str, Any]:
    """核对离线 C 校准实体与已登记来源；参数加载另核对输入协议和题号。"""
    registry = root / (
        "research/phase1/experiments/generation_error_sources/results/calibration.json"
    )
    record = json.loads(registry.read_text("utf-8"))
    entry = record["files"][model_key]
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("校准实体超出项目目录")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError("冻结 C 的校准实体与来源索引不一致")
    current = root / f"research/.runtime/generation-calibration-{model_key}.pt"
    legacy = root / f"research/.runtime/e2-calibration-{model_key}.pt"
    actual = current if current.exists() else legacy
    if hashlib.sha256(actual.read_bytes()).hexdigest() != entry["sha256"]:
        raise RuntimeError("量化入口实际读取的校准实体与登记来源不一致")
    return {**entry, "commit": record["commit"], "protocol": record["protocol"]}


def existing_quality(root: Path, model_key: str, spec: dict[str, Any]) -> dict[str, Any]:
    """按相同格式与 H 保存已有开发质量状态，不预设全层 K8V4 通过。"""
    path = root / (
        "research/phase1/experiments/generation_error_sources/results/residual_diagnostics.json"
    )
    if not path.exists():
        return {"status": "unavailable", "tasks": {}}
    source = json.loads(path.read_text("utf-8"))
    cells = source.get("cells", {}).get(model_key, {}).get(spec["representation"], {})
    tasks = {}
    for task, values in cells.items():
        record = values.get(str(spec["residual_bits"]))
        tasks[task] = {
            key: record.get(key) for key in (
                "state", "class_unit_passes", "measurement_status", "path", "sha256",
                "execution_commit", "n",
            )
        } if record else {"measurement_status": "unmeasured"}
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "existing_condition_evidence", "tasks": tasks,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子保存正式小型结果，不生成报告或图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
