"""第一阶段唯一结果定位、参考视图与已审核测量复用。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kvstudy.artifacts import digest, file_hash, read_json
from phase1.reuse import POLICY, computation_signature, reference_path


EXPERIMENTS = (
    "representation_interactions", "generation_error_sources",
    "static_layer_allocation", "storage_service_cost",
)
BASE = "research/phase1/experiments"
INDEX = "research/phase1/source_index.json"


def project_path(root: Path, relative: str) -> Path:
    """解析项目内实体，拒绝绝对路径、父目录与越界符号链接。"""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"结果来源超出项目：{relative}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise RuntimeError(f"结果来源超出项目：{relative}")
    return target


def quality_path(root: Path, model: str, task: str, condition: str) -> Path:
    """参考只保存追加行，静态候选保存累计实体。"""
    folder = "reference_expansion" if condition in ("bf16", "kv8", "k8v4") else "final_quality"
    return root / BASE / "static_layer_allocation/results" / folder / model / (
        f"{task}_{condition}.json"
    )


def reference_view(
    root: Path, model: str, task: str, condition: str, total: int,
    references: dict[str, Any],
) -> dict[str, Any]:
    """只在内存组合首轮与追加行，同时核对真实来源、计数和输入身份。"""
    if total < 1:
        raise ValueError("参考请求数必须为正")
    path = reference_path(root, model, task, condition, references)
    base = read_json(path)
    rows = list(base["rows"])
    relative = path.relative_to(root).as_posix()
    sources = {relative: file_hash(path)}
    if base["n"] != len(rows) or base["model"] != model:
        raise RuntimeError("首轮参考计数或模型身份异常")
    if base["task"].removeprefix("v-") != task:
        raise RuntimeError("首轮参考任务身份异常")
    if total > len(rows):
        extra_path = quality_path(root, model, task, condition)
        extra = read_json(extra_path)
        if extra["reference_source"] != {"path": relative, "sha256": sources[relative],
                                         "n": len(rows)}:
            raise RuntimeError("追加参考的首轮来源发生改变")
        if (extra["n"] != len(extra["rows"]) or extra["model"] != model
                or extra["task"] != task or extra["condition"] != condition):
            raise RuntimeError("追加参考计数或测量身份异常")
        rows.extend(extra["rows"])
        sources[extra_path.relative_to(root).as_posix()] = file_hash(extra_path)
    if len(rows) < total or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("参考缺少必要样本或存在重复身份")
    if any(not row.get("input_sha256") or not row.get("input_tokens") for row in rows):
        raise RuntimeError("参考缺少冻结输入身份")
    return {"n": total, "rows": rows[:total], "sources": sources}


def review_bindings(root: Path) -> dict[str, str]:
    """绑定审核所用实现、配置、划分和模型清单，调度与说明编辑不使测量失效。"""
    from phase1.execution import source_files

    orchestration = {"entry.py", "execution.py", "audit.py", "results.py", "reuse.py"}
    paths = []
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        if "/tests/" in relative or path.suffix == ".md":
            continue
        if path.parent == root / "research/phase1/src/phase1" and path.name in orchestration:
            continue
        paths.append(path)
    return {path.relative_to(root).as_posix(): file_hash(path) for path in paths}


class ResultCatalog:
    """用当前来源索引与只读审核回执复用正式结果，不扫描历史运行当作新结果。"""

    def __init__(self, root: Path) -> None:
        """读取唯一当前索引；不存在索引时已有文件仍先进入核验。"""
        self.root = root
        path = root / INDEX
        self.index = read_json(path) if path.exists() else {}
        self.files = {item["current_path"]: item["current_sha256"]
                      for item in self.index.get("files", [])}
        if len(self.files) != len(self.index.get("files", [])):
            raise RuntimeError("当前结果索引包含重复实体")
        self._review_reason: str | None = None

    def inventory(self, experiment: str) -> list[dict[str, Any]]:
        """列出当前实体及证据角色，所有消费者共用同一归属。"""
        if experiment not in EXPERIMENTS:
            raise ValueError("未知实验")
        roles = {item["current_path"]: item.get("evidence_role")
                 for item in self.index.get("files", [])}
        folder = self.root / BASE / experiment / "results"
        return [{"path": path.relative_to(self.root).as_posix(), "sha256": file_hash(path),
                 "role": roles.get(path.relative_to(self.root).as_posix())}
                for path in sorted(folder.rglob("*.json"))]

    def _review(self) -> str:
        """复用前核对全链路实体与审核依赖；变化只请求核验，不默认重测。"""
        if self._review_reason is not None:
            return self._review_reason
        proof = self.index.get("reuse_review", {})
        reason = ""
        if proof.get("status") != "passed" or proof.get("results_digest") != digest(self.files):
            reason = "缺少当前正式结果的完整审核回执"
        elif proof.get("shared_inputs_digest") != digest(
            self.index.get("shared_capacity_inputs", {}),
        ):
            reason = "共享容量输入事实已变化"
        elif proof.get("dependencies") != review_bindings(self.root):
            reason = "审核所用实现、模型、输入或配置已变化"
        elif proof.get("computation_signature") != computation_signature(self.root):
            reason = "数值实现已变化，先核验受影响的测量"
        else:
            actual = {item["path"]: item["sha256"] for name in EXPERIMENTS
                      for item in self.inventory(name)}
            if actual != self.files:
                reason = "正式结果与已审核来源索引不一致"
        self._review_reason = reason
        return reason

    def decision(self, experiment: str, relative: str | None = None) -> dict[str, Any]:
        """返回已完成包或单一实体的复用决定，不启动模型、重算结果或写文件。"""
        outputs = self.inventory(experiment)
        if relative is not None:
            project_path(self.root, relative)
            if not relative.startswith(f"{BASE}/{experiment}/results/"):
                raise ValueError("结果不属于指定实验")
            outputs = [item for item in outputs if item["path"] == relative]
        if not outputs:
            return {"action": "measure", "reason": "缺少该条件", "existing_results": []}
        reason = self._review()
        if not reason:
            reviewed = self.index["reuse_review"]["audit"]["experiments"]
            if experiment not in reviewed:
                reason = "该实验没有完整审核记录"
        return {"action": "review" if reason else "reuse",
                "reason": reason or "正式实体及审核依赖一致，复用已完成测量",
                "policy": POLICY, "existing_results": outputs}


def reviewed_receipt(root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    """仅为四包完整只读审核生成复用依据，不将整理代码标作历史测量来源。"""
    if (audit.get("status") != "passed" or set(audit["experiments"]) != set(EXPERIMENTS)
            or audit.get("generation_forward_calls") != 0):
        raise RuntimeError("复用回执需要四个实验的完整只读审核")
    catalog = ResultCatalog(root)
    if audit.get("formal_result_files") != len(catalog.files):
        raise RuntimeError("审核范围与当前正式结果索引不一致")
    return {"status": "passed", "scope": "saved_measurements_and_analysis_dependencies",
            "results_digest": digest(catalog.files), "dependencies": review_bindings(root),
            "shared_inputs_digest": digest(catalog.index.get("shared_capacity_inputs", {})),
            "computation_signature": computation_signature(root), "audit": audit}
