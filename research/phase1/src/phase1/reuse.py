"""统一测量身份、计算实现兼容性和已有前缀的复用判定。"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


from kvstudy.artifacts import digest, file_hash

POLICY = "verified-prefix-append-missing"
GENERATION_RESULTS = "research/phase1/experiments/generation_error_sources/results"
REGISTRY = "research/phase1/experiments/static_layer_allocation/results/reference_reuse.json"
NUMERICAL_FUNCTIONS = (
    "_config", "_calibration", "_layer_params", "_rotate_key", "_quantize", "_recent",
    "_layer_bits", "_prefill_cache", "_write_residual_token", "_decode_step", "_generate",
)


def require_missing_result(path: Path) -> None:
    """独立生成入口只补缺测；已有正式实体先通过统一入口核验复用。"""
    if path.exists():
        raise RuntimeError("已有正式结果，先用 phase1.entry plan 核验复用，不能重复生成")


class _Computation(ast.NodeTransformer):
    """忽略源码位置、注释、文档字符串和导入位置，保留执行语句与常量。"""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        """消去新版解析器添加的空泛型字段，保持 Python 3.11 的计算身份。"""
        if "type_params" in node._fields and not getattr(node, "type_params", None):
            node._fields = tuple(name for name in node._fields if name != "type_params")
        return super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def function_signature(path: Path, name: str) -> str:
    """按实际函数计算兼容性；函数迁移和文档编辑不触发测量失效。"""
    tree = ast.parse(path.read_text("utf-8-sig"))
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    if node is None:
        raise ValueError(f"计算实现中缺少函数：{path.name}/{name}")
    node.name = "measurement_function"
    normalized = _Computation().visit(node)
    return digest(ast.dump(normalized, include_attributes=False))


def computation_signature(root: Path, quantized: bool = True) -> str:
    """核对新公共实现与历史快照中的实际算子，排除入口和加载搬运代码。"""
    shared = root / "research/phase1/src/phase1"
    legacy = root / "research/phase1/experiments/generation_error_sources/code"
    source = shared / "generation.py"
    if not source.exists():
        source = legacy / "quantized_generation.py"
    values = {}
    if quantized:
        values.update({name: function_signature(source, name) for name in NUMERICAL_FUNCTIONS})
        tree = ast.parse(source.read_text("utf-8-sig"))
        values["cache_constants"] = digest([
            ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in (
                    "G_C", "MAX_CONTEXT", "RESIDUAL_TOKENS",
                ) for target in node.targets
            )
        ])
    runner = shared / "runner.py"
    for name in (
        "modeling_module", "cache_lists", "project_qkv", "attention_from_batch",
        "prefill", "decode_step",
    ):
        values[f"runner/{name}"] = function_signature(runner, name)
    bf16 = runner if "def generate_bf16(" in runner.read_text("utf-8-sig") else (
        legacy / "long_baseline.py"
    )
    values["bf16_generation"] = function_signature(
        bf16, "generate_bf16" if bf16 == runner else "_generate",
    )
    models = shared / "models.py"
    values["eos"] = function_signature(
        models if models.exists() else legacy / "short_baseline.py", "_eos_ids",
    )
    for name in ("attention", "quantize", "groups"):
        tree = ast.parse((root / f"research/src/kvstudy/{name}.py").read_text("utf-8-sig"))
        values[name] = digest(ast.dump(_Computation().visit(tree), include_attributes=False))
    return digest(values)


def reference_path(
    root: Path, model: str, task: str, condition: str, references: dict[str, Any],
) -> Path:
    """共同首轮参考只在真实生成实验中保存一个正式实体。"""
    labels = {"bf16": "bf16", "kv8": references["kv8"],
              "k8v4": references["same_format_k8v4"][model]}
    if condition not in labels:
        raise ValueError("未知共同参考条件")
    return root / GENERATION_RESULTS / model / f"v_{task}_{labels[condition]}.json"


def execution_device(value: dict[str, Any]) -> dict[str, Any]:
    """统一历史设备字段；节点、作业和可见序号不作为数值身份。"""
    aliases = {"name": "gpu", "torch_version": "torch", "cuda_version": "cuda"}
    ignored = {"node", "slurm_job", "visible_device", "total_bytes", "total_memory"}
    return {aliases.get(key, key): item for key, item in value.items() if key not in ignored}


@dataclass(frozen=True)
class ReuseDecision:
    """缺失来源等待核验；只有已确认的测量差异才允许重测。"""

    action: str
    reason: str
    completed_n: int = 0
    missing_n: int = 0

    def record(self) -> dict[str, Any]:
        """导出运行前可审阅的复用决定。"""
        return asdict(self)


def assess_reuse(
    result: dict[str, Any] | None, expected_rows: list[dict[str, Any]], *,
    model: str, configuration: str, residual_bits: int = 0,
    device: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None,
    source_sha256: str | None = None, signature: str | None = None,
) -> ReuseDecision:
    """同身份复用已完成前缀；输入差异拒绝合并，缺失证据不自动重跑。"""
    if result is None:
        return ReuseDecision("measure", "无对应测量", missing_n=len(expected_rows))
    rows = result.get("rows", [])
    if result.get("n") != len(rows) or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("已有结果行数或样本身份异常")
    if result.get("model") != model:
        raise RuntimeError("已有结果的模型身份不同")
    actual = result.get("config", result.get("condition"))
    if isinstance(actual, dict):
        actual = actual.get("name")
    if actual == "bf16-h0":
        actual = "bf16"
    if actual != configuration or result.get("residual_bits", 0) != residual_bits:
        return ReuseDecision("remeasure", "已确认量化配置或残留不同")
    available = min(len(rows), len(expected_rows))
    for old, expected in zip(rows[:available], expected_rows):
        if any(old.get(key) != expected.get(key) for key in (
            "id", "input_sha256", "input_tokens",
        )):
            return ReuseDecision("remeasure", "已确认输入身份、全文或分词不同")
    old_device = execution_device(result.get("execution_device", result.get("device", {})))
    verified = bool(source_sha256 and signature and evidence and evidence.get("status") == "passed"
                    and evidence.get("source_sha256") == source_sha256
                    and evidence.get("computation_signature") == signature)
    if verified:
        old_device.update(execution_device(evidence.get("device", {})))
    current = execution_device(device or {})
    for key, value in current.items():
        if key in old_device and old_device[key] != value:
            return ReuseDecision("review", f"执行条件不同，先匹配参考条件：{key}")
    if not verified:
        return ReuseDecision("review", "缺少本次数值与来源核验依据，先查证已有测量")
    missing = max(0, len(expected_rows) - available)
    return ReuseDecision("append" if missing else "reuse",
                         "身份和计算实现已核验，只追加缺少前缀" if missing else
                         "身份和计算实现已核验，复用原实体", available, missing)


def validate_reference_prefix(
    old: dict[str, Any], measured: dict[str, Any],
) -> dict[str, Any]:
    """归并前核对同输入、得分与长度；历史未知停止原因继续保留未知。"""
    if len(measured["rows"]) < len(old["rows"]):
        raise RuntimeError("重测记录不覆盖原参考前缀")
    keys = ("id", "input_sha256", "input_tokens", "new_tokens", "invalid_groups",
            "correct", "em", "f1", "answer_f1", "answer_em", "extracted_answers",
            "missing_answers", "strict_correct")
    unknown_stops = 0
    for first, second in zip(old["rows"], measured["rows"]):
        if any(first.get(key) != second.get(key) for key in keys):
            raise RuntimeError(f"参考前缀存在测量差异，必须保留独立来源：{first['id']}")
        if first.get("stop") is None:
            unknown_stops += 1
        elif first["stop"] != second.get("stop"):
            raise RuntimeError("参考前缀的已记录停止原因不同")
    return {"status": "passed", "n": len(old["rows"]), "checked_fields": list(keys),
            "historical_unknown_stops": unknown_stops}
