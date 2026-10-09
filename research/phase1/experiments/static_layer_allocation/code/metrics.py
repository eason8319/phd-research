"""Key 降位宽的组级误差、固定并列排名与稳定词表 KL。"""

from __future__ import annotations

from typing import Any

import numpy as np


def grouped_rse(
    reference: np.ndarray, left: np.ndarray, right: np.ndarray, kv_heads: int,
) -> tuple[float | None, int]:
    """先按 KV head 对应 query 组归一化，再平均；分母始终为共同参考。"""
    reference, left, right = (np.asarray(x, dtype=np.float64) for x in (reference, left, right))
    if reference.shape != left.shape or left.shape != right.shape:
        raise ValueError("输出形状不一致")
    if kv_heads <= 0 or reference.shape[0] % kv_heads:
        raise ValueError("query head 与 KV head 组不匹配")
    if not all(np.isfinite(x).all() for x in (reference, left, right)):
        raise ValueError("输出含非有限值")
    repeats = reference.shape[0] // kv_heads
    denominator = np.sum(reference.reshape(kv_heads, repeats, -1) ** 2, axis=(1, 2))
    numerator = np.sum((left - right).reshape(kv_heads, repeats, -1) ** 2, axis=(1, 2))
    valid = denominator > 0
    return (
        float(np.mean(numerator[valid] / denominator[valid])) if valid.any() else None,
        int(np.sum(~valid)),
    )


def ranking(rows: list[dict[str, Any]], metric: str, *, descending: bool) -> list[int]:
    """有效测量按指标排序，并列固定取层号升序；无效记录不能生成正式排名。"""
    if any(row[metric] is None or row["invalid_groups"] for row in rows):
        raise RuntimeError("层测量含未定义误差或无效量化组，不能冻结分配")
    sign = -1 if descending else 1
    return sorted(range(len(rows)), key=lambda i: (sign * rows[i][metric], i))


def rank_agreement(first: list[int], second: list[int], top: int) -> dict[str, Any]:
    """统计固定并列规则下的 Spearman 排名相关及高敏感层集合重合。"""
    if sorted(first) != list(range(len(first))) or sorted(second) != sorted(first):
        raise ValueError("排名不是完整层排列")
    n = len(first)
    positions = {layer: i for i, layer in enumerate(second)}
    squares = sum((i - positions[layer]) ** 2 for i, layer in enumerate(first))
    overlap = len(set(first[:top]) & set(second[:top]))
    return {
        "spearman_fixed_ties": 1 - 6 * squares / (n * (n * n - 1)),
        "high_set_size": top, "high_set_overlap": overlap,
        "overlap_fraction": overlap / top,
    }


def vocabulary_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    """FP64 计算 KL(BF16 || 候选)，用归一化恒等式减少近零相消。"""
    reference, candidate = (np.asarray(x, dtype=np.float64).reshape(-1)
                            for x in (reference, candidate))
    if reference.shape != candidate.shape or reference.size == 0:
        raise ValueError("词表 logits 形状不一致或为空")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("词表 logits 含非有限值")

    def log_probs(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        return shifted - np.log(np.sum(np.exp(shifted)))

    old, new = log_probs(reference), log_probs(candidate)
    delta = new - old
    p, q = np.exp(old), np.exp(new)
    terms = np.empty_like(delta)
    small = np.abs(delta) < 1e-3
    d = delta[small]
    terms[small] = p[small] * d * d * (0.5 + d / 6 + d * d / 24 + d ** 3 / 120)
    negative = (~small) & (delta <= 0)
    terms[negative] = p[negative] * (np.expm1(delta[negative]) - delta[negative])
    positive = (~small) & (delta > 0)
    d = delta[positive]
    terms[positive] = q[positive] * (-np.expm1(-d) - d * np.exp(-d))
    value = float(np.sum(terms))
    if not np.isfinite(value) or value < 0:
        raise RuntimeError("稳定 KL 数值核验失败")
    return value


def paired_summary(differences: list[float]) -> dict[str, Any]:
    """登记逐题差值方向及绝对贡献最大的三题占比，不删除少数大值。"""
    values = np.asarray(differences, dtype=np.float64)
    order = np.argsort(-np.abs(values), kind="stable")
    total = float(np.sum(np.abs(values)))
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "negative_questions": int(np.sum(values < 0)),
        "positive_questions": int(np.sum(values > 0)),
        "equal_questions": int(np.sum(values == 0)),
        "largest_absolute_indices": order[:3].tolist(),
        "top_three_absolute_share": float(np.sum(np.abs(values[order[:3]]))) / total
        if total else None,
    }


def distribution_summary(
    rows: list[dict[str, Any]], formats: dict[str, Any], seeds: list[int],
) -> dict[str, Any]:
    """分别比较所有格式与预算，不以 KL 选择保质预算或 V 路线。"""
    means: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for kind in ("math", "retrieval"):
        subset = [row for row in rows if row["kind"] == kind]
        if not subset:
            raise ValueError("校准子集为空")
        names = list(subset[0]["kl"])
        means[kind] = {
            name: float(np.mean([row["kl"][name] for row in subset]))
            if all(row["kl"][name] is not None for row in subset) else None
            for name in names
        }
        paired[kind] = {}
        for name in names:
            if name in ("bf16", "kv8"):
                continue
            format_name = next(key for key in formats if name.startswith(key + "_"))
            baseline = format_name + "_k8v4"
            if means[kind][name] is not None and means[kind][baseline] is not None:
                differences = [row["kl"][name] - row["kl"][baseline] for row in subset]
                paired[kind][name + "_minus_k8v4"] = {
                    **paired_summary(differences), "ids": [row["id"] for row in subset],
                    "differences": differences,
                }
    for format_name, spec in formats.items():
        comparisons[format_name] = {}
        for budget, names in spec["budget_groups"].items():
            prefix = f"{format_name}_{budget}_"
            sensitivity = prefix + "sensitivity"
            references = [prefix + f"random_{seed}" for seed in seeds]
            references += [prefix + "reconstruction", prefix + "inverse"]
            if set(names) != {sensitivity, *references}:
                raise ValueError("预算组不是冻结的六策略")
            value = means["retrieval"][sensitivity]
            comparisons[format_name][budget] = {
                "sensitivity_strictly_below_all_random": value is not None and all(
                    means["retrieval"][name] is not None and value < means["retrieval"][name]
                    for name in references[:3]
                ),
                "sensitivity_below_reconstruction": value is not None
                and means["retrieval"][references[3]] is not None
                and value < means["retrieval"][references[3]],
                "sensitivity_below_inverse": value is not None
                and means["retrieval"][references[4]] is not None
                and value < means["retrieval"][references[4]],
                "status": "complete" if all(means["retrieval"][name] is not None
                                             for name in names) else "invalid_measurement",
            }
            for kind in means:
                subset = [row for row in rows if row["kind"] == kind]
                for name in references:
                    if means[kind][sensitivity] is None or means[kind][name] is None:
                        continue
                    differences = [row["kl"][sensitivity] - row["kl"][name] for row in subset]
                    paired[kind][sensitivity + "_minus_" + name] = {
                        **paired_summary(differences), "ids": [row["id"] for row in subset],
                        "differences": differences,
                    }
    trends: dict[str, Any] = {}
    for format_name, spec in formats.items():
        trends[format_name] = {}
        for strategy in ("sensitivity", "reconstruction", "inverse",
                         *(f"random_{seed}" for seed in seeds)):
            points = []
            previous: dict[str, float | None] | None = None
            for budget in spec["budget_groups"]:
                name = f"{format_name}_{budget}_{strategy}"
                current = {kind: means[kind][name] for kind in means}
                points.append({
                    "r": int(budget.removeprefix("k4_layers_")), "mean_kl": current,
                    "change_from_previous_budget": {
                        kind: current[kind] - previous[kind]
                        if previous is not None and current[kind] is not None
                        and previous[kind] is not None else None for kind in means
                    },
                })
                previous = current
            trends[format_name][strategy] = points
    return {"mean_next_token_kl": means, "budget_comparisons": comparisons,
            "paired_differences": paired, "budget_trends": trends}
