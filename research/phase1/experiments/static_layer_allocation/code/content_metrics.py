"""题内内容 token 平均、正确 token 差距及独立验证集的方案级比较。"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.static_layer_allocation.code.metrics import paired_summary, vocabulary_kl


def token_metrics(logits: np.ndarray, target: int) -> dict[str, float]:
    """目标 NLL 与目标相对最强错误候选的差距，全部在 FP64 中计算。"""
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.isfinite(z).all() or not 0 <= target < z.size:
        raise RuntimeError("目标或词表 logits 无效")
    maximum = float(z.max())
    normalizer = maximum + float(np.log(np.exp(z - maximum).sum()))
    wrong = max(float(z[:target].max()) if target else -np.inf,
                float(z[target + 1:].max()) if target + 1 < z.size else -np.inf)
    return {"nll": normalizer - float(z[target]), "margin": float(z[target]) - wrong}


def question_metrics(
    logits: np.ndarray, targets: list[int], reference: np.ndarray,
    baseline: np.ndarray | None = None,
) -> dict[str, Any]:
    """先平均该题实际内容 token；首内容位置单列，题目汇总等权。"""
    if len(logits) != len(targets) or logits.shape != reference.shape or not targets:
        raise ValueError("内容位置、目标与参照未对齐")
    frames = []
    for i, target in enumerate(targets):
        row = {**token_metrics(logits[i], target),
               "kl_bf16": vocabulary_kl(reference[i], logits[i])}
        if baseline is not None:
            row["kl_baseline"] = vocabulary_kl(baseline[i], logits[i])
            row["delta_nll"] = row["nll"] - token_metrics(baseline[i], target)["nll"]
        frames.append(row)
    return {
        "content_tokens": len(frames),
        **{key: float(np.mean([row[key] for row in frames])) for key in frames[0]},
        "first_content": frames[0], "frames": frames,
    }


def additive_metrics(
    baseline: np.ndarray, singles: dict[int, np.ndarray], actual: np.ndarray,
    layers: list[int], targets: list[int], reference: np.ndarray,
) -> dict[str, Any]:
    """单层 logit 扰动相加后再 softmax；将指标非线性与实际叠加偏离分别保存。"""
    base = baseline.astype(np.float64)
    additive = base + sum((singles[layer].astype(np.float64) - base for layer in layers),
                          np.zeros_like(base))
    measured = question_metrics(actual, targets, reference, baseline)
    predicted = question_metrics(additive, targets, reference, baseline)
    direct = []
    residuals = []
    for i in range(len(targets)):
        direct.append(vocabulary_kl(actual[i], additive[i]))
        perturbation = actual[i].astype(np.float64) - base[i]
        residual = actual[i].astype(np.float64) - additive[i]
        perturbation -= perturbation.mean()
        residual -= residual.mean()
        energy = float(np.sum(perturbation ** 2))
        residuals.append(float(np.sum(residual ** 2)) / energy if energy else None)
    single_sum = sum(float(np.mean([
        vocabulary_kl(base[i], singles[layer][i]) for i in range(len(targets))
    ])) for layer in layers)
    return {
        "actual": {key: measured[key] for key in ("nll", "kl_baseline")},
        "additive": {key: predicted[key] for key in ("nll", "kl_baseline")},
        "actual_to_additive_kl": float(np.mean(direct)),
        "actual_to_additive_kl_frames": direct,
        "centered_logit_residual_energy_fraction": float(np.mean(
            [value for value in residuals if value is not None]
        )) if any(value is not None for value in residuals) else None,
        "sum_single_layer_kl": single_sum,
        "kl_sum_minus_actual": single_sum - measured["kl_baseline"],
        "additive_nll_minus_actual": predicted["nll"] - measured["nll"],
    }


def validation_summary(
    rows: list[dict[str, Any]], groups: dict[str, Any], seeds: list[int],
) -> dict[str, Any]:
    """主判据用内容 NLL，逐题差值与完整答案 EM 同时保留，不选择质量阈值。"""
    names = list(rows[0]["schemes"])
    means = {}
    for name in names:
        selected = [row["schemes"][name] for row in rows]
        means[name] = {
            key: float(np.mean([value[key] for value in selected]))
            for key in ("nll", "margin", "kl_bf16", "free_exact")
        }
        means[name]["first_content_nll"] = float(np.mean(
            [value["first_content"]["nll"] for value in selected]
        ))
    comparisons = {}
    paired = {}
    for format_name, budgets in groups.items():
        comparisons[format_name] = {}
        base = format_name + "_k8v4"
        for budget, strategies in budgets.items():
            prefix = format_name + "_" + budget + "_"
            sensitive = prefix + "sensitivity"
            randoms = [prefix + f"random_{seed}" for seed in seeds]
            candidates = randoms + [prefix + "reconstruction", prefix + "inverse"]
            comparisons[format_name][budget] = {
                "sensitivity_strictly_below_all_random": all(
                    means[sensitive]["nll"] < means[name]["nll"] for name in randoms
                ),
                "sensitivity_below_reconstruction":
                    means[sensitive]["nll"] < means[prefix + "reconstruction"]["nll"],
                "sensitivity_below_inverse":
                    means[sensitive]["nll"] < means[prefix + "inverse"]["nll"],
            }
            for name in strategies:
                key = name + "_minus_" + base
                differences = [row["schemes"][name]["nll"] - row["schemes"][base]["nll"]
                               for row in rows]
                paired[key] = {**paired_summary(differences), "differences": differences,
                               "ids": [row["id"] for row in rows]}
            for name in candidates:
                differences = [row["schemes"][sensitive]["nll"] - row["schemes"][name]["nll"]
                               for row in rows]
                paired[sensitive + "_minus_" + name] = {
                    **paired_summary(differences), "differences": differences,
                    "ids": [row["id"] for row in rows],
                }
    return {"means": means, "budget_comparisons": comparisons, "paired_differences": paired}
