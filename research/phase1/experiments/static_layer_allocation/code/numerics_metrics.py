"""答案规范化、完整路径领先量和配对数值不确定性诊断。"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from experiments.static_layer_allocation.code.content_metrics import token_metrics


def score_answer(prediction: str, answer: str) -> dict[str, int | str | None]:
    """只接受完整小写八位值或 val-形式；格式与内容分开，拒绝多答案及解释。"""
    text = prediction.strip()
    full = re.fullmatch(r"val-([0-9a-f]{8})", text)
    bare = re.fullmatch(r"[0-9a-f]{8}", text)
    value = full.group(1) if full else text if bare else None
    return {"normalized_value": value, "content_correct": int(value == answer[4:]),
            "format_compliant": int(full is not None),
            "formatted_correct": int(text == answer)}


def path_metrics(
    logits: np.ndarray, targets: list[int], tokenizer: Any, content_indices: list[int],
) -> dict[str, Any]:
    """覆盖前缀、分隔符、内容及冻结终止 token，保存首次非正与全局最小位置。"""
    if len(logits) != len(targets):
        raise ValueError("指定路径与预测数目不一致")
    frames = []
    for index, target in enumerate(targets):
        z = np.asarray(logits[index], dtype=np.float64)
        wrong = z.copy()
        wrong[target] = -np.inf
        competitor = int(wrong.argmax())
        row = token_metrics(z, target)
        frames.append({"answer_index": index, "target_id": target,
                       "target_text": tokenizer.decode([target]),
                       "competitor_id": competitor,
                       "competitor_text": tokenizer.decode([competitor]),
                       "greedy_id": int(z.argmax()),
                       "role": "termination" if index == len(targets) - 1 else
                       "content" if index in content_indices else "prefix_or_separator",
                       **row})
    minimum = min(frames, key=lambda row: (row["margin"], row["answer_index"]))
    first = next((row for row in frames if row["margin"] <= 0), None)
    return {"content_nll": float(np.mean([frames[i]["nll"] for i in content_indices])),
            "all_path_nll": float(np.mean([row["nll"] for row in frames])),
            "minimum": minimum, "first_nonpositive": first,
            "all_strictly_positive": first is None, "frames": frames}


def paired_reliability(
    effects: np.ndarray, seed: int, bootstrap: int,
) -> dict[str, Any]:
    """effects 为 [题, 精度/重复]；重采样题目，包络只代表已测计算条件。"""
    values = np.asarray(effects, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("配对差异必须为有限的题目×测量矩阵")
    lower, upper = values.min(axis=1), values.max(axis=1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(bootstrap, len(values)))
    interval = [float(np.quantile(lower[draws].mean(axis=1), 0.025)),
                float(np.quantile(upper[draws].mean(axis=1), 0.975))]
    central = values[:, -1]
    central_interval = np.quantile(central[draws].mean(axis=1), [0.025, 0.975]).tolist()
    status = "reliably_lower" if interval[1] < 0 else \
        "reliably_higher" if interval[0] > 0 else "unresolved"
    return {"n": len(values), "measured_variant_means": values.mean(axis=0).tolist(),
            "measured_numeric_interval": [float(lower.mean()), float(upper.mean())],
            "question_bootstrap_numeric_envelope_95": interval,
            "highest_precision_question_ci_95": central_interval,
            "status": status, "bootstrap_samples": bootstrap, "bootstrap_seed": seed}


def spearman(left: list[float], right: list[float]) -> float | None:
    """并列取平均秩；常量或不足两个对象时保留未定义。"""
    def ranks(values: list[float]) -> np.ndarray:
        array = np.asarray(values)
        ordered = np.argsort(array, kind="stable")
        result = np.empty(len(array), dtype=np.float64)
        start = 0
        while start < len(array):
            end = start + 1
            while end < len(array) and array[ordered[end]] == array[ordered[start]]:
                end += 1
            result[ordered[start:end]] = (start + end - 1) / 2
            start = end
        return result
    if len(left) != len(right) or len(left) < 2:
        return None
    a, b = ranks(left), ranks(right)
    if not a.std() or not b.std():
        return None
    return float(np.corrcoef(a, b)[0, 1])


def centered_energy(values: np.ndarray) -> float:
    """先移除每个位置的词表共同平移，再平均位置和词表的平方，单位 logit²。"""
    z = np.asarray(values, dtype=np.float64)
    centered = z - z.mean(axis=-1, keepdims=True)
    return float(np.mean(centered * centered))


def incremental_comparison(
    base: np.ndarray, actual: np.ndarray, reference_base: np.ndarray,
    reference_actual: np.ndarray,
) -> dict[str, float | None]:
    """比较量化增量，不把两种精度的共同基线位移归入某层量化效应。"""
    change = actual.astype(np.float64) - base.astype(np.float64)
    ref = reference_actual.astype(np.float64) - reference_base.astype(np.float64)
    a = change - change.mean(axis=-1, keepdims=True)
    b = ref - ref.mean(axis=-1, keepdims=True)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"increment_centered_rms": centered_energy(change) ** 0.5,
            "reference_increment_centered_rms": centered_energy(ref) ** 0.5,
            "increment_difference_centered_rms": centered_energy(change - ref) ** 0.5,
            "increment_direction_cosine": float(np.sum(a * b)) / norm if norm else None}
