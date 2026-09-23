"""配对下降区间。二值任务用 Bonferroni 后的 Clopper-Pearson，连续分数用重采样。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

# 开发筛选容差，单位是分。
TOLERANCE_POINTS = 2.0
BOOTSTRAP_DRAWS = 10_000


@dataclass(frozen=True)
class DropInterval:
    """下降 = 参考 − 候选，单位是分。全零连续差异不给零宽区间。"""

    estimate: float
    lower: float | None
    upper: float | None
    kind: str
    n: int
    detail: dict[str, int | float]


def clopper_pearson(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """双侧精确二项区间。alpha 是双侧失误率。"""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("二项计数无效")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须在 (0, 1)")
    if trials == 0:
        return 0.0, 1.0
    if successes == 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    if successes == trials:
        upper = 1.0
    else:
        upper = float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def binary_drop_interval(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    alpha: float = 0.05,
) -> DropInterval:
    """配对二值下降的保守 95% 区间。两类分歧各自用 alpha/2 的精确区间。"""
    reference = np.asarray(reference_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("配对结果必须是等长的一维数组")
    n = int(reference.size)
    degrade = int(np.count_nonzero(reference & ~candidate))
    improve = int(np.count_nonzero(~reference & candidate))
    # 两个区间一起覆盖的概率至少为 1-alpha。
    each = clopper_pearson(degrade, n, alpha / 2)
    other = clopper_pearson(improve, n, alpha / 2)
    estimate = 100.0 * (degrade - improve) / n if n else 0.0
    lower = 100.0 * (each[0] - other[1])
    upper = 100.0 * (each[1] - other[0])
    return DropInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        kind="binary",
        n=n,
        detail={"degrade": degrade, "improve": improve},
    )


def bootstrap_drop_interval(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> DropInterval:
    """样本级配对重采样。全零差异只返回点估计，不报告零宽区间。"""
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("配对分数必须是等长的一维数组")
    delta = reference - candidate
    n = int(delta.size)
    estimate = float(delta.mean()) if n else 0.0
    if n == 0 or np.all(delta == 0):
        return DropInterval(
            estimate=estimate,
            lower=None,
            upper=None,
            kind="all_zero",
            n=n,
            detail={"draws": draws, "seed": seed},
        )
    generator = np.random.default_rng(seed)
    index = generator.integers(0, n, size=(draws, n))
    means = delta[index].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return DropInterval(
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        kind="bootstrap",
        n=n,
        detail={"draws": draws, "seed": seed},
    )


def crosses_tolerance(interval: DropInterval, tolerance: float = TOLERANCE_POINTS) -> bool:
    """区间是否跨过容差。缺失端点不算跨越。"""
    if interval.lower is None or interval.upper is None:
        return False
    return interval.lower <= tolerance <= interval.upper and interval.upper > tolerance


def eliminated(
    versus_reference: DropInterval,
    versus_kv8: DropInterval,
    *,
    format_ok: bool,
    tolerance: float = TOLERANCE_POINTS,
) -> bool:
    """格式不可行，或任一下降区间的下界高于容差。"""
    if not format_ok:
        return True
    for interval in (versus_reference, versus_kv8):
        if interval.lower is not None and interval.lower > tolerance:
            return True
    return False


def quality_state(
    versus_reference: DropInterval,
    versus_kv8: DropInterval,
    *,
    format_ok: bool,
    systematic_failure: bool = False,
    tolerance: float = TOLERANCE_POINTS,
) -> str:
    """先看淘汰，再看保留，其余按区间是否跨越容差分成证据不足或开发晋级。"""
    if eliminated(versus_reference, versus_kv8, format_ok=format_ok, tolerance=tolerance):
        return "淘汰"
    upper_ok = (
        versus_reference.upper is not None
        and versus_kv8.upper is not None
        and versus_reference.upper <= tolerance
        and versus_kv8.upper <= tolerance
        and not systematic_failure
    )
    if upper_ok:
        return "保留"
    if crosses_tolerance(versus_reference, tolerance) or crosses_tolerance(versus_kv8, tolerance):
        return "证据不足"
    return "开发晋级"


def class_unit_passes(
    versus_reference: DropInterval,
    versus_kv8: DropInterval,
    *,
    format_ok: bool,
    tolerance: float = TOLERANCE_POINTS,
) -> bool:
    """单元未淘汰，且相对高精度和 KV8 的下降点估计都不超过容差。"""
    if eliminated(versus_reference, versus_kv8, format_ok=format_ok, tolerance=tolerance):
        return False
    return versus_reference.estimate <= tolerance and versus_kv8.estimate <= tolerance
