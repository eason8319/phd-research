"""位宽与表示交互的配对传播指标与固定基准位置的竞争诊断。"""

from __future__ import annotations

import torch


def _validate_numeric(tensors: list[torch.Tensor]) -> None:
    """检查有限浮点类型及设备一致性。"""
    for tensor in tensors:
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError("指标输入必须为 float32 或 float64")
        if tensor.device != tensors[0].device:
            raise ValueError("所有指标输入必须位于同一设备")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("指标输入包含非有限值；应先排除无效记录")


def _validate_inputs(
    score8: torch.Tensor,
    score4: torch.Tensor,
    value8: torch.Tensor,
    value4: torch.Tensor | None = None,
    reference: torch.Tensor | None = None,
) -> None:
    """检查已对齐 query head 的有限浮点输入，避免静默广播。"""
    if score8.ndim != 2 or score4.shape != score8.shape:
        raise ValueError("score8 与 score4 必须同为 [H, L]")
    heads, length = score8.shape
    if heads == 0 or length == 0:
        raise ValueError("query head 数量和历史长度必须大于零")
    if value8.ndim != 3 or value8.shape[:2] != score8.shape or value8.shape[2] == 0:
        raise ValueError("value8 必须为与分数对齐的 [H, L, D]，且 D 大于零")
    if value4 is not None and value4.shape != value8.shape:
        raise ValueError("value4 与 value8 的形状必须一致")
    if reference is not None and reference.shape != (heads, value8.shape[2]):
        raise ValueError("reference_output 必须为 [H, D]")
    tensors = [score8, score4, value8]
    tensors += [item for item in (value4, reference) if item is not None]
    _validate_numeric(tensors)


def _divide_or_nan(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """分母为零时返回 NaN，调用方同时记录零分母标记。"""
    positive = denominator > 0
    safe_denominator = torch.where(positive, denominator, torch.ones_like(denominator))
    return torch.where(
        positive, numerator / safe_denominator, torch.full_like(numerator, torch.nan)
    )


def _paired_vectors(
    probability8: torch.Tensor,
    probability4: torch.Tensor,
    centered_score: torch.Tensor,
    value8: torch.Tensor,
    value4: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """逐 head 用 float64 计算输出，避免复制完整长上下文 Value。"""
    baseline_outputs = []
    key_outputs = []
    value_deltas = []
    key_predictions = []
    for head in range(probability8.shape[0]):
        value8_h = value8[head].double()
        value4_h = value4[head].double()
        output88_h = probability8[head] @ value8_h
        baseline_outputs.append(output88_h)
        key_outputs.append(probability4[head] @ value8_h)
        value_deltas.append(probability8[head] @ (value4_h - value8_h))
        # 公共偏移被消去，仍等价于 sum p_i (v_i-o88) delta_s_i。
        key_predictions.append(
            (probability8[head] * centered_score[head]) @ (value8_h - output88_h)
        )
    return (
        torch.stack(baseline_outputs), torch.stack(key_outputs),
        torch.stack(value_deltas), torch.stack(key_predictions),
    )


def _error_metrics(
    reference: torch.Tensor,
    baseline_delta: torch.Tensor,
    key_delta: torch.Tensor,
    value_delta: torch.Tensor,
    key_prediction: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """对末轴输出向量做统一归一化分解；调用方决定每行代表 head 或整层。"""
    prediction_delta = key_prediction - key_delta
    reference_energy = reference.square().sum(dim=-1)
    key_energy = key_delta.square().sum(dim=-1)
    value_energy = value_delta.square().sum(dim=-1)
    prediction_energy = key_prediction.square().sum(dim=-1)
    prediction_residual_energy = prediction_delta.square().sum(dim=-1)
    key_cross = 2 * (baseline_delta * key_delta).sum(dim=-1)
    value_cross = 2 * (baseline_delta * value_delta).sum(dim=-1)
    kv8_error = _divide_or_nan(baseline_delta.square().sum(dim=-1), reference_energy)
    k4v8_error = _divide_or_nan(
        (baseline_delta + key_delta).square().sum(dim=-1), reference_energy
    )
    k8v4_error = _divide_or_nan(
        (baseline_delta + value_delta).square().sum(dim=-1), reference_energy
    )
    key_error = _divide_or_nan(key_energy, reference_energy)
    value_error = _divide_or_nan(value_energy, reference_energy)
    baseline_cross = _divide_or_nan(key_cross - value_cross, reference_energy)
    cosine = _divide_or_nan(
        (key_prediction * key_delta).sum(dim=-1),
        prediction_energy.sqrt() * key_energy.sqrt(),
    ).clamp(min=-1.0, max=1.0)
    return {
        "reference_energy": reference_energy,
        "zero_reference": reference_energy == 0,
        "kv8_error": kv8_error,
        "k4v8_error": k4v8_error,
        "k8v4_error": k8v4_error,
        "key_increment_error": key_error,
        "value_increment_error": value_error,
        "key_baseline_cross_term": _divide_or_nan(key_cross, reference_energy),
        "value_baseline_cross_term": _divide_or_nan(value_cross, reference_energy),
        "baseline_cross_term": baseline_cross,
        "difference_residual": k4v8_error - k8v4_error - key_error + value_error - baseline_cross,
        "key_prediction_error": _divide_or_nan(prediction_residual_energy, reference_energy),
        "key_prediction_relative_error": _divide_or_nan(prediction_residual_energy, key_energy),
        "key_prediction_cosine": cosine,
        "key_increment_zero": key_energy == 0,
        "key_prediction_zero": prediction_energy == 0,
    }


def output_metrics(
    reference: torch.Tensor,
    kv8: torch.Tensor,
    k4v8: torch.Tensor,
    k8v4: torch.Tensor,
    predicted_key: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """全头合并后计算整层输出误差，保持既有正式结果的归一化口径。

    五个输入均为 [H, D]；先对所有 head 的平方能量求和，再除以同一参考能量。
    返回标量 float64 张量和布尔标记，不能用逐 head 归一化误差的简单均值替代。
    predicted_key 为预测的 K 增量向量；零分母的处理和字段定义同 paired_head_metrics。
    """
    tensors = [reference, kv8, k4v8, k8v4, predicted_key]
    if reference.ndim != 2 or reference.numel() == 0:
        raise ValueError("输出输入必须为非空 [H, D]")
    if any(tensor.shape != reference.shape for tensor in tensors):
        raise ValueError("所有输出输入必须具有相同 [H, D] 形状")
    _validate_numeric(tensors)
    ref, output88, output48, output84, prediction = [
        tensor.double().reshape(1, -1) for tensor in tensors
    ]
    metrics = _error_metrics(
        ref, output88 - ref, output48 - output88, output84 - output88, prediction
    )
    return {name: tensor.squeeze(0) for name, tensor in metrics.items()}


def paired_head_metrics(
    score8: torch.Tensor,
    score4: torch.Tensor,
    value8_aligned: torch.Tensor,
    value4_aligned: torch.Tensor,
    reference_output: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """在固定 query 下计算 K8V8、K4V8 和 K8V4 的逐 head 配对指标。

    分数为 [H, L]，Value 已按 query head 对齐为 [H, L, D]，参考输出为 [H, D]。
    返回同设备的 [H] 张量，数值均用 float64；KL 使用自然对数，单位为 nat。
    *_error 是平方范数之比，输出误差统一除以高精度参考输出能量。
    key_prediction_relative_error 除以实际 K 增量能量。
    reference 为零时输出归一化指标为 NaN；余弦的零向量情况另设标记。
    本函数不量化、不修改输入，也不假设位置独立或残差服从高斯分布。
    """
    _validate_inputs(
        score8, score4, value8_aligned, value4_aligned, reference_output
    )
    log_probability8 = torch.log_softmax(score8.double(), dim=-1)
    log_probability4 = torch.log_softmax(score4.double(), dim=-1)
    probability8 = log_probability8.exp()
    probability4 = log_probability4.exp()
    delta_score = score4.double() - score8.double()
    weighted_mean = (probability8 * delta_score).sum(dim=-1, keepdim=True)
    centered_score = delta_score - weighted_mean
    weighted_var = (probability8 * centered_score.square()).sum(dim=-1)
    kl = (
        probability8 * (log_probability8 - log_probability4)
    ).sum(dim=-1).clamp_min(0.0)

    reference = reference_output.double()
    output88, output48, value_delta, key_prediction = _paired_vectors(
        probability8, probability4, centered_score, value8_aligned, value4_aligned
    )
    baseline_delta = output88 - reference
    key_delta = output48 - output88

    return {
        **_error_metrics(reference, baseline_delta, key_delta, value_delta, key_prediction),
        "weighted_score_var": weighted_var,
        "paired_probability_kl": kl,
    }
