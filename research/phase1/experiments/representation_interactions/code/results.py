"""按题汇总位宽与表示交互配对诊断，并保存兼容既有候选选择的统一结果。"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any


SCHEMA_VERSION = "paired-propagation-diagnostics"
PAIRS = tuple(
    f"{mode}-{rope}"
    for mode in ("per_channel", "per_token")
    for rope in ("pre", "post")
)
MATRIX_CONFIGS = tuple(
    f"{width}-{pair}"
    for pair in PAIRS
    for width in ("kv8", "k8v4", "k4v8", "kv4")
) + ("kv4-per_channel-pre-vchannel",)
BOOTSTRAP_SEED = 8319
BOOTSTRAP_DRAWS = 2000
REFERENCE_NORMALIZED_OUTPUTS = {
    "kv8_error", "k4v8_error", "k8v4_error", "key_increment_error",
    "value_increment_error", "key_baseline_cross_term", "value_baseline_cross_term",
    "baseline_cross_term", "key_prediction_error",
}


def _validate_json(value: Any, location: str) -> None:
    """拒绝非有限数和不能写成严格 JSON 的值，保留显式 null。"""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} 包含非有限数")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} 的 JSON 键必须为字符串")
            _validate_json(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} 包含不支持的 JSON 类型：{type(value).__name__}")


def _scalar_metrics(value: Any, location: str) -> dict[str, float | int | bool | None]:
    """指标字典仅允许有限数、布尔标记和未定义值 null。"""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{location} 必须为非空指标字典")
    _validate_json(value, location)
    for key, item in value.items():
        if not isinstance(item, (float, int, bool)) and item is not None:
            raise ValueError(f"{location}.{key} 必须为数值、布尔值或 null")
    return value


def _mean(values: Iterable[float | int | bool | None]) -> float | None:
    """仅平均已定义指标；没有有效值时返回 null。"""
    finite = [float(value) for value in values if value is not None]
    return math.fsum(finite) / len(finite) if finite else None


def _mean_dict(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """在字段一致的标量记录上计算均值。"""
    names = tuple(records[0])
    if any(set(row) != set(names) for row in records):
        raise ValueError("同一指标组的字段不一致")
    return {name: _mean(row[name] for row in records) for name in names}


def _quantile(values: list[float], probability: float) -> float | None:
    """用线性插值确定 bootstrap 置信区间的端点。"""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _fold_mean(samples: list[dict[str, Any]], group: str, name: str) -> float | None:
    """先在各折平均题目指标，再对两折等权；整折无定义值时返回 null。"""
    means = [
        _mean(row[group][name] for row in samples if row["fold"] == fold)
        for fold in (0, 1)
    ]
    if any(value is None for value in means):
        return None
    return _mean(means)


def _group_summary(
    samples: list[dict[str, Any]], group: str
) -> tuple[dict[str, float | None], dict[str, int]]:
    """汇总题目均值的两折均值和有效题数，head 不作独立样本。"""
    names = tuple(samples[0][group])
    if any(set(row[group]) != set(names) for row in samples):
        raise ValueError(f"{group} 的题目级指标字段不一致")
    means = {name: _fold_mean(samples, group, name) for name in names}
    counts = {}
    for name in names:
        values = [row[group][name] for row in samples if row[group][name] is not None]
        counts[name] = len(values)
    return means, counts


def _difference_ci(samples: list[dict[str, Any]], seed: int, draws: int) -> dict[str, Any]:
    """以题目为重采样单位，在各折内 bootstrap K 与 V 增量误差的差。"""
    folds: dict[int, list[float]] = {0: [], 1: []}
    for row in samples:
        output = row["layer_output_means"]
        key = output.get("key_increment_error")
        value = output.get("value_increment_error")
        if key is not None and value is not None:
            folds[row["fold"]].append(key - value)
    payload = {
        "estimate": None, "lower": None, "upper": None,
        "samples": sum(len(values) for values in folds.values()),
        "seed": seed, "draws": draws,
        "confidence": 0.95, "unit": "sample", "method": "fold-stratified-percentile",
    }
    if not folds[0] or not folds[1]:
        return payload
    payload["estimate"] = _mean([_mean(folds[0]), _mean(folds[1])])
    generator = random.Random(seed)
    estimates = []
    for _ in range(draws):
        fold_means = [
            _mean(generator.choices(values, k=len(values))) for values in folds.values()
        ]
        estimates.append(_mean(fold_means))
    payload["lower"] = _quantile(estimates, 0.025)
    payload["upper"] = _quantile(estimates, 0.975)
    return payload


def _sample_summary(row: dict[str, Any]) -> tuple[dict[str, Any], tuple[int, int], int]:
    """把一题一表示的逐层明细压缩为题目指标。"""
    layers = row.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("每条配对明细必须包含非空 layers")
    indices = [layer.get("layer") for layer in layers if isinstance(layer, dict)]
    if (len(indices) != len(layers)
            or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or sorted(indices) != list(range(len(layers)))):
        raise ValueError("层编号必须唯一且完整覆盖 0..N-1")
    heads = []
    outputs = []
    head_count = None
    zero_outputs = 0
    for layer in layers:
        layer_heads = layer.get("heads")
        if not isinstance(layer_heads, list) or not layer_heads:
            raise ValueError("每层必须包含非空 heads")
        if head_count is None:
            head_count = len(layer_heads)
        if len(layer_heads) != head_count:
            raise ValueError("同题各层的 query head 数不一致")
        heads.extend(_scalar_metrics(item, "heads") for item in layer_heads)
        output = dict(_scalar_metrics(layer.get("output"), "output"))
        if ("zero_reference" in output and "zero_reference_output" in output
                and output["zero_reference"] != output["zero_reference_output"]):
            raise ValueError("参考输出为零的两个标记不一致")
        zero = output.pop("zero_reference", output.pop("zero_reference_output", False))
        if not isinstance(zero, bool):
            raise ValueError("zero_reference_output 必须为布尔标记")
        zero_outputs += int(zero)
        if not output:
            raise ValueError("output 必须含有输出误差指标")
        if zero and any(
            output.get(name) is not None for name in REFERENCE_NORMALIZED_OUTPUTS
        ):
            raise ValueError("参考输出为零的层，除以参考能量的输出指标必须为 null")
        residual = output.get("difference_residual")
        if residual is not None and abs(residual) > 1e-7:
            raise ValueError(f"逐层输出误差分解不闭合：{residual}")
        outputs.append(output)
    sample = {
        key: row[key] for key in ("id", "kind", "fold", "source_sha256", "pair")
    }
    sample["head_metric_means"] = _mean_dict(heads)
    sample["layer_output_means"] = _mean_dict(outputs)
    required = {"key_increment_error", "value_increment_error"}
    if not required <= set(sample["layer_output_means"]):
        raise ValueError("output 缺少 K/V 增量误差")
    return sample, (len(layers), head_count), zero_outputs


def summarize_rows(
    rows: Iterable[dict[str, Any]],
    expected_ids: Collection[str] | Mapping[str, dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """校验四种表示的完整配对，并仅以冻结题目为统计样本汇总。

    expected_ids 可为 ID 集合，或含 kind、fold、source_sha256 的冻结元数据映射。
    每行有 id、kind、fold、source_sha256、pair 和逐层 layers；每层有 layer、
    heads 标量列表和 output 标量字典。
    总体均值及差值区间按两折等权，区间在折内重采样题目。
    返回四种表示的汇总及紧凑 samples 列表。
    """
    expected_list = list(expected_ids)
    expected = set(expected_list)
    if (not expected or len(expected) != len(expected_list)
            or any(not isinstance(item, str) or not item for item in expected)):
        raise ValueError("冻结样本 ID 必须为非空、唯一的字符串")
    metadata = expected_ids if isinstance(expected_ids, Mapping) else {}
    seen = set()
    observed_metadata = {}
    samples = []
    dimensions = None
    counts = {pair: {"layers": 0, "query_heads": 0, "zero_reference_outputs": 0}
              for pair in PAIRS}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("配对明细行必须为字典")
        _validate_json(row, "row")
        sample_id, pair = row.get("id"), row.get("pair")
        if not isinstance(sample_id, str) or sample_id not in expected or pair not in PAIRS:
            raise ValueError(f"未登记的样本或表示：{sample_id!r}，{pair!r}")
        identity = (sample_id, pair)
        if identity in seen:
            raise ValueError(f"存在重复配对记录：{identity}")
        seen.add(identity)
        if type(row.get("fold")) is not int or row["fold"] not in (0, 1):
            raise ValueError("样本 fold 必须为 0 或 1")
        if any(not isinstance(row.get(key), str) or not row[key]
               for key in ("kind", "source_sha256")):
            raise ValueError("样本 kind 和 source_sha256 必须为非空字符串")
        current = {key: row[key] for key in ("kind", "fold", "source_sha256")}
        if sample_id in observed_metadata and observed_metadata[sample_id] != current:
            raise ValueError(f"同题不同表示的输入或折不一致：{sample_id}")
        observed_metadata[sample_id] = current
        if sample_id in metadata:
            for key, value in metadata[sample_id].items():
                if key in current and current[key] != value:
                    raise ValueError(f"配对记录与冻结元数据不一致：{sample_id}，{key}")
        sample, shape, zero_outputs = _sample_summary(row)
        if dimensions is None:
            dimensions = shape
        if dimensions != shape:
            raise ValueError("各题或各表示的层数、query head 数不一致")
        samples.append(sample)
        counts[pair]["layers"] += shape[0]
        counts[pair]["query_heads"] += shape[0] * shape[1]
        counts[pair]["zero_reference_outputs"] += zero_outputs
    required = {(sample_id, pair) for sample_id in expected for pair in PAIRS}
    if seen != required:
        raise ValueError(f"配对记录不完整：实际 {len(seen)}，预期 {len(required)}")
    if {item["fold"] for item in observed_metadata.values()} != {0, 1}:
        raise ValueError("配对诊断必须包含两个冻结折")
    samples.sort(key=lambda row: (PAIRS.index(row["pair"]), row["fold"], row["id"]))
    metrics = summarize_samples(
        samples, expected_ids, bootstrap_seed=bootstrap_seed, resamples=resamples
    )
    for pair in PAIRS:
        metrics[pair].update(counts[pair])
    return metrics, samples


def _layer_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按层计算两折等权均值及逐题极值；层不作为独立统计样本。"""
    fields = (
        "key_increment_error", "key_prediction_error",
        "key_prediction_relative_error", "key_prediction_cosine",
    )
    pairs = {}
    for pair in PAIRS:
        subset = [row for row in rows if row["pair"] == pair]
        layer_count = len(subset[0]["layers"])
        means, counts = [], []
        maximum_r2 = None
        normalization_errors, decomposition_errors = [], []
        above_one = 0
        for index in range(layer_count):
            values = [row["layers"][index] for row in subset]
            means.append([index])
            counts.append([index])
            for field in fields:
                folds = [
                    _mean(value[field] for row, value in zip(subset, values)
                          if row["fold"] == fold)
                    for fold in (0, 1)
                ]
                means[-1].append(None if None in folds else _mean(folds))
                counts[-1].append(sum(value[field] is not None for value in values))
            for row, value in zip(subset, values):
                key, prediction, r2, cosine = (value[field] for field in fields)
                residual = value["difference_residual"]
                if residual is not None:
                    decomposition_errors.append(abs(residual))
                if key is not None and prediction is not None and r2 is not None:
                    error = abs(prediction - r2 * key)
                    if error > 1e-7 * max(1.0, abs(prediction), abs(r2 * key)):
                        raise ValueError("逐层预测相对残差与 K 增量误差不一致")
                    normalization_errors.append(error)
                if r2 is not None:
                    above_one += int(r2 > 1.0)
                    if maximum_r2 is None or r2 > maximum_r2["r2"]:
                        maximum_r2 = {
                            "id": row["id"], "fold": row["fold"], "layer": index,
                            "DK": key, "prediction_error": prediction,
                            "r2": r2, "cosine": cosine,
                        }
        pairs[pair] = {
            "max_prediction_normalization_error": max(normalization_errors, default=None),
            "max_output_decomposition_error": max(decomposition_errors, default=None),
            "r2_gt_one_layers": above_one, "max_r2_location": maximum_r2,
            "layer_mean_columns": ["layer", "DK", "prediction_error", "r2", "cosine"],
            "layer_means": means, "layer_valid_samples": counts,
        }
    return {"audit_passed": True, "rows": len(rows), "pairs": pairs}


def summarize_diagnostics(
    rows: Iterable[dict[str, Any]],
    expected_ids: Collection[str] | Mapping[str, dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_DRAWS,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """一次读取明细，返回题目汇总、评分和报告使用的逐层审核统计。

    仅缓存逐层输出标量，不缓存 head 明细；完整配对和冻结来源沿用 summarize_rows 校验。
    逐层均值先在折内平均题目，再将两折等权平均；未定义值保留 null 并记录有效题数。
    """
    compact = []
    fields = (
        "key_increment_error", "key_prediction_error", "key_prediction_relative_error",
        "key_prediction_cosine", "difference_residual",
    )

    def collect() -> Iterable[dict[str, Any]]:
        """在题目校验通过后保留生成逐层统计所需的标量。"""
        for row in rows:
            yield row
            layers = sorted(row["layers"], key=lambda layer: layer["layer"])
            if any(not set(fields) <= set(layer["output"]) for layer in layers):
                raise ValueError("逐层统计缺少一阶预测或误差分解指标")
            compact.append({
                "id": row["id"], "fold": row["fold"], "pair": row["pair"],
                "layers": [{field: layer["output"][field] for field in fields}
                           for layer in layers],
            })

    metrics, samples = summarize_rows(
        collect(), expected_ids, bootstrap_seed=bootstrap_seed, resamples=resamples
    )
    return metrics, samples, _layer_checks(compact)


def summarize_samples(
    samples: list[dict[str, Any]],
    expected_ids: Collection[str] | Mapping[str, dict[str, Any]] | None = None,
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """从已有题目级评分重新汇总，不虚构未保存的层/head 指标。

    两折等权均值和折内 bootstrap 均由现有评分计算；不从题目均值
    推断 head 分布。返回值不含明细层数或 query head 数，由真实来源元数据补入。
    expected_ids 提供时检查冻结题目覆盖；未提供时仍检查已出现题目的四种表示。
    """
    if (type(bootstrap_seed) is not int or type(resamples) is not int or resamples < 1):
        raise ValueError("bootstrap_seed 必须为整数，resamples 必须为正整数")
    if not isinstance(samples, list) or not samples:
        raise ValueError("题目评分必须为非空列表")
    _validate_json(samples, "samples")
    expected_list = list(expected_ids) if expected_ids is not None else None
    expected = set(expected_list) if expected_list is not None else set()
    if expected_list is not None and (
        not expected or len(expected) != len(expected_list)
        or any(not isinstance(item, str) or not item for item in expected)
    ):
        raise ValueError("冻结样本 ID 必须为非空、唯一的字符串")
    frozen = expected_ids if isinstance(expected_ids, Mapping) else {}
    seen = set()
    observed = {}
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("题目评分必须为字典")
        sample_id, pair = row.get("id"), row.get("pair")
        if (not isinstance(sample_id, str) or not sample_id or pair not in PAIRS
                or expected_list is not None and sample_id not in expected):
            raise ValueError("题目评分包含未登记样本或表示")
        identity = (sample_id, pair)
        if identity in seen:
            raise ValueError(f"存在重复题目评分：{identity}")
        seen.add(identity)
        if type(row.get("fold")) is not int or row["fold"] not in (0, 1):
            raise ValueError("题目评分 fold 必须为 0 或 1")
        if any(not isinstance(row.get(key), str) or not row[key]
               for key in ("kind", "source_sha256")):
            raise ValueError("题目评分缺少输入来源")
        metadata = {key: row[key] for key in ("kind", "fold", "source_sha256")}
        if sample_id in observed and observed[sample_id] != metadata:
            raise ValueError(f"同题评分的输入或折不一致：{sample_id}")
        observed[sample_id] = metadata
        if sample_id in frozen:
            for key, value in frozen[sample_id].items():
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"题目评分与冻结元数据不一致：{sample_id}，{key}")
        for group in ("head_metric_means", "layer_output_means"):
            _scalar_metrics(row.get(group), group)
        if not {"key_increment_error", "value_increment_error"} <= set(
            row["layer_output_means"]
        ):
            raise ValueError("题目评分缺少 K/V 增量误差")
        residual = row["layer_output_means"].get("difference_residual")
        if residual is not None and abs(residual) > 1e-7:
            raise ValueError(f"逐题输出误差分解不闭合：{sample_id}，{residual}")
    ids = expected if expected_list is not None else set(observed)
    if seen != {(sample_id, pair) for sample_id in ids for pair in PAIRS}:
        raise ValueError("题目评分缺少冻结样本或四种配对表示")
    if {metadata["fold"] for metadata in observed.values()} != {0, 1}:
        raise ValueError("题目评分必须包含两个冻结折")
    metrics = {}
    for pair in PAIRS:
        subset = sorted(
            (row for row in samples if row["pair"] == pair),
            key=lambda row: (row["fold"], row["id"]),
        )
        entry = {"samples": len(subset)}
        groups = [
            ("head_metric_means", "head_metric"),
            ("layer_output_means", "layer_output"),
        ]
        for group, prefix in groups:
            means, valid_counts = _group_summary(subset, group)
            entry[f"{prefix}_means"] = means
            entry[f"{prefix}_valid_samples"] = valid_counts
        differences = [
            row["layer_output_means"]["key_increment_error"]
            - row["layer_output_means"]["value_increment_error"]
            for row in subset
            if row["layer_output_means"]["key_increment_error"] is not None
            and row["layer_output_means"]["value_increment_error"] is not None
        ]
        entry["key_increment_greater_samples"] = sum(value > 0 for value in differences)
        entry["key_increment_comparable_samples"] = len(differences)
        entry["key_increment_minus_value_ci95"] = _difference_ci(
            subset, bootstrap_seed, resamples
        )
        variance = entry["head_metric_means"].get("weighted_score_var")
        kl = entry["head_metric_means"].get("paired_probability_kl")
        entry["paired_kl_over_half_weighted_variance"] = (
            kl / (0.5 * variance)
            if kl is not None and variance is not None and variance > 0 else None
        )
        residual = entry["layer_output_means"].get("difference_residual")
        if residual is not None and abs(residual) > 1e-7:
            raise ValueError(f"输出误差分解不闭合：{pair}，{residual}")
        metrics[pair] = entry
    _validate_json(metrics, "metrics")
    return metrics


def write_results(
    destination: Path,
    matrix_payload: dict[str, Any],
    diagnostic_payload: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    measurement_status: dict[str, Any] | None = None,
) -> None:
    """分别原子替换 summary.json 和 samples.json，保留原主矩阵与冻结选择。

    diagnostic_payload 必须提供 model、measurement_version、provenance、metrics，
    以及显式 measurement_status；后者放在 summary 顶层，不能把待测指标写成已测。
    两份严格 JSON 先全部序列化、写入同目录临时文件，再分别替换目标。
    summary 中登记 samples 文件校验值，调用方可检测替换过程中断造成的不一致。
    此函数不提供两文件整体事务，也不清理其他文件。
    """
    required = {"protocol", "commit", "summary", "kept", "reason"}
    if not required <= set(matrix_payload):
        raise ValueError("主矩阵缺少下游兼容字段")
    if set(matrix_payload["summary"]) != set(MATRIX_CONFIGS):
        raise ValueError("主矩阵必须保留完整 16 格和 Value 诊断")
    if (not isinstance(matrix_payload["kept"], list)
            or any(name not in matrix_payload["summary"] for name in matrix_payload["kept"])):
        raise ValueError("冻结候选必须来自原主矩阵")
    for name, entry in matrix_payload["summary"].items():
        if not isinstance(entry, dict) or not {"error", "invalid", "bytes"} <= set(entry):
            raise ValueError(f"主矩阵缺少误差、无效组或字节字段：{name}")
    needed = {"model", "measurement_version", "provenance", "metrics"}
    if not needed <= set(diagnostic_payload):
        raise ValueError("配对诊断缺少测量格式、来源、汇总或测量状态")
    if not isinstance(samples, list):
        raise ValueError("samples.json 必须为题目记录列表")
    payload = copy.deepcopy(matrix_payload)
    payload["schema_version"] = SCHEMA_VERSION
    payload["model"] = diagnostic_payload["model"]
    payload["paired_diagnostic"] = {
        key: copy.deepcopy(diagnostic_payload[key])
        for key in ("measurement_version", "provenance", "metrics")
    }
    status = measurement_status
    if status is None:
        status = diagnostic_payload.get("measurement_status")
    if status is None:
        raise ValueError("必须显式提供 measurement_status，避免把待测指标写成已测")
    payload["measurement_status"] = copy.deepcopy(status)
    _validate_json(samples, "samples.json")
    sample_text = json.dumps(samples, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    payload["samples_file"] = {
        "path": "samples.json",
        "sha256": hashlib.sha256(sample_text.encode("utf-8")).hexdigest(),
        "rows": len(samples),
    }
    _validate_json(payload, "summary.json")
    serialized = {
        "summary.json": json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        "samples.json": sample_text,
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    try:
        for name, text in serialized.items():
            handle, temporary = tempfile.mkstemp(
                prefix=".representation-", suffix=".tmp", dir=destination
            )
            staged[name] = Path(temporary)
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for name in ("samples.json", "summary.json"):
            os.replace(staged[name], destination / name)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
