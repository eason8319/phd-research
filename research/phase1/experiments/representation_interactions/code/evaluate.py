"""位宽与表示交互唯一入口：固定轨迹配对诊断，可显式重测主矩阵或仅汇总明细。"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Iterator
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.representation_interactions.code.metrics import output_metrics, paired_head_metrics
from experiments.representation_interactions.code.results import (
    PAIRS,
    SCHEMA_VERSION,
    summarize_diagnostics,
    write_results,
)
from experiments.representation_interactions.code.select import (
    CANDIDATE_POLICY,
    select_representations,
)
from kvstudy.attention import attention_fp32, relative_output_error
from kvstudy.ledger import account_pair
from kvstudy.paths import project_root
from phase1.representation import (
    G_C,
    MODELS,
    _capture,
    _empty_minmax,
    _fit_minmax,
    _key_params,
    _load_c_items,
    _quantize_side,
    _render_gsm,
    _rotate_key,
    _rotate_query,
    _update_minmax,
)
from phase1.runner import load_model


VALUE_DIAGNOSTIC = "kv4-per_channel-pre-vchannel"
HEAD_FIELDS = (
    "weighted_score_var", "paired_probability_kl",
)


def _settings() -> dict[str, Any]:
    """读取唯一诊断配置，拒绝缺失字段或不兼容的测量格式。"""
    path = project_root() / "research/phase1/configs/representation_diagnostics.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    if settings["measurement_version"] != SCHEMA_VERSION:
        raise ValueError("诊断配置的测量格式与执行代码不一致")
    for name in ("bootstrap_seed", "bootstrap_resamples"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"诊断参数 {name} 必须为整数")
        if name != "bootstrap_seed" and value < 1:
            raise ValueError(f"诊断参数 {name} 必须大于零")
    return settings


def _sha256(path: Path) -> str:
    """流式计算文件校验值，不把完整明细文件加载到内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_hashes() -> dict[str, str]:
    """索引位宽与表示交互四个模块和阶段共用辅助模块的代码校验值。"""
    root = project_root()
    directory = "research/phase1/experiments/representation_interactions/code"
    paths = [f"{directory}/{name}.py" for name in ("evaluate", "metrics", "results", "select")]
    paths.append("research/phase1/src/phase1/representation.py")
    return {path: _sha256(root / path) for path in paths}


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    """逐行读取严格 JSON 明细；解析失败立即停止汇总。"""
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写完整临时文件，再替换 JSON 索引。"""
    partial = path.with_suffix(".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _expected_items(model_key: str) -> dict[str, dict[str, Any]]:
    """从冻结划分读取来源标识；仅汇总时无需模型和原题全文。"""
    root = project_root()
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    indexed = {
        row["id"]: {
            "kind": "math", "fold": row["fold"],
            "source_sha256": row["question_sha256"],
        }
        for row in gsm["c"]
    }
    retrieval = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
    for row in retrieval:
        if row["id"].startswith("synth-c-"):
            indexed[row["id"]] = {
                "kind": "retrieval", "fold": row["fold"],
                "source_sha256": row["models"][model_key]["sha256"],
            }
    return indexed


def _capture_items(model_key: str, tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    """每题只捕获一次高精度轨迹，核对冻结来源和检索 token 长度。"""
    captured = []
    for item in _load_c_items(model_key):
        if item["kind"] == "math":
            ids = _render_gsm(tokenizer, item["question"], model_key == "llama")
        else:
            ids = tokenizer(
                item["text"], add_special_tokens=False, return_tensors="pt"
            ).input_ids.to("cuda")
            if int(ids.shape[1]) != item["tokens"]:
                raise RuntimeError(f"检索长度与冻结记录不符：{item['id']}")
        blob = _capture(model, ids)
        blob.update({
            name: item[name] for name in ("id", "kind", "fold", "source_sha256")
        })
        blob["seq"] = int(ids.shape[1])
        captured.append(blob)
        print(f"捕获 {item['id']}，输入 {blob['seq']} token", flush=True)
    return captured


def _fit_scales(
    captured: list[dict[str, Any]], refresh_matrix: bool
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """拟合互补折校准参数；只有主矩阵重测才生成下游使用的 all-C 参数。"""
    layers = len(captured[0]["pre"])
    kv_heads, _length, dim = captured[0]["pre"][0].shape
    folds = {}
    full = None
    groups = ("pre", "post", "value") if refresh_matrix else ("pre", "post")
    for fold in (0, 1):
        accumulators = {name: _empty_minmax(layers, kv_heads, dim) for name in groups}
        for blob in captured:
            if blob["fold"] == fold:
                for name, accumulator in accumulators.items():
                    _update_minmax(accumulator, blob[name])
        folds[fold] = {
            f"k_{rope}{suffix}": _fit_minmax(accumulators[rope], bits)
            for rope in ("pre", "post") for bits, suffix in ((8, ""), (4, "4"))
        }
        if refresh_matrix:
            folds[fold]["v4"] = _fit_minmax(accumulators["value"], 4)
    if refresh_matrix:
        accumulators = {name: _empty_minmax(layers, kv_heads, dim) for name in groups}
        for blob in captured:
            for name, accumulator in accumulators.items():
                _update_minmax(accumulator, blob[name])
        full = {
            f"k_{rope}{bits}": _fit_minmax(accumulators[rope], bits)
            for rope in ("pre", "post") for bits in (8, 4)
        }
        full.update({
            f"v{bits}": _fit_minmax(accumulators["value"], bits) for bits in (8, 4)
        })
    return folds, full


def _scalar(value: torch.Tensor) -> float | int | bool | None:
    """将标量张量转成严格 JSON 值；未定义指标写 null，保留旁边的布尔标记。"""
    result = value.item()
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result


def _diagnostic_layer(
    query: torch.Tensor,
    key8: torch.Tensor,
    key4: torch.Tensor,
    value8: torch.Tensor,
    value4: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    """用 FP64 分数和概率计算解释指标，按 KV head 分块以避免复制完整 Value。"""
    repeats = query.shape[0] // key8.shape[0]
    heads = []
    outputs = {name: [] for name in ("kv8", "k4v8", "k8v4", "prediction")}
    for kv_head in range(key8.shape[0]):
        start = kv_head * repeats
        query_group = query[start:start + repeats].double()
        score8 = query_group @ key8[kv_head].double().T / math.sqrt(query.shape[-1])
        score4 = query_group @ key4[kv_head].double().T / math.sqrt(query.shape[-1])
        value8_group = value8[kv_head].unsqueeze(0).expand(repeats, -1, -1)
        value4_group = value4[kv_head].unsqueeze(0).expand(repeats, -1, -1)
        metrics = paired_head_metrics(
            score8, score4, value8_group, value4_group, reference[start:start + repeats]
        )
        for head in range(repeats):
            heads.append({name: _scalar(metrics[name][head]) for name in HEAD_FIELDS})
        probability8 = torch.log_softmax(score8, dim=-1).exp()
        probability4 = torch.log_softmax(score4, dim=-1).exp()
        delta = score4 - score8
        centered = delta - (probability8 * delta).sum(dim=-1, keepdim=True)
        outputs["kv8"].append(probability8 @ value8[kv_head].double())
        outputs["k4v8"].append(probability4 @ value8[kv_head].double())
        outputs["k8v4"].append(probability8 @ value4[kv_head].double())
        outputs["prediction"].append((probability8 * centered) @ value8[kv_head].double())
    output = output_metrics(
        reference, *(torch.cat(outputs[name]) for name in outputs)
    )
    return {
        "heads": heads,
        "output": {name: _scalar(tensor) for name, tensor in output.items()},
    }


def _matrix_configs() -> list[dict[str, Any]]:
    """构建原主矩阵的 16 格与 Value 诊断格，保留原选择口径。"""
    configs = []
    for pair in PAIRS:
        mode, rope = pair.rsplit("-", 1)
        for width, key_bits, value_bits in (
            ("kv8", 8, 8), ("k8v4", 8, 4), ("k4v8", 4, 8), ("kv4", 4, 4)
        ):
            configs.append({
                "name": f"{width}-{pair}", "key_mode": mode, "rope": rope,
                "k_bits": key_bits, "v_bits": value_bits, "v_mode": "per_token",
                "tier": "6" if width in ("k8v4", "k4v8") else (
                    "4" if width == "kv4" else "base"
                ),
            })
    configs.append({
        "name": VALUE_DIAGNOSTIC, "key_mode": "per_channel", "rope": "pre",
        "k_bits": 4, "v_bits": 4, "v_mode": "per_channel", "tier": "4",
    })
    return configs


def _matrix_bytes(cfg: dict[str, Any], blob: dict[str, Any]) -> float:
    """按原输入长度计算整层常驻字节，保持已有主矩阵的记账口径。"""
    kv_heads, _length, dim = blob["pre"][0].shape
    ledger = account_pair(
        layers=len(blob["pre"]), kv_heads=kv_heads, seq=blob["seq"], dim=dim,
        k_bits=cfg["k_bits"], v_bits=cfg["v_bits"], k_mode=cfg["key_mode"],
        v_mode=cfg["v_mode"], g_c=G_C, g_t=32, block_tokens=64,
    )
    return float(ledger.resident_bytes)


def _score_item(
    blob: dict[str, Any], model: Any, scales: dict[int, dict[str, Any]],
    refresh_matrix: bool,
) -> list[dict[str, Any]]:
    """在同一轨迹上逐层量化一次 K/V，复用到四种配对和可选主矩阵。"""
    rows = {
        pair: {
            **{name: blob[name] for name in ("id", "kind", "fold", "source_sha256")},
            "pair": pair, "input_tokens": blob["seq"], "layers": [],
        }
        for pair in PAIRS
    }
    configs = _matrix_configs() if refresh_matrix else []
    matrix = {
        cfg["name"]: {
            "layer_errors": [], "invalid": 0, "bytes": _matrix_bytes(cfg, blob),
            "zero_reference_outputs": 0,
        }
        for cfg in configs
    }
    scale_pack = scales[1 - blob["fold"]]
    for layer, value in enumerate(blob["value"]):
        values = {bits: _quantize_side(value, bits, "per_token", None) for bits in (8, 4)}
        query = _rotate_query(model, blob["query"][layer], blob["seq"])
        reference, _score, _probability = attention_fp32(query, blob["post"][layer], value)
        keys = {}
        for pair in PAIRS:
            mode, rope = pair.rsplit("-", 1)
            source = blob[rope][layer]
            for bits in (8, 4):
                cfg = {"key_mode": mode, "rope": rope, "k_bits": bits}
                params = _key_params(scale_pack, cfg, layer)
                restored, stats = _quantize_side(source, bits, mode, params)
                keys[pair, bits] = (
                    _rotate_key(model, restored) if rope == "pre" else restored, stats
                )
            diagnostic = _diagnostic_layer(
                query, keys[pair, 8][0], keys[pair, 4][0],
                values[8][0], values[4][0], reference,
            )
            rows[pair]["layers"].append({"layer": layer, **diagnostic})
        if refresh_matrix:
            value_channel = _quantize_side(value, 4, "per_channel", scale_pack["v4"][layer])
            for cfg in configs:
                pair = f"{cfg['key_mode']}-{cfg['rope']}"
                key_hat, key_stats = keys[pair, cfg["k_bits"]]
                value_hat, value_stats = (
                    value_channel if cfg["v_mode"] == "per_channel" else values[cfg["v_bits"]]
                )
                candidate, _score, _probability = attention_fp32(query, key_hat, value_hat)
                error, zero = relative_output_error(reference, candidate)
                matrix[cfg["name"]]["layer_errors"].append(None if zero else error)
                matrix[cfg["name"]]["zero_reference_outputs"] += int(zero)
                matrix[cfg["name"]]["invalid"] += (
                    key_stats.invalid_groups + value_stats.invalid_groups
                )
    if refresh_matrix:
        rows[PAIRS[0]]["matrix"] = matrix
    return list(rows.values())


def _matrix_payload(
    path: Path, commit: str, expected_ids: Collection[str]
) -> dict[str, Any]:
    """从逐题主矩阵记录重建两折均值、容量和原候选选择。"""
    configs = _matrix_configs()
    totals = {
        cfg["name"]: {
            "folds": {0: [], 1: []}, "invalid": 0, "bytes": [], "zero_reference_outputs": 0,
        }
        for cfg in configs
    }
    expected = set(expected_ids)
    seen = set()
    for row in _read_rows(path):
        matrix_rows = row.get("matrix", {})
        if not isinstance(matrix_rows, dict):
            raise RuntimeError(f"逐题主矩阵字段不是字典：{row['id']}")
        if row["pair"] == PAIRS[0] and set(matrix_rows) != set(totals):
            raise RuntimeError(f"逐题主矩阵配置不完整：{row['id']}")
        if matrix_rows and row["pair"] != PAIRS[0]:
            raise RuntimeError(f"逐题主矩阵重复记录：{row['id']}")
        if not matrix_rows:
            continue
        if row["id"] not in expected or row["id"] in seen:
            raise RuntimeError(f"逐题主矩阵样本重复或不属于冻结划分：{row['id']}")
        seen.add(row["id"])
        if (isinstance(row["fold"], bool) or not isinstance(row["fold"], int)
                or row["fold"] not in (0, 1)):
            raise RuntimeError(f"逐题主矩阵折编号无效：{row['id']}")
        for name, matrix in matrix_rows.items():
            layer_errors = matrix["layer_errors"]
            if (not isinstance(layer_errors, list) or not layer_errors
                    or len(layer_errors) != len(row["layers"])):
                raise RuntimeError(f"逐题主矩阵层数不完整：{row['id']} {name}")
            errors = [value for value in layer_errors if value is not None]
            if not errors:
                raise RuntimeError(f"该题没有可定义的主矩阵输出误差：{row['id']} {name}")
            for value in errors:
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not math.isfinite(value) or value < 0):
                    raise RuntimeError(f"逐题主矩阵输出误差无效：{row['id']} {name}")
            invalid = matrix["invalid"]
            if (isinstance(invalid, bool) or not isinstance(invalid, (float, int))
                    or not math.isfinite(invalid) or invalid < 0 or int(invalid) != invalid):
                raise RuntimeError(f"逐题主矩阵无效组计数无效：{row['id']} {name}")
            resident_bytes = matrix["bytes"]
            if (isinstance(resident_bytes, bool) or not isinstance(resident_bytes, (float, int))
                    or not math.isfinite(resident_bytes) or resident_bytes < 0):
                raise RuntimeError(f"逐题主矩阵容量无效：{row['id']} {name}")
            zero_outputs = len(layer_errors) - len(errors)
            zero_count = matrix["zero_reference_outputs"]
            if (isinstance(zero_count, bool) or not isinstance(zero_count, int)
                    or zero_count != zero_outputs):
                raise RuntimeError(f"逐题主矩阵零参考计数不一致：{row['id']} {name}")
            slot = totals[name]
            slot["folds"][row["fold"]].append(sum(errors) / len(errors))
            slot["invalid"] += invalid
            slot["bytes"].append(resident_bytes)
            slot["zero_reference_outputs"] += zero_outputs
    if seen != expected:
        raise RuntimeError("逐题主矩阵未完整覆盖冻结样本")
    summary = {}
    for cfg in configs:
        slot = totals[cfg["name"]]
        if not slot["folds"][0] or not slot["folds"][1]:
            raise RuntimeError(f"主矩阵缺少完整的两折记录：{cfg['name']}")
        error = 0.5 * sum(sum(values) / len(values) for values in slot["folds"].values())
        resident_bytes = sum(slot["bytes"])
        summary[cfg["name"]] = {
            "error": error, "invalid": float(slot["invalid"]), "bytes": resident_bytes,
            "zero_reference_outputs": slot["zero_reference_outputs"],
        }
    selected = select_representations(summary)
    return {
        "protocol": "phase1-full-key-retrieval", "commit": commit, "summary": summary,
        "kept": list(selected.kept), "reason": selected.reason,
        "candidate_policy": CANDIDATE_POLICY,
    }


def _finalize(model_key: str, manifest: dict[str, Any], commit: str) -> None:
    """审核来源、SHA256 和逐题完整性后写正式结果，不生成报告。"""
    root = project_root()
    allowed_paths = {
        f"research/.runtime/representation_diagnostics-{model_key}.jsonl",
        f"research/.runtime/h1-{model_key}.jsonl",
    }
    if manifest["detail_path"] not in allowed_paths:
        raise RuntimeError("位宽与表示交互明细路径不属于该模型的正式运行路径")
    detail = root / manifest["detail_path"]
    if _sha256(detail) != manifest["detail_sha256"]:
        raise RuntimeError("位宽与表示交互明细的校验值与来源索引不一致")
    expected = _expected_items(model_key)
    if manifest["expected_items"] != expected:
        raise RuntimeError("位宽与表示交互明细来源与当前冻结样本划分不一致")
    settings = manifest["settings"]
    metrics, samples, detail_checks = summarize_diagnostics(
        _read_rows(detail), expected, bootstrap_seed=settings["bootstrap_seed"],
        resamples=settings["bootstrap_resamples"],
    )
    provenance = {
        **manifest["provenance"], "summarizer_commit": commit,
        "summarizer_code_sha256": _code_hashes(),
    }
    detail_checks.update({"model": model_key, "detail_sha256": manifest["detail_sha256"]})
    provenance["review"] = {
        **provenance.get("review", {}),
        "status": "passed",
        "layer_statistics_unit": "equal-weight fold means by layer; no independent-layer CI",
        "detail_checks": detail_checks,
        "statistics_derivation": {
            "derived_at_utc": datetime.now(timezone.utc).isoformat(),
            "commit": commit, "code_sha256": provenance["summarizer_code_sha256"],
            "detail_sha256": manifest["detail_sha256"],
        },
    }
    diagnostic = {
        "model": model_key, "measurement_version": provenance["measurement_version"],
        "provenance": provenance, "metrics": metrics,
    }
    destination = root / "research/phase1/experiments/representation_interactions/results"
    write_results(
        destination / model_key, manifest["matrix_payload"], diagnostic, samples,
        measurement_status=manifest["measurement_status"],
    )


def run_model(model_key: str, commit: str, *, refresh_matrix: bool = False) -> None:
    """运行冻结 C 的配对诊断；显式重测主矩阵时才更新候选和 all-C 校准包。"""
    root = project_root()
    settings = _settings()
    destination = root / "research/phase1/experiments/representation_interactions/results"
    matrix_payload = json.loads((destination / model_key / "summary.json").read_text("utf-8"))
    source_summary_commit = matrix_payload["commit"]
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    with torch.inference_mode():
        captured = _capture_items(model_key, tokenizer, model)
        scales, full_scales = _fit_scales(captured, refresh_matrix)
        detail = root / "research/.runtime" / f"representation_diagnostics-{model_key}.jsonl"
        partial = detail.with_suffix(".partial")
        with partial.open("w", encoding="utf-8") as stream:
            for blob in captured:
                for row in _score_item(blob, model, scales, refresh_matrix):
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"评分 {blob['id']}，四种配对完成", flush=True)
        partial.replace(detail)
    del model
    torch.cuda.empty_cache()
    if refresh_matrix:
        matrix_payload = _matrix_payload(detail, commit, {blob["id"] for blob in captured})
    provenance = {
        "protocol": "phase1-full-key-retrieval",
        "measurement_version": settings["measurement_version"],
        "commit": commit, "model": model_key, "samples": len(captured), "pairs": len(PAIRS),
        "layers": len(captured[0]["pre"]), "query_heads": captured[0]["query"][0].shape[0],
        "detail_path": detail.relative_to(root).as_posix(), "detail_sha256": _sha256(detail),
        "code_sha256": _code_hashes(), "settings": settings,
        "source_summary_commit": source_summary_commit, "main_matrix_refreshed": refresh_matrix,
        "numeric_precision": {
            "captured_qkv": "float32", "reference_output": "float32",
            "diagnostic_scores_probabilities_outputs": "float64", "main_matrix": "float32",
        },
    }
    manifest = {
        "model": model_key, "settings": settings, "expected_items": _expected_items(model_key),
        "detail_path": provenance["detail_path"], "detail_sha256": provenance["detail_sha256"],
        "provenance": provenance, "matrix_payload": matrix_payload,
        "measurement_status": {
            "main_matrix": {
                "status": "measured", "measurement_version": "phase1-full-key-retrieval",
                "commit": matrix_payload["commit"],
            },
            **{
                name: {
                    "status": "measured", "measurement_version": settings["measurement_version"],
                }
                for name in (
                    "output_increments", "probability_change",
                    "first_order_prediction",
                )
            },
            "sample_statistics": {
                "status": "derived_from_measured",
                "measurement_version": settings["measurement_version"], "unit": "sample",
            },
        },
    }
    manifest_path = (
        root / "research/.runtime" / f"representation_diagnostics-{model_key}.manifest.json"
    )
    _write_json(manifest_path, manifest)
    _finalize(model_key, manifest, commit)
    if refresh_matrix:
        torch.save(
            {
                "protocol": "phase1-full-key-retrieval", "commit": commit, "source": "all-c",
                "ids": [blob["id"] for blob in captured], "params": full_scales,
            },
            root / "research/.runtime" / f"generation-calibration-{model_key}.pt",
        )


def summarize_only(model_key: str, commit: str) -> None:
    """仅审核既有位宽与表示交互来源索引并汇总，不捕获激活、不量化、不使用 GPU。"""
    root = project_root()
    path = root / "research/.runtime" / f"representation_diagnostics-{model_key}.manifest.json"
    if not path.exists():
        path = root / "research/.runtime" / f"h1-{model_key}.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["model"] != model_key:
        raise RuntimeError("位宽与表示交互来源索引的模型标识不匹配")
    if manifest["settings"]["measurement_version"] not in {SCHEMA_VERSION, "h1-paired-v2"}:
        raise RuntimeError("位宽与表示交互来源索引的测量格式不受当前代码支持")
    _finalize(model_key, manifest, commit)


def main() -> None:
    """选择配对诊断、显式主矩阵重测或无 GPU 汇总，并要求提供代码提交标识。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--commit", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refresh-matrix", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.commit.strip() or args.commit == "unknown":
        parser.error("--commit 必须提供真实的代码提交标识")
    if args.summarize_only:
        summarize_only(args.model, args.commit)
    else:
        run_model(args.model, args.commit, refresh_matrix=args.refresh_matrix)


if __name__ == "__main__":
    main()
