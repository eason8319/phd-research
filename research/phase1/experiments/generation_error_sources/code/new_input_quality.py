"""在新检索和四问联答上比较三个 K8V4 工作点、同输入 KV8，以及离线 pre 的 H=0。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from kvstudy.intervals import class_unit_passes, quality_state
from kvstudy.paths import project_root
from phase1.quality_stats import _interval, _paired
from phase1.ruler_retrieval import file_hash


BF16_GENERATION = "bf16_weights_sdpa_prefill_fp32_attention_decode_greedy"
CONDITIONS = (
    ("kv8-per_token-post", 0),
    ("k8v4-per_token-pre", 0),
    ("k8v4-per_token-post", 0),
    ("k8v4-per_channel-pre", 0),
    ("k8v4-per_channel-pre", 8),
)
PLANNED_NEW_GENERATIONS = 800
OFFLINE_PRE = "k8v4-per_channel-pre"
CANDIDATES = CONDITIONS[1:]
MODELS = ("qwen", "llama")
TASKS = ("retrieval_ruler", "hotpot_four")
MIN_LONG_TOKENS = 256
RESULTS = Path("research/phase1/experiments/generation_error_sources/results")
RUNTIME = Path("research/.runtime/new-input-quality")
FORBIDDEN_PREFIXES = ("v_retrieval_bf16.json", "v_hotpot_long_")
NUMERICAL_FILES = {
    "retrieval_ruler": (
        "research/phase1/src/phase1/runner.py",
        "research/phase1/src/phase1/score.py",
        "research/phase1/src/phase1/result_format.py",
        "research/src/kvstudy/attention.py",
        "research/phase1/experiments/generation_error_sources/code/long_baseline.py",
        "research/phase1/experiments/generation_error_sources/code/short_baseline.py",
        "research/phase1/src/phase1/ruler_retrieval.py",
        "research/phase1/configs/retrieval.json",
        "research/phase1/splits/retrieval_ruler.json",
    ),
    "hotpot_four": (
        "research/phase1/experiments/generation_error_sources/code/four_question_baseline.py",
        "research/phase1/src/phase1/runner.py",
        "research/phase1/src/phase1/score.py",
        "research/phase1/src/phase1/result_format.py",
        "research/src/kvstudy/attention.py",
        "research/phase1/experiments/generation_error_sources/code/long_baseline.py",
        "research/phase1/experiments/generation_error_sources/code/short_baseline.py",
        "research/phase1/configs/long_decode.json",
        "research/phase1/splits/hotpot_four.json",
        "research/phase1/splits/hotpot.json",
    ),
}
PUBLICATIONS = {
    "retrieval_ruler": "research/.runtime/retrieval-ruler/publication.json",
    "hotpot_four": "research/.runtime/four-question-decode/publication.json",
}
REVIEWS = {
    "retrieval_ruler": RESULTS / "retrieval_baseline_review.json",
    "hotpot_four": RESULTS / "long_decode_baseline_review.json",
}
TRACKED = (
    "research/phase1/src/phase1/generation.py",
    "research/phase1/src/phase1/inputs.py",
    "research/phase1/src/phase1/models.py",
    "research/phase1/src/phase1/quality_stats.py",
    "research/phase1/src/phase1/reuse.py",
    "research/phase1/experiments/generation_error_sources/code/new_input_quality.py",
    "research/phase1/experiments/generation_error_sources/code/quantized_generation.py",
    *NUMERICAL_FILES["retrieval_ruler"],
    *NUMERICAL_FILES["hotpot_four"],
    "research/src/kvstudy/intervals.py",
    "research/phase1/configs/quality.md",
    "research/phase1/guide.md",
)


def planned_new_generations() -> int:
    """两模型、两任务、四个量化条件加 KV8 的计划生成次数。"""
    return len(MODELS) * len(CONDITIONS) * (64 + 16)


def assert_allowed_reference(path: str) -> None:
    """旧检索和历史单问长解码不能进入这组配对。"""
    name = Path(path).name
    if name == "v_retrieval_bf16.json" or name.startswith("v_hotpot_long_"):
        raise RuntimeError(f"旧参考不能进入新输入比较：{name}")


def length_subset(rows: list[dict[str, Any]]) -> list[str]:
    """长解码配对成员只由参照的生成长度决定。"""
    return [row["id"] for row in rows if row["new_tokens"] >= MIN_LONG_TOKENS]


def paired_ids(result: dict[str, Any], task: str) -> list[str]:
    """检索使用全部冻结题；四问使用 BF16 达到 256 token 的请求。"""
    rows = result["rows"]
    if task == "retrieval_ruler":
        if result["n"] != 64 or len(rows) != 64:
            raise RuntimeError("新检索对照必须是 64 题")
        return [row["id"] for row in rows]
    if task != "hotpot_four":
        raise ValueError(f"未知新输入任务：{task}")
    ids = length_subset(rows)
    if len(ids) != 16 or not result.get("baseline_usable"):
        raise RuntimeError("四问联答 BF16 长度子集不是 16 个可用请求")
    return ids


def _view(payload: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    """按给定 ID 取配对行；不因停止原因、缺答案或生成长度丢弃候选。"""
    indexed = {row["id"]: row for row in payload["rows"]}
    missing = [item_id for item_id in ids if item_id not in indexed]
    if missing:
        raise RuntimeError(f"配对样本缺失：{missing[0]}")
    rows = [indexed[item_id] for item_id in ids]
    return {"n": len(rows), "rows": rows}


def offline_residual_category(h0_passes: bool, h8_passes: bool) -> str:
    """只按新输入上已测的 H=0 与 H=8 记录离线 pre 是否依赖残留。"""
    if h0_passes:
        return "无残留可行"
    if h8_passes:
        return "依赖缓冲"
    return "两个已测精度都未类通过"


def offline_residual_pair(
    h0: dict[str, Any], h8: dict[str, Any], ids: list[str], task: str,
    seed: int, draws: int, h0_passes: bool, h8_passes: bool,
) -> dict[str, Any]:
    """H=8 减 H=0；正值表示残留精度高于无残留。"""
    score_task = "hotpot" if task == "hotpot_four" else "retrieval"
    drop = _paired(_view(h0, ids), _view(h8, ids), score_task, seed, draws)
    return {
        "category": offline_residual_category(h0_passes, h8_passes),
        "h16": "未测",
        "gain_h8_minus_h0": {
            **drop, "estimate": -drop["estimate"],
            "lower": None if drop["upper"] is None else -drop["upper"],
            "upper": None if drop["lower"] is None else -drop["lower"],
            "direction": "h8_minus_h0",
        },
    }


def judge_condition(
    reference: dict[str, Any], kv8: dict[str, Any], candidate: dict[str, Any],
    task: str, seed: int, draws: int,
) -> dict[str, Any]:
    """计算相对 BF16 与 KV8 的下降，并分写类通过和保留。"""
    ids = paired_ids(reference, task)
    old, medium, new = _view(reference, ids), _view(kv8, ids), _view(candidate, ids)
    score_task = "hotpot" if task == "hotpot_four" else "retrieval"
    versus_bf16 = _paired(old, new, score_task, seed, draws)
    versus_kv8 = _paired(medium, new, score_task, seed, draws)
    invalid = sum(row.get("invalid_groups", 0) > 0 for row in new["rows"])
    format_ok = invalid < 2
    state = quality_state(
        _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok,
    )
    bounds = (
        format_ok and versus_bf16["upper"] is not None and versus_kv8["upper"] is not None
        and versus_bf16["upper"] <= 2 and versus_kv8["upper"] <= 2
    )
    if state == "保留":
        state = "开发晋级"
    score_field = "f1" if task == "hotpot_four" else "correct"
    if score_field == "correct":
        score = 100 * sum(row["correct"] for row in new["rows"]) / len(ids)
    else:
        score = sum(row["f1"] for row in new["rows"]) / len(ids)
    return {
        "n": len(ids), "score": score, "invalid_samples": invalid,
        "versus_bf16": versus_bf16, "versus_kv8": versus_kv8, "state": state,
        "class_unit_passes": class_unit_passes(
            _interval(versus_bf16), _interval(versus_kv8), format_ok=format_ok,
        ),
        "interval_meets_retention_bounds": bounds, "retention_awarded": False,
        "stops": dict(Counter(row["stop"] for row in new["rows"])),
        "below_256_tokens": sum(row["new_tokens"] < MIN_LONG_TOKENS for row in new["rows"]),
        "missing_answers": sum(row.get("missing_answers", 0) for row in new["rows"]),
    }


def publication_hashes(payload: dict[str, Any]) -> dict[str, str]:
    """兼容两种已发布清单的文件记录。"""
    files = payload["files"]
    if isinstance(files, dict):
        return dict(files)
    return {item["path"]: item["sha256"] for item in files}


def check_reuse(
    *,
    result: dict[str, Any],
    actual_sha256: str,
    review_sha256: str,
    publication_hashes_by_path: dict[str, str],
    current_hashes: dict[str, str],
    numerical_files: tuple[str, ...],
    manifest_rows: list[dict[str, Any]],
    task: str,
    compatible_measurement: bool = False,
) -> None:
    """已审核 BF16 仅在数值实现、冻结输入和结果字节都未变时复用。"""
    protocol = {
        "retrieval_ruler": "phase1-ruler-number-retrieval",
        "hotpot_four": "phase1-four-question-long-decode",
    }[task]
    if (result.get("generation") != BF16_GENERATION or result.get("config") != "bf16-h0"
            or result.get("protocol") != protocol):
        raise RuntimeError("BF16 数值实现或协议与已审核基线不一致")
    if actual_sha256 != review_sha256:
        raise RuntimeError("BF16 结果已不是审核通过的那一份")
    for relative in numerical_files:
        if (not compatible_measurement
                and current_hashes[relative] != publication_hashes_by_path[relative]):
            raise RuntimeError(f"数值实现或冻结输入与已审核 BF16 不一致：{relative}")
    if [row["id"] for row in result["rows"]] != [row["id"] for row in manifest_rows]:
        raise RuntimeError("BF16 题目顺序与冻结清单不一致")
    for row, identity in zip(result["rows"], manifest_rows):
        if (row["input_sha256"] != identity["sha256"]
                or row["input_tokens"] != identity["tokens"]):
            raise RuntimeError(f"BF16 输入与冻结清单不一致：{row['id']}")


def _read(path: Path) -> dict[str, Any]:
    assert_allowed_reference(path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def _measurement(root: Path, model: str, task: str, name: str, h_bits: int) -> Path:
    suffix = f"-residual{h_bits}" if h_bits else ""
    path = root / RESULTS / model / f"v_{task}_{name}{suffix}.json"
    assert_allowed_reference(path.name)
    return path


def _manifest_rows(root: Path, model: str, task: str) -> list[dict[str, Any]]:
    if task == "retrieval_ruler":
        from phase1.inputs import _samples

        return _samples(root, model, task, "v")
    from phase1.inputs import four_samples as samples

    return samples(root, model)


def _review_sha(review: dict[str, Any], model: str, task: str) -> str:
    entry = review["models"][model]
    if task == "retrieval_ruler":
        return entry["result_sha256"]
    return entry["result"]["sha256"]


def reuse_baselines(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """核对两模型两任务的已审核 BF16，不打开旧检索或单问长解码。"""
    from phase1.reuse import computation_signature

    registry_path = root / (
        "research/phase1/experiments/static_layer_allocation/results/reference_reuse.json"
    )
    registry = _read(registry_path) if registry_path.exists() else {}
    signature = computation_signature(root)
    loaded = {}
    for task in TASKS:
        publication = _read(root / PUBLICATIONS[task])
        review = _read(root / REVIEWS[task])
        if review.get("status") != "passed":
            raise RuntimeError(f"BF16 审核未通过：{task}")
        published = publication_hashes(publication)
        current = {relative: file_hash(root / relative) for relative in NUMERICAL_FILES[task]}
        loaded[task] = {}
        for model in MODELS:
            path = root / RESULTS / model / f"v_{task}_bf16.json"
            result = _read(path)
            validation = registry.get("inventory", {}).get(
                f"{model}/{task}/bf16", {}
            ).get("validation", {})
            check_reuse(
                result=result, actual_sha256=file_hash(path),
                review_sha256=_review_sha(review, model, task),
                publication_hashes_by_path=published, current_hashes=current,
                numerical_files=NUMERICAL_FILES[task],
                manifest_rows=_manifest_rows(root, model, task), task=task,
                compatible_measurement=(
                    validation.get("status") == "passed"
                    and validation.get("source_sha256") == file_hash(path)
                    and validation.get("computation_signature") == signature
                ),
            )
            if task == "retrieval_ruler" and result["em"] < 90:
                raise RuntimeError(f"{model} 新检索 BF16 未达下限")
            loaded[task][model] = result
    if planned_new_generations() != PLANNED_NEW_GENERATIONS:
        raise RuntimeError(f"计划生成次数不是 {PLANNED_NEW_GENERATIONS}")
    return loaded


def _tracked_hashes(root: Path) -> dict[str, str]:
    unique = list(dict.fromkeys(TRACKED))
    return {relative: file_hash(root / relative) for relative in unique}


def _wait_for_gpu(device_family: str) -> None:
    device_name, required_mib = {
        "ada48": ("RTX 6000 Ada", 24576), "rtx4090": ("RTX 4090", 23552),
    }[device_family]
    while True:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        ready = []
        for line in raw.strip().splitlines():
            index, name, free = [part.strip() for part in line.split(",")]
            if device_name in name and int(free) >= required_mib:
                ready.append((int(free), index))
        print(datetime.now(timezone.utc).isoformat(), raw.strip(), flush=True)
        if ready:
            os.environ["CUDA_VISIBLE_DEVICES"] = max(ready)[1]
            return
        time.sleep(60)


def _validate_measurement(
    path: Path, model: str, task: str, name: str, h_bits: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    value = _read(path)
    if (value["model"] != model or value["config"] != name or value["task"] != f"v-{task}"
            or value.get("residual_bits", 0) != h_bits or value["n"] != reference["n"]
            or [row["id"] for row in value["rows"]] != [row["id"] for row in reference["rows"]]):
        raise RuntimeError(f"量化结果与冻结输入不一致：{path.name}")
    for old, new in zip(reference["rows"], value["rows"]):
        if (old["input_sha256"] != new["input_sha256"]
                or old["input_tokens"] != new["input_tokens"]
                or new["stop"] not in ("eos", "limit", "invalid")
                or new["new_tokens"] < 1):
            raise RuntimeError(f"量化样本不能与 BF16 配对：{new['id']}")
    return value


def summarize(root: Path, commit: str, baselines: dict[str, dict[str, dict[str, Any]]]) -> None:
    """只汇总本轮新输入；保留判据另记，不把类通过写成保留。"""
    seeds = _read(root / "research/phase1/splits/seeds.json")
    seed, draws = seeds["bootstrap"], seeds["bootstrap_draws"]
    models: dict[str, Any] = {}
    for model in MODELS:
        models[model] = {}
        for task in TASKS:
            reference = baselines[task][model]
            kv8_path = _measurement(root, model, task, *CONDITIONS[0])
            kv8 = _validate_measurement(kv8_path, model, task, *CONDITIONS[0], reference)
            ids = paired_ids(reference, task)
            versus_bf16 = _paired(
                _view(reference, ids), _view(kv8, ids),
                "hotpot" if task == "hotpot_four" else "retrieval", seed, draws,
            )
            cells = {}
            measured: dict[tuple[str, int], dict[str, Any]] = {}
            for name, h_bits in CANDIDATES:
                path = _measurement(root, model, task, name, h_bits)
                candidate = _validate_measurement(path, model, task, name, h_bits, reference)
                cell = judge_condition(reference, kv8, candidate, task, seed, draws)
                cell.update(path=path.relative_to(root).as_posix(), sha256=file_hash(path),
                            residual_bits=h_bits, residual_tokens=128 if h_bits else 0)
                cells[f"{name}/H={h_bits}"] = cell
                measured[(name, h_bits)] = candidate
            h0_key, h8_key = f"{OFFLINE_PRE}/H=0", f"{OFFLINE_PRE}/H=8"
            models[model][task] = {
                "bf16_reused": True, "paired_ids": ids,
                "kv8": {
                    "path": kv8_path.relative_to(root).as_posix(),
                    "sha256": file_hash(kv8_path), "versus_bf16": versus_bf16,
                },
                "candidates": cells,
                "offline_pre_residual": offline_residual_pair(
                    measured[(OFFLINE_PRE, 0)], measured[(OFFLINE_PRE, 8)], ids, task,
                    seed, draws, cells[h0_key]["class_unit_passes"],
                    cells[h8_key]["class_unit_passes"],
                ),
            }
    payload = {
        "protocol": "phase1-new-input-candidate-quality", "commit": commit,
        "planned_new_generations": planned_new_generations(),
        "conditions": [
            {"config": name, "residual_bits": h_bits} for name, h_bits in CONDITIONS
        ],
        "bf16_generation": BF16_GENERATION,
        "excluded_references": list(FORBIDDEN_PREFIXES),
        "label_rule": {
            "class_unit_passes": "未淘汰，且相对 BF16 与 KV8 的下降点估计都不超过 2",
            "state": "状态表结果；区间上界已满足保留时仍记开发晋级",
            "retention_awarded": "本轮不是 128/256 扩样后的最终候选，不授予保留",
        },
        "residual_dependence": {
            "format": OFFLINE_PRE, "compared": [0, 8], "not_measured": 16,
            "rule": "H=0 类通过为无残留可行；仅 H=8 类通过为依赖缓冲；两者都未类通过不外推 H=16",
        },
        "models": models,
    }
    destination = root / RESULTS / "new_input_quality.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _progress_path(root: Path, model: str, task: str, name: str, h_bits: int) -> Path:
    """定位逐题恢复文件；新条件尚无进度时返回将要写入的路径。"""
    from phase1.runtime_sources import _runtime_path

    suffix = f"-residual{h_bits}" if h_bits else ""
    return _runtime_path(
        root, f"generation-{model}-{task}-v-{name}{suffix}.json",
        f"e2-{model}-{task}-v-{name}{suffix}.json",
    )


def _generation_source(
    root: Path, publication: dict[str, Any], model: str, task: str, name: str, h_bits: int,
) -> dict[str, Any]:
    """数值实现未变时沿用已有进度里的来源，避免把编排更新当成重新生成。"""
    progress = _progress_path(root, model, task, name, h_bits)
    if not progress.exists():
        return publication
    stored = json.loads(progress.read_text(encoding="utf-8")).get("execution_source") or {}
    stored_files = stored.get("files") or {}
    current = publication["files"]
    for relative in NUMERICAL_FILES["retrieval_ruler"] + NUMERICAL_FILES["hotpot_four"]:
        if "quantized_generation.py" in relative:
            continue
        if stored_files.get(relative) != current[relative]:
            raise RuntimeError(f"生成实现已变，不能沿用旧进度：{relative}")
    quantized = "research/phase1/experiments/generation_error_sources/code/quantized_generation.py"
    if stored_files.get(quantized) != current[quantized]:
        raise RuntimeError("量化生成实现已变，不能沿用旧进度")
    return stored


def run(commit: str, device_family: str = "ada48", summarize_only: bool = False) -> None:
    """先核对可复用 BF16，再补齐缺测生成并汇总。不调用选择或下游实验。"""
    root = project_root()
    if planned_new_generations() != PLANNED_NEW_GENERATIONS:
        raise RuntimeError(f"计划生成次数不是 {PLANNED_NEW_GENERATIONS}")
    baselines = reuse_baselines(root)
    runtime = root / RUNTIME
    runtime.mkdir(parents=True, exist_ok=True)
    publication_path = runtime / "publication.json"
    hashes = _tracked_hashes(root)
    publication = {
        "base_commit": commit, "files": hashes,
        "scope": "new-input three candidates, offline pre H=0, plus kv8",
        "planned_new_generations": PLANNED_NEW_GENERATIONS,
    }
    if publication_path.exists():
        existing = _read(publication_path)
        if existing.get("base_commit") != commit:
            raise RuntimeError("恢复时的基准提交与已发布清单不一致")
        if existing.get("files") != hashes:
            _generation_source(root, publication, "qwen", "retrieval_ruler", CONDITIONS[0][0], 0)
    publication_path.write_text(
        json.dumps(publication, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    missing = [
        (model, task, name, h_bits)
        for model in MODELS for task in TASKS for name, h_bits in CONDITIONS
        if not _measurement(root, model, task, name, h_bits).exists()
    ]
    if missing and not summarize_only:
        from experiments.generation_error_sources.code.quantized_generation import run as generate

        _wait_for_gpu(device_family)
        for model, task, name, h_bits in missing:
            reference = baselines[task][model]
            path = _measurement(root, model, task, name, h_bits)
            if not path.exists():
                print("START", model, task, name, h_bits, flush=True)
                generate(
                    model, task, "v", name, commit, h_bits, save_predictions=True,
                    guard_severe_failure=False,
                    execution_source=_generation_source(
                        root, publication, model, task, name, h_bits,
                    ),
                )
            _validate_measurement(path, model, task, name, h_bits, reference)
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    elif missing:
        raise RuntimeError(f"仍缺量化结果：{missing[0]}")
    summarize(root, commit, baselines)
    print("COMPLETE", planned_new_generations(), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device-family", choices=("ada48", "rtx4090"), default="ada48")
    parser.add_argument("--summarize-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.commit, arguments.device_family, arguments.summarize_only)
