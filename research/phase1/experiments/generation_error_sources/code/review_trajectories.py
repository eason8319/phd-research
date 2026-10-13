"""独立核对传播结果与完整运行实体，汇总题目等权的机制指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
from typing import Any

from kvstudy.paths import project_root
from phase1.score import extract_final_answer, extract_gsm8k, gsm8k_exact, token_f1


CONDITIONS = [("k8v4-per_token-pre", 0), ("k8v4-per_token-post", 0),
              ("k8v4-per_channel-pre", 8)]


def _hash(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("缺少有效来源校验")


def _nonnegative(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("误差指标必须有限且非负")


def _layers(rows: list[dict[str, float]], count: int) -> None:
    if len(rows) != count:
        raise ValueError("层数不一致")
    for row in rows:
        for name in ("hidden",):
            _nonnegative(row[name])


def validate_trajectories(payload: dict[str, Any]) -> None:
    """验证完整病例、回放步数、首次分歧与重建位置，不从分歧推断答题正确性。"""
    if payload.get("schema") != "kv-propagation-v1":
        raise ValueError("不是当前传播实验结果")
    model = payload["model"]
    if model not in ("qwen", "llama"):
        raise ValueError("未知模型")
    count = 28 if model == "qwen" else 32
    rows = payload["rows"]
    if payload["n"] != 8 or len(rows) != 8 or len({row["id"] for row in rows}) != 8:
        raise ValueError("病例数量或身份不一致")
    _hash(payload["raw_source"]["sha256"])
    for task in ("gsm8k", "hotpot"):
        selected = [row for row in rows if row["task"] == task]
        if len(selected) != 4 or sum(row["role"] == "stable_control" for row in selected) != 1:
            raise ValueError("任务病例组成不符")
    for row in rows:
        _hash(row["input_sha256"])
        candidates = row["candidates"]
        if [(c["representation"], c["h"]) for c in candidates] != CONDITIONS:
            raise ValueError("三个共同候选不完整或顺序变化")
        reference_tokens = None
        for candidate in candidates:
            if candidate["invalid_groups"] != 0 or candidate["r"] != (128 if candidate["h"] else 0):
                raise ValueError("候选缓存条件无效")
            steps = candidate["steps"]
            if len(steps) != row["bf16_new_tokens"] - 1 or not steps:
                raise ValueError("回放缺步")
            current_tokens = [step["bf16_argmax"] for step in steps]
            if reference_tokens is not None and reference_tokens != current_tokens:
                raise ValueError("候选没有使用共同 BF16 参考")
            reference_tokens = current_tokens
            divergent = []
            for index, step in enumerate(steps):
                if step["decode_step"] != index or step["prediction_index"] != index + 1:
                    raise ValueError("回放位置不连续")
                for name in ("kl", "final_hidden_rse", "bf16_margin"):
                    _nonnegative(step[name])
                margin = step["candidate_reference_margin"]
                if not math.isfinite(margin):
                    raise ValueError("预测间隔不是有限数")
                differs = step["bf16_argmax"] != step["candidate_argmax"]
                if (differs and margin > 0) or (not differs and margin < 0):
                    raise ValueError("argmax 与预测间隔矛盾")
                if differs:
                    divergent.append(index + 1)
            expected = divergent[0] if divergent else None
            if candidate["first_divergence_index"] != expected:
                raise ValueError("首次分歧定位错误")
            free = candidate["paired_free_generation"]
            if (free["bf16_score"] != row["bf16_score"]
                    or free["delta"] != free["score"] - row["bf16_score"]
                    or not 0 <= free["score"] <= 100):
                raise ValueError("同次自由生成配对分数不一致")
            _layers(candidate["layer_mean_rse"], count)
            reset = candidate["history_reset"]
            if (reset is None) != (expected is None):
                raise ValueError("缺少或多出历史重建对照")
            if reset is not None:
                if (reset["decode_step"] != expected - 1
                        or reset["history_generated_tokens"] != expected - 1
                        or reset["bf16_argmax"] != steps[expected - 1]["bf16_argmax"]):
                    raise ValueError("历史重建位置或参考不符")
                _nonnegative(reset["kl"])


def reuse_quality(payload: dict[str, Any], root: Path) -> None:
    """只审核已引用的自由质量，不修改原评分或合成逐题质量结论。"""
    validate_trajectories(payload)
    plan_path = root / "research/phase1/configs/trajectory_replay.json"
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != payload["plan_sha256"]:
        raise ValueError("诊断病例配置变化")
    plan = json.loads(plan_path.read_text("utf-8-sig"))
    for source in plan["sources"]:
        if hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("冻结病例的历史质量来源变化")
    if [(r["task"], r["id"]) for r in payload["rows"]] != [
            (r["task"], r["id"]) for r in plan["cases"]]:
        raise ValueError("病例未按冻结配置执行")
    for row in payload["rows"]:
        for name, quality in row["free_quality"].items():
            path = root / quality["source"]
            path.resolve().relative_to(root.resolve())
            if hashlib.sha256(path.read_bytes()).hexdigest() != quality["sha256"]:
                raise ValueError("自由评分来源校验失败")
            source = json.loads(path.read_text("utf-8"))
            paired = next(r for r in source["rows"] if r["id"] == row["id"])
            if any(paired[k] != row[k] for k in ("input_sha256", "input_tokens")):
                raise ValueError("复用评分的实际输入不一致")
            score = paired["f1"] if row["task"] == "hotpot" else 100 * paired["correct"]
            if quality["score"] != score or quality["new_tokens"] != paired["new_tokens"]:
                raise ValueError("复用评分或生成长度不一致")
            if name == "bf16":
                matched = (paired["new_tokens"] == row["bf16_new_tokens"]
                           and abs(score - row["bf16_score"]) <= 1e-8)
                if matched != row["historical_bf16_score_length_match"]:
                    raise ValueError("历史 BF16 复现状态不一致")


def _read(path: Path) -> Any:
    return json.loads(path.read_text("utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _audit_raw(payload: dict[str, Any], root: Path, questions: dict[str, Any]
               ) -> list[dict[str, Any]]:
    """从完整 token 和层步矩阵独立重算分歧、评分、层均值。"""
    source = root / payload["raw_source"]["path"]
    source.resolve().relative_to(root.resolve())
    if _sha(source) != payload["raw_source"]["sha256"]:
        raise ValueError("完整轨迹来源校验失败")
    raw = _read(source)
    answers = []
    for key in ("model", "commit", "plan_sha256", "publication_sha256"):
        if raw[key] != payload[key]:
            raise ValueError("原始执行身份与回传记录不符")
    runtime = root / "research/.runtime"
    publication = runtime / f"trajectory-publication-{payload['model']}.json"
    if _sha(publication) != payload["publication_sha256"]:
        raise ValueError("执行源码发布清单不符")
    with tarfile.open(runtime / f"trajectory-source-{payload['model']}.tar.gz") as archive:
        for path, digest in _read(publication)["files"].items():
            content = archive.extractfile(path)
            if content is None or hashlib.sha256(content.read()).hexdigest() != digest:
                raise ValueError("执行源码快照与发布清单不一致")
    if len(raw["rows"]) != len(payload["rows"]):
        raise ValueError("完整轨迹病例数量不符")
    for row, full in zip(payload["rows"], raw["rows"]):
        if row["id"] != full["id"] or row["input_sha256"] != full["input_sha256"]:
            raise ValueError("完整轨迹病例身份不符")
        question = questions[f"{row['task']}/{row['id']}"]
        answer = {"id": row["id"], "task": row["task"], "question": question["question"],
                  "gold": extract_gsm8k(question["standard_answer"]) if row["task"] == "gsm8k"
                  else question["standard_answer"],
                  "bf16_final": extract_final_answer(full["bf16_text"]), "candidates": []}
        scorer = (lambda text: 100 * gsm8k_exact(text, question["standard_answer"])) if (
            row["task"] == "gsm8k"
        ) else (lambda text: token_f1(text, question["standard_answer"]))
        if abs(scorer(full["bf16_text"]) - row["bf16_score"]) > 1e-8:
            raise ValueError("完整 BF16 文本重新评分不符")
        tokens = full["bf16_tokens"]
        if len(tokens) != row["bf16_new_tokens"]:
            raise ValueError("完整 BF16 token 数不符")
        for measured, complete in zip(row["candidates"], full["candidates"]):
            steps = complete["steps"]
            if [step["bf16_argmax"] for step in steps] != tokens[1:]:
                raise ValueError("逐步参考 logits 与 BF16 实际 token 不符")
            for compact, detailed in zip(measured["steps"], steps):
                expected = {k: v for k, v in detailed.items() if k != "layers"}
                expected["final_hidden_rse"] = detailed["layers"][-1]["hidden"]
                if compact != expected:
                    raise ValueError("逐步回传标量不符")
            for layer, summary in enumerate(measured["layer_mean_rse"]):
                for name in ("hidden",):
                    actual = _mean([step["layers"][layer][name] for step in steps])
                    if not math.isclose(summary[name], actual, rel_tol=1e-12, abs_tol=1e-15):
                        raise ValueError("层汇总复算不符")
            if measured["history_reset"] != complete["history_reset"]:
                raise ValueError("历史干预结果不符")
            free = complete["paired_free_generation"]
            candidate_tokens = free["token_ids"]
            first = next((i for i, (a, b) in enumerate(zip(tokens, candidate_tokens))
                          if a != b), None)
            if first is None and len(tokens) != len(candidate_tokens):
                first = min(len(tokens), len(candidate_tokens))
            if first != measured["first_divergence_index"]:
                raise ValueError("自由生成与回放首分歧不一致")
            compact_free = measured["paired_free_generation"]
            if (abs(scorer(free["text"]) - compact_free["score"]) > 1e-8
                    or len(candidate_tokens) != compact_free["new_tokens"]):
                raise ValueError("自由文本评分或 token 数复算不符")
            answer["candidates"].append({"representation": measured["representation"],
                                         "h": measured["h"], "score": free["score"],
                                         "final": extract_final_answer(free["text"])})
        answers.append(answer)
    return answers


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    """步内先平均、题间再等权平均；干预只对出现分歧的配对位置汇总。"""
    rows = payload["rows"]
    summaries = []
    for index in range(3):
        selected = [row["candidates"][index] for row in rows]
        reset_pairs = [(candidate["steps"][candidate["history_reset"]["decode_step"]],
                        candidate["history_reset"])
                       for candidate in selected if candidate["history_reset"] is not None]
        summaries.append({
            "representation": selected[0]["representation"], "h": selected[0]["h"],
            "n": len(rows),
            "mean_kl": _mean([_mean([s["kl"] for s in c["steps"]]) for c in selected]),
            "mean_final_hidden_rse": _mean([
                _mean([s["final_hidden_rse"] for s in c["steps"]]) for c in selected]),
            "mean_layer_hidden_rse": [_mean([c["layer_mean_rse"][layer]["hidden"]
                                              for c in selected])
                                      for layer in range(len(selected[0]["layer_mean_rse"]))],
            "reset_n": len(reset_pairs),
            "reset_normal_mean_kl": _mean([a["kl"] for a, b in reset_pairs])
            if reset_pairs else None,
            "reset_rebuilt_mean_kl": _mean([b["kl"] for a, b in reset_pairs])
            if reset_pairs else None,
            "reset_recovered_argmax_n": sum(b["candidate_argmax"] == b["bf16_argmax"]
                                             for a, b in reset_pairs),
            "first_divergence_indices": [c["first_divergence_index"] for c in selected],
        })
    return {"conditions": summaries,
            "historical_bf16_matches": sum(r["historical_bf16_score_length_match"] for r in rows),
            "max_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in rows),
            "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            "total_decode_steps": sum(r["bf16_new_tokens"] - 1 for r in rows),
            "device_names": sorted({r["device_name"] for r in rows})}


def _question_catalog(root: Path) -> dict[str, Any]:
    """从冻结数据核对八个传播病例的题面与标准答案。"""
    import pyarrow.parquet as pq

    plan = _read(root / "research/phase1/configs/trajectory_replay.json")
    questions = {}
    for task, dataset in (("gsm8k", "gsm8k"), ("hotpot", "hotpotqa")):
        manifest_path = f"research/manifests/data/{dataset}.source.json"
        manifest = _read(root / manifest_path)
        source_path = Path(manifest["destination"])
        source_path.resolve().relative_to(root.resolve())
        if _sha(source_path) != manifest["files"][0]["sha256"]:
            raise ValueError("冻结题目数据来源变化")
        data = pq.read_table(source_path).to_pylist()
        split = {item["id"]: item for item in
                 _read(root / f"research/phase1/splits/{task}.json")["v"]}
        for case in plan["cases"]:
            if case["task"] != task:
                continue
            item = split[case["id"]]
            source = data[item["row"]]
            if task == "gsm8k":
                digest = hashlib.sha256(source["question"].encode("utf-8")).hexdigest()
                if digest != item["question_sha256"]:
                    raise ValueError("冻结数学题身份不符")
            elif source["id"] != item["id"] or source["answer"] != item["answer"]:
                raise ValueError("冻结问答身份不符")
            questions[f"{task}/{case['id']}"] = {
                "task": task, "id": case["id"], "question": source["question"],
                "standard_answer": source["answer"], "dataset_source": manifest_path,
                "source_row": item["row"],
            }
    return questions


def run(models: tuple[str, ...] = ("qwen", "llama")) -> None:
    """指定模型通过结构、来源与原始数据审核后，原位更新唯一审核汇总。"""
    root = project_root()
    folder = root / "research/phase1/experiments/generation_error_sources/results"
    questions = _question_catalog(root)
    reviewed = {"schema": "kv-propagation-review-v1", "reviewed": True,
                "complete": set(models) == {"qwen", "llama"},
                "audit_code_sha256": _sha(Path(__file__)), "models": {}, "sources": [],
                "answer_extracts": {}, "questions": questions}
    validated = {}
    for model in models:
        path = folder / model / "trajectories.json"
        payload = _read(path)
        reuse_quality(payload, root)
        reviewed["answer_extracts"][model] = _audit_raw(payload, root, questions)
        reviewed["models"][model] = _summary(payload)
        validated[path] = payload
    for path, payload in validated.items():
        payload["reviewed"] = True
        payload["review_source"] = (
            "research/phase1/experiments/generation_error_sources/results/trajectory_review.json"
        )
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        reviewed["sources"].append({"path": path.relative_to(root).as_posix(),
                                    "sha256": _sha(path)})
    output = folder / "trajectory_review.json"
    output.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(reviewed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "llama"))
    args = parser.parse_args()
    run((args.model,) if args.model else ("qwen", "llama"))
