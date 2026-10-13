"""冻结四问联答的完整 V 输入，运行 BF16 并按请求审核长解码评分。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import socket
import sys
import time

import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.inputs import build_input, four_samples as samples
from phase1.models import MODELS, _eos_ids
from phase1.result_format import write_quality_result
from phase1.reuse import require_missing_result
from phase1.ruler_retrieval import file_hash, write_json
from phase1.runner import generate_bf16 as _generate
from phase1.runner import load_model
from phase1.score import answer_exact, token_f1
from phase1.score import score_request
from phase1.splits_build import load_hotpot, support_text, _marker_gap


CONFIG = "research/phase1/configs/long_decode.json"
MANIFEST = "research/phase1/splits/hotpot_four.json"
RUNTIME = "research/.runtime/four-question-decode"
RESULTS = "research/phase1/experiments/generation_error_sources/results"


def freeze_inputs() -> None:
    """在任何前向之前冻结全部 64 个 V 问题、16 个请求和两模型全文。"""
    root = project_root()
    config = json.loads((root / CONFIG).read_text("utf-8"))
    frozen = json.loads((root / config["identity_source"]).read_text("utf-8"))
    source = json.loads((root / "research/manifests/data/hotpotqa.source.json").read_text())
    path = Path(source["destination"]).resolve()
    if not path.is_relative_to(root) or file_hash(path) != source["files"][0]["sha256"]:
        raise RuntimeError("数据来源不在项目内或校验失败")
    rows = load_hotpot(path)
    indexed = {row["id"]: row for row in rows}
    selected = [indexed[item["id"]] for item in frozen["v"]]
    if len(selected) != 64 or len({row["id"] for row in selected}) != 64:
        raise RuntimeError("冻结 V 不是 64 道不同问题")
    if any(row["answer"] != old["answer"] for row, old in zip(selected, frozen["v"])):
        raise RuntimeError("数据源标准答案与冻结 V 不一致")
    banned = {item["id"] for split in ("v", "expand", "t") for item in frozen[split]}
    for offset, n in ((config["pilot_single_seed_offset"], config["pilot_single_n"]),
                      (config["pilot_four_seed_offset"], config["pilot_four_n"])):
        allowed = [row for row in rows if row["id"] not in banned
                   and len(row["support_titles"]) == 2]
        random.Random(config["master_seed"] + offset).shuffle(allowed)
        banned.update(row["id"] for row in allowed[:n])
    pool = [f"{title}: {sentence}" for row in sorted(rows, key=lambda r: r["id"])
            if row["id"] not in banned for title, sentences in sorted(row["docs"].items())
            for sentence in sentences if sentence.strip()]
    items = []
    for group in range(16):
        group_rows = selected[group * 4:group * 4 + 4]
        items.append({
            "id": f"hotpot-four-v-{group:02d}",
            "question_ids": [row["id"] for row in group_rows],
            "answers": [row["answer"] for row in group_rows],
            "support_titles": [row["support_titles"] for row in group_rows], "models": {},
        })
    for model_key, model_dir in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(root / model_dir)
        for group, item in enumerate(items):
            text, identity = build_input(
                tokenizer, model_key, selected[group * 4:group * 4 + 4], pool, group, config,
            )
            path = root / "research/.runtime/data/phase1-text/hotpot_four"
            path = path / model_key / f"{item['id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and file_hash(path) != identity["sha256"]:
                raise RuntimeError("既有四问输入与当前构造不同")
            path.write_text(text, "utf-8")
            identity["path"] = path.relative_to(root).as_posix()
            item["models"][model_key] = identity
        print("FROZEN", model_key, len(items), flush=True)
    manifest = {
        "protocol": config["protocol"], "config_sha256": file_hash(root / CONFIG),
        "identity_source_sha256": file_hash(root / config["identity_source"]),
        "n": 16, "n_questions": 64, "statistical_unit": "request", "items": items,
    }
    path = root / MANIFEST
    if path.exists() and json.loads(path.read_text("utf-8")) != manifest:
        raise RuntimeError("已有冻结清单不同")
    write_json(path, manifest)
    print("MANIFEST", file_hash(path), flush=True)


def run(model_key: str, commit: str) -> None:
    """逐请求运行统一 BF16 路径；完整预测留在服务器，可按冻结来源恢复。"""
    root = project_root()
    require_missing_result(root / RESULTS / model_key / "v_hotpot_four_bf16.json")
    config = json.loads((root / CONFIG).read_text("utf-8"))
    items = samples(root, model_key)
    publication_path = root / RUNTIME / "publication.json"
    publication = json.loads(publication_path.read_text("utf-8"))
    if commit != publication["base_commit"]:
        raise RuntimeError("执行提交与发布来源不一致")
    for record in publication["files"]:
        if file_hash(root / record["path"]) != record["sha256"]:
            raise RuntimeError(f"发布后文件变更：{record['path']}")
    identity = {"commit": commit, "manifest_sha256": file_hash(root / MANIFEST),
                "publication_sha256": file_hash(publication_path)}
    progress = root / RUNTIME / f"{model_key}.json"
    state = json.loads(progress.read_text("utf-8")) if progress.exists() else {}
    if state and any(state.get(key) != value for key, value in identity.items()):
        raise RuntimeError("既有恢复状态与发布或输入不同")
    rows = state.get("rows", [])
    if [row["id"] for row in rows] != [item["id"] for item in items[:len(rows)]]:
        raise RuntimeError("恢复状态不是冻结请求前缀")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    torch.set_num_threads(8)
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    device = {"node": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "slurm_job": os.environ.get("SLURM_JOB_ID")}
    for item in items[len(rows):]:
        source_path = root / item["path"]
        if file_hash(source_path) != item["sha256"]:
            raise RuntimeError("冻结正文校验失败")
        source = source_path.read_text("utf-8")
        ids = tokenizer.encode(source, add_special_tokens=False)
        if len(ids) != item["tokens"] or len(ids) + 1024 > config["context_limit"]:
            raise RuntimeError("输入 token 预算不符")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            produced, stopped = _generate(
                model, torch.tensor([ids], device="cuda"), eos, config["max_new_tokens"],
                consume_cache=True,
            )
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row = {
            "id": item["id"], "question_ids": item["question_ids"],
            "input_sha256": item["sha256"], "input_tokens": len(ids),
            "new_tokens": len(produced), "stop": stopped,
            "prediction": prediction, "produced_ids": produced,
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            **score_request(prediction, item["answers"]),
        }
        rows.append(row)
        write_json(progress, {**identity, "device": device, "rows": rows})
        print(model_key, len(rows), row["new_tokens"], row["stop"], row["f1"],
              "missing", row["missing_answers"], flush=True)
        torch.cuda.empty_cache()
    minimum = config["scoring"]["minimum_output_tokens"]
    eligible = [row for row in rows if row["new_tokens"] >= minimum]
    eligible_f1 = sum(row["f1"] for row in eligible) / len(eligible) if eligible else None
    payload = {
        "protocol": config["protocol"], **identity, "model": model_key, "config": "bf16-h0",
        "task": "v-hotpot_four", "n": len(rows), "n_questions": 4 * len(rows),
        "statistical_unit": "request", "questions_per_request": 4,
        "f1": sum(row["f1"] for row in rows) / len(rows),
        "em": 100 * sum(row["em"] for row in rows) / len(rows),
        "answer_em": sum(row["answer_em_percent"] for row in rows) / len(rows),
        "eligible_n": len(eligible), "eligible_ids": [row["id"] for row in eligible],
        "shorter_than_256_n": len(rows) - len(eligible), "eligible_f1": eligible_f1,
        "baseline_usable": (
            bool(eligible) and eligible_f1 >= config["scoring"]["baseline_f1_minimum"]
        ),
        "missing_answers": sum(row["missing_answers"] for row in rows), "device": device,
        "generation": "bf16_weights_sdpa_prefill_fp32_attention_decode_greedy",
        "raw_source": {"path": progress.relative_to(root).as_posix(),
                       "sha256": file_hash(progress)},
        "execution_source": {"path": publication_path.relative_to(root).as_posix(),
                             "sha256": file_hash(publication_path)},
        "rows": [{k: v for k, v in row.items() if k not in ("prediction", "produced_ids")}
                 for row in rows],
    }
    write_quality_result(root / RESULTS / model_key / "v_hotpot_four_bf16.json", payload)
    print("COMPLETE", model_key, payload["f1"], payload["eligible_n"], flush=True)


def audit_results() -> None:
    """重算完整预测的四个答案与请求均值，核对所有输入、证据间隔和生成轨迹。"""
    from phase1.review_results import _quality_plan
    root = project_root()
    config = json.loads((root / CONFIG).read_text("utf-8"))
    publication_path = root / RUNTIME / "publication.json"
    publication = json.loads(publication_path.read_text("utf-8"))
    for record in publication["files"]:
        if file_hash(root / record["path"]) != record["sha256"]:
            raise RuntimeError(f"审核时发布文件不同：{record['path']}")
    summaries, records = {}, {}
    for model_key, model_dir in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(root / model_dir)
        eos = _eos_ids(tokenizer, model_key == "llama")
        items = samples(root, model_key)
        progress = root / RUNTIME / f"{model_key}.json"
        raw = json.loads(progress.read_text("utf-8"))
        path = root / RESULTS / model_key / "v_hotpot_four_bf16.json"
        result = json.loads(path.read_text("utf-8"))
        if len(raw["rows"]) != len(result["rows"]) or len(result["rows"]) != 16:
            raise RuntimeError("未完成全部 16 个请求")
        if result["raw_source"]["sha256"] != file_hash(progress):
            raise RuntimeError("原始来源校验失败")
        identity = {
            "commit": publication["base_commit"], "manifest_sha256": file_hash(root / MANIFEST),
            "publication_sha256": file_hash(publication_path),
        }
        if any(raw[key] != value or result[key] != value for key, value in identity.items()):
            raise RuntimeError("执行来源与当前冻结不一致")
        if result["execution_source"]["sha256"] != file_hash(publication_path):
            raise RuntimeError("发布来源校验失败")
        for item, source, row in zip(items, raw["rows"], result["rows"]):
            if row["id"] != item["id"] or source["id"] != item["id"]:
                raise RuntimeError("请求顺序不匹配")
            text = (root / item["path"]).read_text("utf-8")
            if file_hash(root / item["path"]) != row["input_sha256"]:
                raise RuntimeError("输入正文校验失败")
            gaps = [_marker_gap(tokenizer, text, f"[[Q{i}-DOCUMENT-A]]",
                                f"[[Q{i}-DOCUMENT-B]]") for i in range(1, 5)]
            if gaps != item["support_gaps"] or min(gaps) < 8192:
                raise RuntimeError("证据间隔不匹配")
            decoded = tokenizer.decode(source["produced_ids"], skip_special_tokens=True)
            if decoded != source["prediction"]:
                raise RuntimeError("生成 token 与预测文本不匹配")
            if (len(source["produced_ids"]) != row["new_tokens"]
                    or row["new_tokens"] > config["max_new_tokens"]):
                raise RuntimeError("生成 token 数不匹配")
            stopped = "eos" if source["produced_ids"][-1] in eos else "limit"
            if stopped != row["stop"] or any(t in eos for t in source["produced_ids"][:-1]):
                raise RuntimeError("停止原因不匹配")
            scores = score_request(source["prediction"], item["answers"])
            if any(row[key] != value or source[key] != value for key, value in scores.items()):
                raise RuntimeError("原始预测重评分不一致")
            if row["question_ids"] != item["question_ids"]:
                raise RuntimeError("请求包含的问题不匹配")
        records[path] = result
        eligible = [row for row in result["rows"]
                    if row["new_tokens"] >= config["scoring"]["minimum_output_tokens"]]
        expected_f1 = sum(row["f1"] for row in eligible) / len(eligible) if eligible else None
        expected_usable = bool(eligible) and expected_f1 >= config["scoring"]["baseline_f1_minimum"]
        if (len(eligible) != result["eligible_n"] or expected_f1 != result["eligible_f1"]
                or expected_usable != result["baseline_usable"]
                or result["eligible_ids"] != [row["id"] for row in eligible]):
            raise RuntimeError("长度准入或合格子集汇总不一致")
        if (result["missing_answers"] != sum(row["missing_answers"] for row in result["rows"])
                or result["answer_em"] != sum(row["answer_em_percent"]
                                             for row in result["rows"]) / 16
                or result["n_questions"] != 64 or result["statistical_unit"] != "request"):
            raise RuntimeError("题级指标或统计单位不一致")
        summaries[model_key] = {
            "n_requests": 16, "n_questions": 64, "f1": result["f1"],
            "request_em": result["em"], "answer_em": result["answer_em"],
            "eligible_n": result["eligible_n"], "eligible_f1": result["eligible_f1"],
            "baseline_usable": result["baseline_usable"],
            "missing_answers": result["missing_answers"],
            "output_min": min(row["new_tokens"] for row in result["rows"]),
            "output_max": max(row["new_tokens"] for row in result["rows"]),
            "output_mean": sum(row["new_tokens"] for row in result["rows"]) / 16,
            "eos": sum(row["stop"] == "eos" for row in result["rows"]),
            "result": {"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)},
            "raw_source": result["raw_source"],
        }
    _quality_plan(root, records)
    write_json(root / RESULTS / "long_decode_baseline_review.json", {
        "status": "passed", "protocol": config["protocol"], "statistical_unit": "request",
        "manifest_sha256": file_hash(root / MANIFEST), "models": summaries,
        "boundary": "完整 V 四问联答 BF16 基线；量化候选尚未运行；历史单问结果保留",
    })
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1] in ("freeze", "audit"):
        {"freeze": freeze_inputs, "audit": audit_results}[sys.argv[1]]()
    else:
        run(sys.argv[1], sys.argv[2])
