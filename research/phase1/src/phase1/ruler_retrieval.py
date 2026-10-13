"""冻结 RULER 改编检索输入，并审核完整 BF16 重跑的原始预测。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

from transformers import AutoTokenizer

from kvstudy.artifacts import file_hash, write_json
from kvstudy.paths import project_root
from phase1.score import retrieval_number_exact, retrieval_numbers, retrieval_exact


CONFIG = "research/phase1/configs/retrieval.json"
MANIFEST = "research/phase1/splits/retrieval_ruler.json"
MODELS = {"qwen": "models/Qwen2.5-7B-Instruct", "llama": "models/Llama-3.1-8B-Instruct"}


def verify_publication(root: Path) -> None:
    """运行前核对实际发布的源码和冻结文件，拒绝使用变更后的代码恢复。"""
    publication = root / "research/.runtime/retrieval-ruler/publication.json"
    record = json.loads(publication.read_text("utf-8"))
    for source in record["files"]:
        if file_hash(root / source["path"]) != source["sha256"]:
            raise RuntimeError(f"发布后源码或输入清单发生变化：{source['path']}")


def build_input(
    tokenizer: Any, spec: dict[str, Any], model_key: str, config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """复用小样本构造；只计数 token，不调用模型前向。"""
    rng = random.Random(spec["seed"])
    count = 1 + config["distractors"][spec["interference"]]
    population = range(config["range_start"], config["range_stop"])
    keys = rng.sample(population, count)
    values = rng.sample(population, count)
    lines = [config["needle"].format(key=k, value=v) for k, v in zip(keys, values)]

    def build(n: int) -> str:
        background = [config["noise"]] * n
        pos = min(n - 1, round(n * spec["depth"]))
        background[pos] = lines[0]
        used = {pos}
        for i, line in enumerate(lines[1:]):
            slot = (pos + 3 + i * max(1, n // (count + 1))) % n
            while slot in used:
                slot = (slot + 1) % n
            used.add(slot)
            background[slot] = line
        user = config["template"].format(context="\n".join(background), query=keys[0])
        kwargs = {"date_string": config["date_string"]} if model_key == "llama" else {}
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, **kwargs,
        )
        return text + config["answer_prefix"].format(query=keys[0])

    lo, hi = config["background_search"]
    best = ""
    best_n = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        text = build(mid)
        if len(tokenizer.encode(text, add_special_tokens=False)) <= config["target_tokens"]:
            best, best_n = text, mid
            lo = mid + 1
        else:
            hi = mid - 1
    if not best:
        raise RuntimeError(f"无法满足输入预算：{spec['id']}")
    tokens = len(tokenizer.encode(best, add_special_tokens=False))
    offset = len(tokenizer.encode(best[:best.index(lines[0])], add_special_tokens=False))
    if not 31600 <= tokens <= config["target_tokens"]:
        raise RuntimeError(f"输入长度越界：{spec['id']}")
    if abs(offset / tokens - spec["depth"]) > 0.03:
        raise RuntimeError(f"证据深度越界：{spec['id']}")
    if any(best.count(line) != 1 for line in lines):
        raise RuntimeError(f"记录数不符合单目标构造：{spec['id']}")
    return best, {
        "key": str(keys[0]), "value": str(values[0]), "tokens": tokens,
        "sha256": hashlib.sha256(best.encode("utf-8")).hexdigest(),
        "actual_depth": offset / tokens, "background_sentences": best_n,
        "records": [{"key": str(k), "value": str(v)} for k, v in zip(keys, values)],
    }


def freeze_inputs() -> None:
    """冻结两模型全部 64 题，保留原 ID、种子与分层，且拒绝覆盖不同冻结。"""
    root = project_root()
    config = json.loads((root / CONFIG).read_text("utf-8"))
    original = json.loads((root / config["identity_source"]).read_text("utf-8"))
    specs = [{k: v for k, v in item.items() if k != "models"}
             for item in original if item["id"].startswith("synth-v-")]
    strata = Counter((s["depth"], s["interference"]) for s in specs)
    if len(specs) != 64 or len(strata) != 8 or set(strata.values()) != {8}:
        raise RuntimeError("原 V 划分不是四深度 × 两干扰 × 八题")
    items = [dict(spec, models={}) for spec in specs]
    pilot = root / "research/.runtime/scratch/ruler-comparison/inputs.json"
    pilot_rows = json.loads(pilot.read_text("utf-8")) if pilot.exists() else []
    checked = []
    for model_key, model_dir in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(root / model_dir)
        for item in items:
            text, identity = build_input(tokenizer, item, model_key, config)
            path = root / "research/.runtime/data/phase1-text/retrieval_ruler"
            path = path / model_key / f"{item['id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and file_hash(path) != identity["sha256"]:
                raise RuntimeError(f"既有新输入与构造不一致：{path}")
            path.write_text(text, "utf-8")
            identity["path"] = path.relative_to(root).as_posix()
            item["models"][model_key] = identity
            if model_key == "llama":
                matches = [p for p in pilot_rows if p["id"] == item["id"]
                           and p["condition"] == "ruler_construction"]
                for row in matches:
                    if row["input_sha256"] != identity["sha256"]:
                        raise RuntimeError(f"没有逐字节复现既有小样本：{item['id']}")
                    checked.append(item["id"])
        print("FROZEN", model_key, len(items), flush=True)
    for item in items:
        if item["models"]["qwen"]["records"] != item["models"]["llama"]["records"]:
            raise RuntimeError("两模型使用了不同键值")
    manifest = {
        "protocol": config["protocol"], "config_sha256": file_hash(root / CONFIG),
        "identity_source_sha256": file_hash(root / config["identity_source"]),
        "pilot_byte_identity_checked": checked, "n": len(items), "items": items,
    }
    path = root / MANIFEST
    if path.exists() and json.loads(path.read_text("utf-8")) != manifest:
        raise RuntimeError("冻结清单已存在且内容不同")
    write_json(path, manifest)
    print("MANIFEST", file_hash(path), flush=True)


def audit_results() -> None:
    """独立从完整原始预测重算评分，核对冻结输入及正式逐题结果。"""
    root = project_root()
    config = json.loads((root / CONFIG).read_text("utf-8"))
    frozen = json.loads((root / MANIFEST).read_text("utf-8"))
    verify_publication(root)
    if frozen["config_sha256"] != file_hash(root / CONFIG):
        raise RuntimeError("冻结后配置发生变化")
    summaries = {}
    for model_key in MODELS:
        raw_path = root / "research/.runtime/retrieval-ruler" / f"{model_key}.json"
        result_path = root / "research/phase1/experiments/generation_error_sources/results"
        result_path = result_path / model_key / "v_retrieval_ruler_bf16.json"
        raw = json.loads(raw_path.read_text("utf-8"))
        result = json.loads(result_path.read_text("utf-8"))
        if len(raw["rows"]) != 64 or result["n"] != 64:
            raise RuntimeError("正式结果未跑完整 64 题")
        if result["protocol"] != config["protocol"]:
            raise RuntimeError("结果协议不匹配")
        correct = strict = 0
        strata: dict[str, dict[str, int]] = {}
        for item, source, row in zip(frozen["items"], raw["rows"], result["rows"]):
            identity = item["models"][model_key]
            if source["id"] != item["id"] or row["id"] != item["id"]:
                raise RuntimeError("结果顺序不匹配")
            if file_hash(root / identity["path"]) != identity["sha256"]:
                raise RuntimeError("实际输入全文与冻结不匹配")
            if row["input_sha256"] != identity["sha256"]:
                raise RuntimeError("生成输入与冻结不匹配")
            values = sorted(set(re.findall(config["scoring"]["pattern"], source["prediction"])))
            passed = values == [identity["value"]]
            if passed != retrieval_number_exact(source["prediction"], identity["value"]):
                raise RuntimeError("独立抽取与评分实现不一致")
            if values != row["extracted_values"] or passed != row["correct"]:
                raise RuntimeError("原始预测重评分不一致")
            if retrieval_numbers(source["prediction"]) != values:
                raise RuntimeError("抽取值不一致")
            formatted = retrieval_exact(source["prediction"], identity["value"])
            if formatted != row["strict_correct"]:
                raise RuntimeError("严格格式评分不一致")
            if row["input_tokens"] != identity["tokens"] or row["new_tokens"] > 128:
                raise RuntimeError("token 预算不匹配")
            if row["new_tokens"] != len(source["produced_ids"]):
                raise RuntimeError("生成 token 记录不匹配")
            correct += passed
            strict += formatted
            key = f"depth={item['depth']};interference={item['interference']}"
            cell = strata.setdefault(key, {"n": 0, "correct": 0})
            cell["n"] += 1
            cell["correct"] += passed
        if result["correct"] != correct or result["em"] != 100 * correct / 64:
            raise RuntimeError("结果汇总不一致")
        if result["raw_source"]["sha256"] != file_hash(raw_path):
            raise RuntimeError("原始预测来源校验失败")
        summaries[model_key] = {
            "n": 64, "correct": correct, "em": 100 * correct / 64,
            "strict_correct": strict, "strata": strata,
            "input_tokens_min": min(row["input_tokens"] for row in result["rows"]),
            "input_tokens_max": max(row["input_tokens"] for row in result["rows"]),
            "baseline_passed": 100 * correct / 64 >= config["scoring"]["baseline_em_minimum"],
            "eos": sum(row["stop"] == "eos" for row in result["rows"]),
            "result_path": result_path.relative_to(root).as_posix(),
            "result_sha256": file_hash(result_path), "raw_source": result["raw_source"],
        }
    out = root / "research/phase1/experiments/generation_error_sources/results"
    write_json(out / "retrieval_baseline_review.json", {
        "status": "passed", "protocol": config["protocol"],
        "manifest_sha256": file_hash(root / MANIFEST), "models": summaries,
        "boundary": "新构造的完整 BF16 基线；未运行同输入量化候选，不重算历史结论",
    })
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    {"freeze": freeze_inputs, "audit": audit_results}[sys.argv[1]]()
