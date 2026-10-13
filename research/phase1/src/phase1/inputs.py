"""第一阶段共用的冻结输入、分词身份和四问请求构造。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from phase1.ruler_retrieval import CONFIG, MANIFEST, file_hash
from phase1.score import retrieval_exact, retrieval_number_exact, retrieval_numbers, score_request
from phase1.splits_build import _marker_gap, support_text


FOUR_CONFIG = "research/phase1/configs/long_decode.json"
FOUR_MANIFEST = "research/phase1/splits/hotpot_four.json"


def _samples(root: Path, model_key: str, task: str, split: str) -> list[dict[str, Any]]:
    """按冻结顺序返回题目、全文位置和校验值。"""
    if task == "retrieval_ruler":
        frozen = json.loads((root / MANIFEST).read_text("utf-8"))
        if split != "v" or frozen["n"] != 64:
            raise RuntimeError("RULER 改编检索仅冻结完整 V 64 题")
        if file_hash(root / CONFIG) != frozen["config_sha256"]:
            raise RuntimeError("冻结后配置发生变化")
        return [
            {"id": item["id"], "answer": item["models"][model_key]["value"],
             "tokens": item["models"][model_key]["tokens"],
             "sha256": item["models"][model_key]["sha256"],
             "path": root / item["models"][model_key]["path"]}
            for item in frozen["items"]
        ]
    if task == "retrieval":
        items = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
        prefix = f"synth-{split}-"
        return [
            {
                "id": item["id"],
                "answer": item["models"][model_key]["value"],
                "tokens": item["models"][model_key]["tokens"],
                "sha256": item["models"][model_key]["sha256"],
                "path": root / "research/.runtime/data/phase1-text/retrieval"
                / model_key / f"{item['id']}.txt",
            }
            for item in items if item["id"].startswith(prefix)
        ]
    if task in ("hotpot", "hotpot_long"):
        frozen = json.loads((root / "research/phase1/splits/hotpot.json").read_text("utf-8"))
        if task == "hotpot_long":
            if split != "v":
                raise ValueError("长解码诊断只使用 V")
            indexed = {item["id"]: item for item in frozen["v"]}
            items = [indexed[item_id] for item_id in frozen["long_decode"]]
            variant = "long"
        else:
            items = frozen[split]
            variant = "main"
        return [
            {
                "id": item["id"],
                "answer": item["answer"],
                "tokens": item["models"][model_key][f"{variant}_tokens"],
                "sha256": item["models"][model_key][f"{variant}_sha256"],
                "path": root / "research/.runtime/data/phase1-text/hotpot"
                / model_key / split / f"{item['id']}.{variant}.txt",
            }
            for item in items
        ]
    raise ValueError(f"未知任务：{task}")


def build_input(
    tokenizer: Any, model_key: str, rows: list[dict[str, Any]], pool: list[str],
    group: int, config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """复用已验证的材料顺序和末尾问法，四个证据对各自至少相隔 8K token。"""
    first, second, own = [], [], []
    for index, row in enumerate(rows, 1):
        titles = row["support_titles"]
        if len(titles) != 2 or any(title not in row["docs"] for title in titles):
            raise RuntimeError(f"支持材料不完整：{row['id']}")
        first.append(f"[[Q{index}-DOCUMENT-A]]\n" + support_text(row, titles[0]))
        second.append(f"[[Q{index}-DOCUMENT-B]]\n" + support_text(row, titles[1]))
        own.extend(f"{title}: {' '.join(sentences)}" for title, sentences in row["docs"].items()
                   if title not in titles)
    start = (config["master_seed"] + group * config["pool_start_stride"]) % len(pool)
    questions = "\n".join(f"Question {i}: {row['question']}" for i, row in enumerate(rows, 1))

    def build(gap: int, tail: int) -> tuple[str, list[int]]:
        gap_lines = [pool[(start + i) % len(pool)] for i in range(gap)]
        tail_lines = [pool[(start + gap + i) % len(pool)] for i in range(tail)]
        documents = "\n\n".join([*first, *own, *gap_lines, *second, *tail_lines])
        user = documents + "\n\n" + questions + "\n\n" + config["prompt"]
        kwargs = {"date_string": config["date_string"]} if model_key == "llama" else {}
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, **kwargs,
        )
        return text, tokenizer.encode(text, add_special_tokens=False)

    def gaps(text: str) -> list[int]:
        return [_marker_gap(tokenizer, text, f"[[Q{i}-DOCUMENT-A]]", f"[[Q{i}-DOCUMENT-B]]")
                for i in range(1, 5)]

    lo, hi = config["gap_search"]
    gap = None
    while lo <= hi:
        mid = (lo + hi) // 2
        text, _ = build(mid, 0)
        if min(gaps(text)) >= config["minimum_evidence_gap_tokens"]:
            gap, hi = mid, mid - 1
        else:
            lo = mid + 1
    if gap is None:
        raise RuntimeError(f"证据间隔装箱失败：请求 {group}")
    lo, hi = config["tail_search"]
    tail = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _, ids = build(gap, mid)
        if len(ids) <= config["target_input_tokens"]:
            tail, lo = mid, mid + 1
        else:
            hi = mid - 1
    text, ids = build(gap, tail)
    actual_gaps = gaps(text)
    if (min(actual_gaps) < config["minimum_evidence_gap_tokens"]
            or not config["minimum_input_tokens"] <= len(ids) <= config["target_input_tokens"]
            or len(ids) + config["max_new_tokens"] > config["context_limit"]):
        raise RuntimeError(f"输入长度或证据间隔不符合冻结：请求 {group}")
    return text, {
        "tokens": len(ids), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "support_gaps": actual_gaps, "gap_fillers": gap, "tail_fillers": tail,
        "pool_start": start,
    }


def four_samples(root: Path, model_key: str) -> list[dict[str, Any]]:
    """读取冻结请求顺序与答案，不访问历史单问长解码输入。"""
    frozen = json.loads((root / FOUR_MANIFEST).read_text("utf-8"))
    if frozen["config_sha256"] != file_hash(root / FOUR_CONFIG):
        raise RuntimeError("冻结后四问配置发生变化")
    return [{"id": item["id"], "answers": item["answers"],
             "question_ids": item["question_ids"], **item["models"][model_key]}
            for item in frozen["items"]]


def _inputs(root: Path, model_key: str, task: str, split: str) -> list[dict[str, Any]]:
    """按冻结顺序加载题目索引；GSM8K 正文延迟到运行时构造。"""
    if task == "hotpot_four":
        if split != "v":
            raise ValueError("四问联答只使用 V")
        return four_samples(root, model_key)
    if task != "gsm8k":
        return _samples(root, model_key, task, split)
    if split not in ("c", "v", "expand"):
        raise ValueError(f"不支持的数学题划分：{split}")
    frozen = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    manifest = json.loads((root / "research/manifests/data/gsm8k.source.json").read_text("utf-8"))
    import pyarrow.parquet as pq

    table = pq.read_table(root / manifest["destination"])
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    return [
        {
            "id": item["id"], "question": questions[item["row"]],
            "answer": answers[item["row"]], "question_sha256": item["question_sha256"],
        }
        for item in frozen[split]
    ]


def _ids_and_source(
    root: Path, tokenizer: Any, model_key: str, task: str, item: dict[str, Any]
) -> tuple[list[int], str]:
    """返回完整输入 token 与全文 SHA256，并核对冻结数据。"""
    if task == "gsm8k":
        question = item["question"]
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != item["question_sha256"]:
            raise RuntimeError(f"数学题源数据校验失败：{item['id']}")
        templates = json.loads((root / "research/phase1/splits/templates.json").read_text("utf-8"))
        user = templates["gsm8k"].format(question=question)
        kwargs = {"date_string": "26 Jul 2024"} if model_key == "llama" else {}
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, **kwargs,
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=True,
            add_generation_prompt=True, **kwargs,
        )
    else:
        source = (root / item["path"]).read_text("utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != item["sha256"]:
            raise RuntimeError(f"长任务输入校验失败：{item['id']}")
        ids = tokenizer(source, add_special_tokens=False)["input_ids"]
        if len(ids) != item["tokens"]:
            raise RuntimeError(f"长任务输入长度不符：{item['id']}")
    return list(ids), hashlib.sha256(source.encode("utf-8")).hexdigest()


def _task_protocol(root: Path, task: str) -> str:
    """新输入使用各自冻结协议；既有任务保持原协议名。"""
    if task == "retrieval_ruler":
        config = root / "research/phase1/configs/retrieval.json"
    elif task == "hotpot_four":
        config = root / "research/phase1/configs/long_decode.json"
    else:
        return "phase1-full-key-retrieval"
    return json.loads(config.read_text(encoding="utf-8"))["protocol"]


def _score_new_input(
    task: str, prediction: str, item: dict[str, Any], invalid: int,
) -> dict[str, Any]:
    """新检索按唯一七位数字评分；四问先平均四题。无效组将该样本记为 0 分。"""
    if task == "retrieval_ruler":
        return {
            "correct": not invalid and retrieval_number_exact(prediction, item["answer"]),
            "strict_correct": not invalid and retrieval_exact(prediction, item["answer"]),
            "extracted_values": retrieval_numbers(prediction),
            "target": item["answer"],
        }
    scored = score_request(prediction, item["answers"])
    if invalid:
        scored.update(
            f1=0.0, em=False, answer_em_percent=0.0,
            answer_f1=[0.0, 0.0, 0.0, 0.0], answer_em=[False, False, False, False],
        )
    return scored
