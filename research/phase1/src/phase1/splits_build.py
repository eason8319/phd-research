"""在 M0 之后一次生成并冻结 splits/。只使用 tokenizer 计数，不跑模型前向。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from kvstudy.paths import project_root

MASTER_SEED = 20260923
DATE_STRING = "26 Jul 2024"
TARGET_32K = 31 * 1024
GSM8K_C = 64
GSM8K_V = 64
GSM8K_EXPAND = 192
GSM8K_T = 64
HOTPOT_V = 64
HOTPOT_EXPAND = 192
HOTPOT_T = 64
LONG_DECODE = 8
GAP_TOKENS = 8192

GSM8K_TEMPLATE = (
    "Solve the math problem. Show your work, and on the last line write the "
    "final numeric answer in the form #### <number>.\n\n{question}"
)
RETRIEVAL_TEMPLATE = (
    "Find the value stored for the given key. Reply with the value only, "
    "with no extra words.\n\n{context}\n\nKey: {key}\nValue:"
)
HOTPOT_MAIN = (
    "Read the documents and answer the question. Give a short explanation, "
    "then the final answer on its own line in the form: Final answer: <answer>\n\n"
    "{documents}\n\nQuestion: {question}"
)
HOTPOT_LONG = (
    "Read the documents and answer the question. Reason step by step, then give "
    "the final answer on its own line in the form: Final answer: <answer>\n\n"
    "{documents}\n\nQuestion: {question}"
)
FILLER_SENTENCE = "Background note {i} says the sky is clear and the street is quiet today."
NEEDLE_SENTENCE = "The register {key} stores {value}."


def stream_seed(stream: int, index: int) -> int:
    """由主种子、流编号和序号得到互不重叠的 31 位种子。"""
    return (MASTER_SEED * 1_000_003 + stream * 10_007 + index) % (2**31 - 1)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tokenizer(root: Path, model_dir: str) -> Any:
    return AutoTokenizer.from_pretrained(root / model_dir)


def render(tokenizer: Any, user: str, llama: bool) -> tuple[str, list[int]]:
    """渲染聊天模板。Llama 固定 date_string，不另写系统提示。"""
    kwargs: dict[str, str] = {"date_string": DATE_STRING} if llama else {}
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    return text, list(ids)


def parquet_path(root: Path, manifest_name: str) -> Path:
    meta = json.loads((root / "research/manifests/data" / manifest_name).read_text(encoding="utf-8"))
    return Path(meta["destination"])


def load_gsm8k(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["question", "answer"])
    rows = []
    for index, (question, answer) in enumerate(zip(table.column(0).to_pylist(), table.column(1).to_pylist())):
        rows.append({"row": index, "question": question, "answer": answer})
    return rows


def load_hotpot(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    rows = []
    ids = table.column("id").to_pylist()
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    facts = table.column("supporting_facts").to_pylist()
    contexts = table.column("context").to_pylist()
    for index, item_id in enumerate(ids):
        titles = list(dict.fromkeys(facts[index]["title"]))
        context_titles = contexts[index]["title"]
        sentences = contexts[index]["sentences"]
        docs = {title: sentences[pos] for pos, title in enumerate(context_titles)}
        rows.append(
            {
                "row": index,
                "id": item_id,
                "question": questions[index],
                "answer": answers[index],
                "support_titles": titles,
                "docs": docs,
            }
        )
    return rows


def allocate_gsm8k(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(stream_seed(1, 0))
    order = rng.permutation(len(rows))
    needed = GSM8K_C + GSM8K_V + GSM8K_EXPAND + GSM8K_T
    picked = [rows[int(order[i])] for i in range(needed)]
    c_rows = picked[:GSM8K_C]
    v_rows = picked[GSM8K_C : GSM8K_C + GSM8K_V]
    expand = picked[GSM8K_C + GSM8K_V : GSM8K_C + GSM8K_V + GSM8K_EXPAND]
    held = picked[needed - GSM8K_T :]
    return {"c": assign_folds(c_rows), "v": v_rows, "expand": expand, "t": held}


def assign_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按题面长度四分位分层，再把每一层对半放进两折。"""
    ranked = sorted(rows, key=lambda row: (len(row["question"]), row["row"]))
    rng = np.random.default_rng(stream_seed(6, 0))
    folded: list[dict[str, Any]] = []
    width = len(ranked) // 4
    for quartile in range(4):
        chunk = ranked[quartile * width : (quartile + 1) * width]
        order = rng.permutation(len(chunk))
        half = len(chunk) // 2
        for position, source in enumerate(order):
            item = dict(chunk[int(source)])
            item["fold"] = 0 if position < half else 1
            folded.append(item)
    return folded


def retrieval_spec(kind: str, index: int, length: int, depth: float, level: str, fold: int | None) -> dict[str, Any]:
    return {
        "id": f"synth-{kind}-{index:04d}",
        "seed": stream_seed({"c": 3, "v": 4, "t": 5, "expand": 8}[kind], index),
        "target_tokens": length,
        "depth": depth,
        "interference": level,
        "fold": fold,
    }


def retrieval_plan() -> list[dict[str, Any]]:
    depths = (0.10, 0.35, 0.65, 0.90)
    levels = ("low", "high")
    items: list[dict[str, Any]] = []
    number = 0
    for length in (2048, 4096):
        for depth in depths:
            for level in levels:
                for fold in (0, 1):
                    items.append(retrieval_spec("c", number, length, depth, level, fold))
                    number += 1
    number = 0
    for depth in depths:
        for level in levels:
            for _ in range(8):
                items.append(retrieval_spec("v", number, TARGET_32K, depth, level, None))
                number += 1
    for index in range(192):
        depth = depths[index % 4]
        level = levels[(index // 4) % 2]
        items.append(retrieval_spec("expand", index, TARGET_32K, depth, level, None))
    for index in range(64):
        depth = depths[index % 4]
        level = levels[(index // 4) % 2]
        items.append(retrieval_spec("t", index, TARGET_32K, depth, level, None))
    return items


def build_retrieval(tokenizer: Any, spec: dict[str, Any], llama: bool) -> dict[str, Any]:
    """用填料句数二分，使聊天模板后的长度接近目标。"""
    key = f"key-{spec['seed']:08x}"
    value = f"val-{spec['seed']:08x}"
    distractors = 1 if spec["interference"] == "low" else 8
    others = [f"val-{spec['seed']:08x}-d{i}" for i in range(distractors)]
    low, high = 1, 4000
    best_n = 1
    best_gap = 10**9
    while low <= high:
        mid = (low + high) // 2
        built = _retrieval_text(tokenizer, spec, key, value, others, mid, spec["depth"], llama)
        gap = abs(built["tokens"] - spec["target_tokens"])
        if built["tokens"] <= spec["target_tokens"] and gap < best_gap:
            best_gap = gap
            best_n = mid
        if built["tokens"] < spec["target_tokens"]:
            low = mid + 1
        else:
            high = mid - 1
    fraction = float(spec["depth"])
    best = _retrieval_text(tokenizer, spec, key, value, others, best_n, fraction, llama)
    for _ in range(4):
        delta = float(spec["depth"]) - float(best["actual_depth"])
        if best["actual_depth"] < 0 or abs(delta) <= 0.03:
            break
        fraction = min(0.98, max(0.02, fraction + delta))
        best = _retrieval_text(tokenizer, spec, key, value, others, best_n, fraction, llama)
    return best


def _retrieval_text(
    tokenizer: Any,
    spec: dict[str, Any],
    key: str,
    value: str,
    others: list[str],
    sentences: int,
    depth: float,
    llama: bool,
) -> dict[str, Any]:
    lines = [FILLER_SENTENCE.format(i=i) for i in range(max(sentences, 1))]
    needle_at = min(len(lines) - 1, max(0, round(depth * (len(lines) - 1))))
    lines[needle_at] = NEEDLE_SENTENCE.format(key=key, value=value)
    used = {needle_at}
    for index, other in enumerate(others):
        slot = (needle_at + 3 + index * max(1, sentences // (len(others) + 2))) % sentences
        if slot in used:
            slot = (slot + 1) % sentences
        lines[slot] = NEEDLE_SENTENCE.format(key=key, value=other)
        used.add(slot)
    user = RETRIEVAL_TEMPLATE.format(context="\n".join(lines), key=key)
    text, ids = render(tokenizer, user, llama)
    needle = NEEDLE_SENTENCE.format(key=key, value=value)
    depth = _span_fraction(tokenizer, text, needle)
    return {
        "text": text,
        "tokens": len(ids),
        "actual_depth": depth,
        "key": key,
        "value": value,
        "sha256": sha256_text(text),
    }


def _span_fraction(tokenizer: Any, text: str, needle: str) -> float:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    start = text.find(needle)
    if start < 0 or not offsets:
        return -1.0
    token_at = 0
    for index, (left, right) in enumerate(offsets):
        if left <= start < right or (right == left and left == start):
            token_at = index
            break
    return token_at / len(offsets)


def support_text(row: dict[str, Any], title: str) -> str:
    return title + "\n" + " ".join(row["docs"][title])


def filler_pool(rows: list[dict[str, Any]], held_ids: set[str]) -> list[str]:
    """按来源 id 和标题排序。T 的支持段落不进入填料。"""
    pool: list[tuple[str, str, int, str]] = []
    for row in rows:
        supports = set(row["support_titles"])
        for title, sentences in row["docs"].items():
            if row["id"] in held_ids and title in supports:
                continue
            for index, sentence in enumerate(sentences):
                pool.append((row["id"], title, index, f"{title}: {sentence}"))
    pool.sort()
    return [item[3] for item in pool]


def usable_hotpot(row: dict[str, Any]) -> bool:
    titles = row["support_titles"]
    return len(titles) == 2 and all(title in row["docs"] for title in titles)


def pack_hotpot(
    tokenizer: Any,
    row: dict[str, Any],
    pool: list[str],
    cursor: int,
    llama: bool,
) -> tuple[dict[str, Any], int]:
    """把两份支持文档隔开至少 8192 token，并把较长问法压进 30K–31K。"""
    supports = [support_text(row, title) for title in row["support_titles"]]
    own = [
        f"{title}: {' '.join(sentences)}"
        for title, sentences in row["docs"].items()
        if title not in row["support_titles"]
    ]
    gap: list[str] = list(own)
    tail: list[str] = []

    def consume(count: int) -> list[str]:
        nonlocal cursor
        picked = []
        for _ in range(count):
            picked.append(pool[cursor % len(pool)])
            cursor += 1
        return picked

    def assemble(gap_lines: list[str], tail_lines: list[str], template: str) -> tuple[str, list[int]]:
        documents = "\n\n".join([supports[0], *gap_lines, supports[1], *tail_lines])
        user = template.format(documents=documents, question=row["question"])
        return render(tokenizer, user, llama)

    # 先放够间隔，再补尾部。每轮用整段 tokenize 校正，避免逐句重编码。
    for _ in range(6):
        long_text, long_ids = assemble(gap, tail, HOTPOT_LONG)
        gap_tokens = _gap_tokens(tokenizer, long_text, supports[0], supports[1])
        if gap_tokens < GAP_TOKENS:
            gap.extend(consume(40))
            continue
        if len(long_ids) > TARGET_32K and len(tail) > 20:
            del tail[-20:]
            continue
        if len(long_ids) > TARGET_32K and len(gap) > len(own) + 40:
            del gap[-20:]
            continue
        if len(long_ids) < 30 * 1024:
            tail.extend(consume(40))
            continue
        short_text, short_ids = assemble(gap, tail, HOTPOT_MAIN)
        return (
            {
                "main_text": short_text,
                "main_tokens": len(short_ids),
                "main_sha256": sha256_text(short_text),
                "long_text": long_text,
                "long_tokens": len(long_ids),
                "long_sha256": sha256_text(long_text),
                "support_titles": row["support_titles"],
                "gap_tokens": gap_tokens,
                "support_fraction": [
                    _span_fraction(tokenizer, long_text, supports[0][:80]),
                    _span_fraction(tokenizer, long_text, supports[1][:80]),
                ],
            },
            cursor,
        )
    raise RuntimeError(f"无法把 {row['id']} 装进 32K 窗口")


def _gap_tokens(tokenizer: Any, text: str, first: str, second: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    first_at = text.find(first[:80])
    second_at = text.find(second[:80])
    if first_at < 0 or second_at < 0:
        return 0

    def locate(char_pos: int) -> int:
        for index, (left, right) in enumerate(offsets):
            if left <= char_pos < max(right, left + 1):
                return index
        return 0

    return locate(second_at) - locate(first_at)


def main() -> None:
    root = project_root()
    out = root / "research/phase1/splits"
    if (out / "seeds.json").exists():
        raise SystemExit("splits/seeds.json 已存在，不再改写")
    out.mkdir(parents=True, exist_ok=True)
    text_root = root / "research/.runtime/data/phase1-text"
    text_root.mkdir(parents=True, exist_ok=True)
    qwen = load_tokenizer(root, "models/Qwen2.5-7B-Instruct")
    llama = load_tokenizer(root, "models/Llama-3.1-8B-Instruct")
    prefix_user = "PREFIX-CHECK"
    prefixes = {}
    for name, tokenizer, is_llama in (("qwen", qwen, False), ("llama", llama, True)):
        text, _ = render(tokenizer, prefix_user, is_llama)
        prefixes[name] = text.split(prefix_user)[0]
    llama_prefix = prefixes["llama"]
    has_cutoff = "Cutting Knowledge Date:" in llama_prefix
    has_today = "Today Date: 26 Jul 2024" in llama_prefix
    if not has_cutoff or not has_today:
        raise SystemExit("Llama 模板没有同时注入知识截止日期和固定日期")
    if "Qwen" not in prefixes["qwen"]:
        raise SystemExit("Qwen 模板没有注入默认身份句")
    gsm = allocate_gsm8k(load_gsm8k(parquet_path(root, "gsm8k.source.json")))
    hotpot_rows = [
        row for row in load_hotpot(parquet_path(root, "hotpotqa.source.json")) if usable_hotpot(row)
    ]
    rng = np.random.default_rng(stream_seed(2, 0))
    order = rng.permutation(len(hotpot_rows))
    chosen = [hotpot_rows[int(order[i])] for i in range(HOTPOT_V + HOTPOT_EXPAND + HOTPOT_T)]
    hotpot_v = chosen[:HOTPOT_V]
    hotpot_expand = chosen[HOTPOT_V : HOTPOT_V + HOTPOT_EXPAND]
    hotpot_t = chosen[HOTPOT_V + HOTPOT_EXPAND :]
    held_ids = {row["id"] for row in hotpot_t}
    pool = filler_pool(load_hotpot(parquet_path(root, "hotpotqa.source.json")), held_ids)
    retrieval_public = []
    for spec in retrieval_plan():
        record = {key: spec[key] for key in ("id", "seed", "target_tokens", "depth", "interference", "fold")}
        record["models"] = {}
        for name, tokenizer, is_llama in (("qwen", qwen, False), ("llama", llama, True)):
            built = build_retrieval(tokenizer, spec, is_llama)
            _store_text(text_root / "retrieval" / name / f"{spec['id']}.txt", built["text"])
            record["models"][name] = {key: built[key] for key in ("tokens", "actual_depth", "key", "value", "sha256")}
        retrieval_public.append(record)
        if len(retrieval_public) % 20 == 0:
            print(f"retrieval {len(retrieval_public)}", flush=True)
    cursor = 0
    hotpot_public = {"v": [], "expand": [], "t": [], "long_decode": []}
    for split_name, split_rows in (("v", hotpot_v), ("expand", hotpot_expand), ("t", hotpot_t)):
        for row in split_rows:
            item = {"id": row["id"], "row": row["row"], "answer": row["answer"], "models": {}}
            for name, tokenizer, is_llama in (("qwen", qwen, False), ("llama", llama, True)):
                packed, cursor = pack_hotpot(tokenizer, row, pool, cursor, is_llama)
                _store_text(text_root / "hotpot" / name / split_name / f"{row['id']}.main.txt", packed["main_text"])
                _store_text(text_root / "hotpot" / name / split_name / f"{row['id']}.long.txt", packed["long_text"])
                item["models"][name] = {key: packed[key] for key in ("main_tokens", "main_sha256", "long_tokens", "long_sha256", "support_titles", "gap_tokens", "support_fraction")}
            hotpot_public[split_name].append(item)
            print(f"hotpot {split_name} {row['id']}", flush=True)
    hotpot_public["long_decode"] = [item["id"] for item in hotpot_public["v"][:LONG_DECODE]]
    write_json(
        out / "seeds.json",
        {
            "master": MASTER_SEED,
            "bootstrap": MASTER_SEED,
            "bootstrap_draws": 10000,
            "e3_random": [202609231, 202609232, 202609233],
        },
    )
    write_json(
        out / "templates.json",
        {
            "gsm8k": GSM8K_TEMPLATE,
            "retrieval": RETRIEVAL_TEMPLATE,
            "hotpot_main": HOTPOT_MAIN,
            "hotpot_long": HOTPOT_LONG,
            "needle": NEEDLE_SENTENCE,
            "filler": FILLER_SENTENCE,
            "prefixes": prefixes,
            "date_string": DATE_STRING,
        },
    )
    write_json(
        out / "gsm8k.json",
        {
            "c": [_public_gsm(row) for row in gsm["c"]],
            "v": [_public_gsm(row) for row in gsm["v"]],
            "expand": [_public_gsm(row) for row in gsm["expand"]],
            "t": [_public_gsm(row) for row in gsm["t"]],
        },
    )
    write_json(out / "retrieval.json", retrieval_public)
    write_json(out / "hotpot.json", hotpot_public)
    write_json(out / "trajectories.json", _trajectories(gsm, retrieval_public, hotpot_public))
    write_json(out / "checks.json", _checks(gsm, retrieval_public, hotpot_public))
    print("splits frozen", flush=True)


def _public_gsm(row: dict[str, Any]) -> dict[str, Any]:
    item = {"id": f"gsm8k-train-{row['row']}", "row": row["row"], "question_sha256": sha256_text(row["question"])}
    if "fold" in row:
        item["fold"] = row["fold"]
    return item


def _store_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _trajectories(gsm: dict[str, list[dict[str, Any]]], retrieval: list[dict[str, Any]], hotpot: dict[str, Any]) -> dict[str, Any]:
    math_ids = [f"gsm8k-train-{row['row']}" for row in gsm["c"]]
    by_fold: dict[int, list[str]] = {0: [], 1: []}
    for row, item_id in zip(gsm["c"], math_ids):
        by_fold[int(row["fold"])].append(item_id)
    c_retrieval = [item["id"] for item in retrieval if item["id"].startswith("synth-c-")]
    c_by_fold: dict[int, list[str]] = {0: [], 1: []}
    for item in retrieval:
        if item["fold"] in (0, 1) and item["id"].startswith("synth-c-"):
            c_by_fold[int(item["fold"])].append(item["id"])
    v_retrieval = [item["id"] for item in retrieval if item["id"].startswith("synth-v-")]
    return {
        "c_math": by_fold[0][:4] + by_fold[1][:4],
        "c_retrieval": c_by_fold[0][:4] + c_by_fold[1][:4],
        "v_math": [f"gsm8k-train-{row['row']}" for row in gsm["v"][:4]],
        "v_retrieval_32k": v_retrieval[:8],
        "v_hotpot": [item["id"] for item in hotpot["v"][LONG_DECODE : LONG_DECODE + 4]],
        "counts": {"c_math": 8, "c_retrieval": 8, "v_math": 4, "v_retrieval_32k": 8, "v_hotpot": 4},
    }


def _checks(gsm: dict[str, list[dict[str, Any]]], retrieval: list[dict[str, Any]], hotpot: dict[str, Any]) -> dict[str, Any]:
    gsm_ids = [f"gsm8k-train-{row['row']}" for split in gsm.values() for row in split]
    hotpot_ids = [item["id"] for split in ("v", "expand", "t") for item in hotpot[split]]
    long_ids = set(hotpot["long_decode"])
    v_ids = {item["id"] for item in hotpot["v"]}
    return {
        "gsm8k_unique": len(gsm_ids) == len(set(gsm_ids)),
        "hotpot_unique": len(hotpot_ids) == len(set(hotpot_ids)),
        "long_decode_subset": long_ids <= v_ids,
        "c_folds": sorted({row["fold"] for row in gsm["c"]}) == [0, 1],
        "retrieval_count": len(retrieval),
    }


if __name__ == "__main__":
    main()
