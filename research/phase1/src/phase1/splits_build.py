"""在环境与高精度冒烟之后一次生成并冻结 splits/。只使用 tokenizer 计数，不跑模型前向。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
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
    "Solve the math problem. Show your work. End with a separate line containing "
    "exactly four # characters, one space, and the numeric answer. "
    "For example, if the answer is 42, the last line is: #### 42\n\n{question}"
)
RETRIEVAL_TEMPLATE = (
    "Find the value stored for the exact key below. Keys differing in any character "
    "are different. Reply with the value only, "
    "with no extra words.\n\n{context}\n\nKey: {key}\nValue:"
)
HOTPOT_MAIN = (
    "Read the documents and answer the question. Give a short explanation, "
    "then finish with a line that starts with Final answer: and gives the actual answer.\n\n"
    "{documents}\n\nQuestion: {question}"
)
HOTPOT_LONG = (
    "Read the documents and answer the question. Reason step by step, then give "
    "the actual answer on the last line, starting with Final answer:.\n\n"
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
    meta = json.loads(
        (root / "research/manifests/data" / manifest_name).read_text(encoding="utf-8")
    )
    return Path(meta["destination"])


def load_gsm8k(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["question", "answer"])
    rows = []
    for index, (question, answer) in enumerate(zip(
        table.column(0).to_pylist(), table.column(1).to_pylist()
    )):
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


def retrieval_spec(
    kind: str, index: int, length: int, depth: float, level: str, fold: int | None,
) -> dict[str, Any]:
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


def independent_mapping(seed: int, distractors: int) -> tuple[str, str, list[tuple[str, str]]]:
    """独立抽取键和值；拒绝重复后缀，避免正确值或干扰值泄露查询后缀。"""
    rng = np.random.default_rng(seed)
    used: set[str] = set()

    def draw(prefix: str) -> str:
        while True:
            suffix = f"{int(rng.integers(0, 2**32)):08x}"
            if suffix not in used:
                used.add(suffix)
                return prefix + suffix

    key, value = draw("key-"), draw("val-")
    return key, value, [(draw("key-"), draw("val-")) for _ in range(distractors)]


def build_retrieval(tokenizer: Any, spec: dict[str, Any], llama: bool) -> dict[str, Any]:
    """用填料句数二分，使聊天模板后的长度接近目标。"""
    distractors = 1 if spec["interference"] == "low" else 8
    key, value, others = independent_mapping(spec["seed"], distractors)
    low, high = distractors + 1, 4000
    best_n = low
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
    others: list[tuple[str, str]],
    sentences: int,
    depth: float,
    llama: bool,
) -> dict[str, Any]:
    lines = [FILLER_SENTENCE.format(i=i) for i in range(max(sentences, 1))]
    needle_at = min(len(lines) - 1, max(0, round(depth * (len(lines) - 1))))
    lines[needle_at] = NEEDLE_SENTENCE.format(key=key, value=value)
    used = {needle_at}
    for index, (other_key, other_value) in enumerate(others):
        slot = (needle_at + 3 + index * max(1, sentences // (len(others) + 2))) % sentences
        while slot in used:
            slot = (slot + 1) % sentences
        lines[slot] = NEEDLE_SENTENCE.format(key=other_key, value=other_value)
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
    start: int,
    llama: bool,
) -> dict[str, Any] | None:
    """两份支持文档隔开至少 8192 token。装不进 30K–31K 时返回 None。"""
    supports = [support_text(row, title) for title in row["support_titles"]]
    own = [
        f"{title}: {' '.join(sentences)}"
        for title, sentences in row["docs"].items()
        if title not in set(row["support_titles"])
    ]
    first = "[[SUPPORT-1]]\n" + supports[0] + "\n[[SUPPORT-1-END]]"
    second = "[[SUPPORT-2]]\n" + supports[1]

    def build(extra_gap: int, extra_tail: int, template: str) -> tuple[str, list[int]]:
        gap_lines = own + _pool_slice(pool, start, extra_gap)
        tail_lines = _pool_slice(pool, start + extra_gap, extra_tail)
        documents = "\n\n".join([first, *gap_lines, second, *tail_lines])
        user = template.format(documents=documents, question=row["question"])
        return render(tokenizer, user, llama)

    def gap_of(text: str) -> int:
        return _marker_gap(tokenizer, text, "[[SUPPORT-1-END]]", "[[SUPPORT-2]]")

    probe, probe_ids = build(0, 0, HOTPOT_LONG)
    if len(probe_ids) > TARGET_32K:
        return None
    best_gap = _smallest_gap(build, gap_of)
    if best_gap is None:
        return None
    best_tail = _longest_tail(build, best_gap)
    if best_tail is None:
        return None
    long_text, long_ids = build(best_gap, best_tail, HOTPOT_LONG)
    short_text, short_ids = build(best_gap, best_tail, HOTPOT_MAIN)
    gap_tokens = gap_of(long_text)
    long_n = len(long_ids)
    short_n = len(short_ids)
    if gap_tokens < GAP_TOKENS or long_n > TARGET_32K or short_n > TARGET_32K:
        return None
    if long_n < 30 * 1024:
        return None
    return {
        "main_text": short_text,
        "main_tokens": short_n,
        "main_sha256": sha256_text(short_text),
        "long_text": long_text,
        "long_tokens": long_n,
        "long_sha256": sha256_text(long_text),
        "support_titles": row["support_titles"],
        "gap_tokens": gap_tokens,
        "support_fraction": [
            _span_fraction(tokenizer, long_text, "[[SUPPORT-1]]"),
            _span_fraction(tokenizer, long_text, "[[SUPPORT-2]]"),
        ],
    }


def _pool_slice(pool: list[str], start: int, count: int) -> list[str]:
    if count <= 0:
        return []
    return [pool[(start + offset) % len(pool)] for offset in range(count)]


def _smallest_gap(build: Any, gap_of: Any) -> int | None:
    """找仍能放进窗口的最少填料句，使两份支持文档的间隔达到 8192 token。"""
    low, high = 0, 2500
    best: int | None = None
    while low <= high:
        mid = (low + high) // 2
        _text, ids = build(mid, 0, HOTPOT_LONG)
        if len(ids) > TARGET_32K:
            high = mid - 1
            continue
        if gap_of(_text) >= GAP_TOKENS:
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    return best


def _longest_tail(build: Any, extra_gap: int) -> int | None:
    """在不越过 31744 token 的前提下，把较长问法尽量放到 30K 以上。"""
    low, high = 0, 2500
    best: int | None = None
    best_len = -1
    while low <= high:
        mid = (low + high) // 2
        _text, ids = build(extra_gap, mid, HOTPOT_LONG)
        length = len(ids)
        if length > TARGET_32K:
            high = mid - 1
            continue
        if length >= best_len:
            best = mid
            best_len = length
        low = mid + 1
    if best is None or best_len < 30 * 1024:
        return None
    return best


def _marker_gap(tokenizer: Any, text: str, first: str, second: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    first_at = text.find(first)
    second_at = text.find(second)
    if first_at < 0 or second_at < 0:
        return 0

    def locate(char_pos: int) -> int:
        for index, (left, right) in enumerate(offsets):
            if left <= char_pos < max(right, left + 1):
                return index
        return 0

    return locate(second_at) - locate(first_at)


def saved_retrieval(tokenizer: Any, spec: dict[str, Any], text: str) -> dict[str, Any]:
    """从已经写好的全文恢复长度、深度和校验值，避免重跑二分搜索。"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    key = f"key-{spec['seed']:08x}"
    value = f"val-{spec['seed']:08x}"
    needle = NEEDLE_SENTENCE.format(key=key, value=value)
    return {
        "text": text,
        "tokens": len(ids),
        "actual_depth": _span_fraction(tokenizer, text, needle),
        "key": key,
        "value": value,
        "sha256": sha256_text(text),
    }


def _take_hotpot(
    rows: list[dict[str, Any]],
    count: int,
    label: str,
    banned: set[str],
    pool: list[str],
    text_root: Path,
    qwen: Any,
    llama: Any,
) -> tuple[list[dict[str, Any]], set[str]]:
    """按顺序装箱，装不下的题跳过，直到凑满 count。"""
    taken: list[dict[str, Any]] = []
    blocked = set(banned)
    models = (("qwen", qwen, False), ("llama", llama, True))
    for index, row in enumerate(rows):
        if len(taken) >= count:
            break
        if row["id"] in blocked:
            continue
        start = (index * 2500) % len(pool)
        packed_by_model: dict[str, dict[str, Any]] = {}
        for name, tokenizer, is_llama in models:
            packed = pack_hotpot(tokenizer, row, pool, start, is_llama)
            if packed is None:
                packed_by_model = {}
                break
            packed_by_model[name] = packed
        if not packed_by_model:
            print(f"skip {label} {row['id']}", flush=True)
            continue
        item: dict[str, Any] = {
            "id": row["id"],
            "row": row["row"],
            "answer": row["answer"],
            "models": {},
        }
        for name, packed in packed_by_model.items():
            folder = text_root / "hotpot" / name / label
            _store_text(folder / f"{row['id']}.main.txt", packed["main_text"])
            _store_text(folder / f"{row['id']}.long.txt", packed["long_text"])
            item["models"][name] = {
                "main_tokens": packed["main_tokens"],
                "main_sha256": packed["main_sha256"],
                "long_tokens": packed["long_tokens"],
                "long_sha256": packed["long_sha256"],
                "support_titles": packed["support_titles"],
                "gap_tokens": packed["gap_tokens"],
                "support_fraction": packed["support_fraction"],
            }
        taken.append(item)
        print(f"hotpot {label} {len(taken)} {row['id']}", flush=True)
    if len(taken) < count:
        raise SystemExit(f"{label} 只装进 {len(taken)} 题，需要 {count}")
    return taken, {item["id"] for item in taken}


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
    ordered = [hotpot_rows[int(index)] for index in order]
    t_block = ordered[:500]
    rest = ordered[500:]
    held_ids = {row["id"] for row in t_block}
    all_hotpot = load_hotpot(parquet_path(root, "hotpotqa.source.json"))
    pool = filler_pool(all_hotpot, held_ids)
    retrieval_public = []
    for spec in retrieval_plan():
        record = {
            "id": spec["id"],
            "seed": spec["seed"],
            "target_tokens": spec["target_tokens"],
            "depth": spec["depth"],
            "interference": spec["interference"],
            "fold": spec["fold"],
        }
        record["models"] = {}
        for name, tokenizer, is_llama in (("qwen", qwen, False), ("llama", llama, True)):
            path = text_root / "retrieval" / name / f"{spec['id']}.txt"
            if path.exists():
                built = saved_retrieval(tokenizer, spec, path.read_text(encoding="utf-8"))
            else:
                built = build_retrieval(tokenizer, spec, is_llama)
                _store_text(path, built["text"])
            record["models"][name] = {
                "tokens": built["tokens"],
                "actual_depth": built["actual_depth"],
                "key": built["key"],
                "value": built["value"],
                "sha256": built["sha256"],
            }
        retrieval_public.append(record)
        if len(retrieval_public) % 20 == 0:
            print(f"retrieval {len(retrieval_public)}", flush=True)
    hotpot_t, _used_t = _take_hotpot(t_block, HOTPOT_T, "t", set(), pool, text_root, qwen, llama)
    used = {item["id"] for item in hotpot_t}
    hotpot_v, _used_v = _take_hotpot(rest, HOTPOT_V, "v", used, pool, text_root, qwen, llama)
    used.update(item["id"] for item in hotpot_v)
    hotpot_expand, _used_e = _take_hotpot(
        rest, HOTPOT_EXPAND, "expand", used, pool, text_root, qwen, llama
    )
    hotpot_public = {
        "v": hotpot_v,
        "expand": hotpot_expand,
        "t": hotpot_t,
        "long_decode": [item["id"] for item in hotpot_v[:LONG_DECODE]],
    }
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
    item = {
        "id": f"gsm8k-train-{row['row']}", "row": row["row"],
        "question_sha256": sha256_text(row["question"]),
    }
    if "fold" in row:
        item["fold"] = row["fold"]
    return item


def _store_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rebuild_hotpot_prompts() -> None:
    """仅替换 HotpotQA 问法；先核对全部旧文本，再更新全文和冻结元数据。"""
    root = project_root()
    split_root = root / "research/phase1/splits"
    text_root = root / "research/.runtime/data/phase1-text/hotpot"
    templates_path = split_root / "templates.json"
    hotpot_path = split_root / "hotpot.json"
    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    hotpot = json.loads(hotpot_path.read_text(encoding="utf-8"))
    previous = {"main": templates["hotpot_main"], "long": templates["hotpot_long"]}
    replacement = {"main": HOTPOT_MAIN, "long": HOTPOT_LONG}
    if any("<answer>" not in value for value in previous.values()):
        raise SystemExit("旧 HotpotQA 问法不是预期的占位符版本")
    tokenizers = {
        "qwen": load_tokenizer(root, "models/Qwen2.5-7B-Instruct"),
        "llama": load_tokenizer(root, "models/Llama-3.1-8B-Instruct"),
    }
    pending: list[tuple[Path, str]] = []
    for split in ("v", "expand", "t"):
        for item in hotpot[split]:
            for model, tokenizer in tokenizers.items():
                meta = item["models"][model]
                for kind in ("main", "long"):
                    path = text_root / model / split / f"{item['id']}.{kind}.txt"
                    old_text = path.read_text(encoding="utf-8")
                    if sha256_text(old_text) != meta[f"{kind}_sha256"]:
                        raise SystemExit(f"旧全文与冻结校验值不符：{path}")
                    old_start = previous[kind].split("{documents}", maxsplit=1)[0]
                    new_start = replacement[kind].split("{documents}", maxsplit=1)[0]
                    if old_text.count(old_start) != 1:
                        raise SystemExit(f"旧问法在全文中未恰好出现一次：{path}")
                    new_text = old_text.replace(old_start, new_start, 1)
                    encoded = tokenizer(
                        new_text, add_special_tokens=False, return_offsets_mapping=True
                    )
                    offsets = encoded["offset_mapping"]
                    count = len(encoded["input_ids"])
                    if count < 30 * 1024 or count > TARGET_32K:
                        raise SystemExit(f"修订后长度越界：{path}，{count} token")
                    meta[f"{kind}_tokens"] = count
                    meta[f"{kind}_sha256"] = sha256_text(new_text)
                    if kind == "long":
                        markers = ("[[SUPPORT-1]]", "[[SUPPORT-1-END]]", "[[SUPPORT-2]]")
                        positions = [new_text.find(marker) for marker in markers]
                        if any(position < 0 for position in positions):
                            raise SystemExit(f"支持文档标记缺失：{path}")
                        indices = []
                        for position in positions:
                            indices.append(next(
                                index for index, (left, right) in enumerate(offsets)
                                if left <= position < max(right, left + 1)
                            ))
                        meta["gap_tokens"] = indices[2] - indices[1]
                        meta["support_fraction"] = [indices[0] / count, indices[2] / count]
                        if meta["gap_tokens"] < GAP_TOKENS:
                            raise SystemExit(f"支持文档间隔不足：{path}")
                    pending.append((path, new_text))
    templates["hotpot_main"] = HOTPOT_MAIN
    templates["hotpot_long"] = HOTPOT_LONG
    for path, new_text in pending:
        _store_text(path, new_text)
    write_json(templates_path, templates)
    write_json(hotpot_path, hotpot)
    print(f"HotpotQA 答案占位符已移除：{len(pending)} 份全文", flush=True)


def rebuild_retrieval_inputs() -> None:
    """核对旧检索全文与样本计划后，重建唯一键版本的所有冻结输入。"""
    root = project_root()
    split_root = root / "research/phase1/splits"
    text_root = root / "research/.runtime/data/phase1-text/retrieval"
    templates_path = split_root / "templates.json"
    retrieval_path = split_root / "retrieval.json"
    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    if "Find the value stored for the given key." not in templates["retrieval"]:
        raise SystemExit("旧检索问法不是预期版本")
    records = json.loads(retrieval_path.read_text(encoding="utf-8"))
    plans = retrieval_plan()
    if len(records) != len(plans):
        raise SystemExit("冻结检索题数与生成计划不一致")
    tokenizers = {
        "qwen": load_tokenizer(root, "models/Qwen2.5-7B-Instruct"),
        "llama": load_tokenizer(root, "models/Llama-3.1-8B-Instruct"),
    }
    pending: list[tuple[Path, str]] = []
    for record, spec in zip(records, plans):
        for field in ("id", "seed", "target_tokens", "depth", "interference", "fold"):
            if record[field] != spec[field]:
                raise SystemExit(f"冻结检索题的 {field} 与计划不一致：{record['id']}")
        for model, tokenizer in tokenizers.items():
            path = text_root / model / f"{record['id']}.txt"
            old_text = path.read_text(encoding="utf-8")
            if sha256_text(old_text) != record["models"][model]["sha256"]:
                raise SystemExit(f"旧检索全文校验失败：{path}")
            built = build_retrieval(tokenizer, spec, model == "llama")
            count = built["tokens"]
            if count > spec["target_tokens"] or (
                spec["target_tokens"] == TARGET_32K and count < 30 * 1024
            ):
                raise SystemExit(f"重建检索全文长度越界：{path}，{count} token")
            needle = NEEDLE_SENTENCE.format(key=built["key"], value=built["value"])
            if built["text"].count(needle) != 1:
                raise SystemExit(f"目标证据不唯一：{path}")
            record["models"][model] = {
                "tokens": count,
                "actual_depth": built["actual_depth"],
                "key": built["key"],
                "value": built["value"],
                "sha256": built["sha256"],
            }
            pending.append((path, built["text"]))
        if len(pending) % 64 == 0:
            print(f"retrieval rebuilt {len(pending)}", flush=True)
    templates["retrieval"] = RETRIEVAL_TEMPLATE
    for path, text in pending:
        _store_text(path, text)
    write_json(templates_path, templates)
    write_json(retrieval_path, records)
    print(f"完整键匹配检索输入已更新：{len(pending)} 份全文", flush=True)


def _trajectories(
    gsm: dict[str, list[dict[str, Any]]], retrieval: list[dict[str, Any]], hotpot: dict[str, Any],
) -> dict[str, Any]:
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


def _checks(
    gsm: dict[str, list[dict[str, Any]]], retrieval: list[dict[str, Any]], hotpot: dict[str, Any],
) -> dict[str, Any]:
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
    if sys.argv[1:] == ["--rebuild-hotpot-prompts"]:
        rebuild_hotpot_prompts()
    elif sys.argv[1:] == ["--rebuild-retrieval-inputs"]:
        rebuild_retrieval_inputs()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("只支持 --rebuild-hotpot-prompts 或 --rebuild-retrieval-inputs")
