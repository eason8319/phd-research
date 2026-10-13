"""冻结独立质量验证的检索扩样与四问请求，保留原 V 输入字节。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

from kvstudy.artifacts import read_json as read
from experiments.static_layer_allocation.code.protocol import digest, load_plan, write_json


def expansion_groups(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """原扩样顺序每四题组成请求，拒绝重复或不足四题。"""
    if len(items) != 192 or len({item["id"] for item in items}) != 192:
        raise RuntimeError("四问扩样须来自冻结的 192 道不同问题")
    return [items[start:start + 4] for start in range(0, 192, 4)]


def generation_config(config: dict[str, Any]) -> dict[str, Any]:
    """扩样元数据不改变原 V 的题面和生成协议。"""
    return {key: value for key, value in config.items() if key != "expansion"}


def _save_input(root: Path, path: Path, text: str, identity: dict[str, Any]) -> None:
    """冻结新全文；不同输入不能覆盖已冻结实体。"""
    import hashlib

    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
        raise RuntimeError(f"扩样输入冲突：{path.relative_to(root)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")
    identity["path"] = path.relative_to(root).as_posix()


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    """恢复时要求扩样清单完全相同。"""
    if path.exists() and read(path) != payload:
        raise RuntimeError(f"扩样冻结清单冲突：{path.name}")
    write_json(path, payload)


def freeze_inputs(root: Path) -> dict[str, Any]:
    """仅用 tokenizer 冻结全部扩样，不运行模型前向或改写原 V。"""
    from transformers import AutoTokenizer
    from phase1.ruler_retrieval import build_input as build_retrieval, file_hash
    from phase1.splits_build import load_hotpot
    from phase1.inputs import build_input
    from phase1.models import MODELS

    retrieval = read(root / "research/phase1/configs/retrieval.json")
    four = read(root / "research/phase1/configs/long_decode.json")
    originals = read(root / retrieval["identity_source"])
    specs = [{k: v for k, v in item.items() if k != "models"} for item in originals
             if item["id"].startswith(retrieval["expansion"]["identity_prefix"])]
    if len(specs) != 192:
        raise RuntimeError("检索扩样不是原冻结的 192 题")
    for size in (64, 192):
        strata = Counter((s["depth"], s["interference"]) for s in specs[:size])
        if len(strata) != 8 or len(set(strata.values())) != 1:
            raise RuntimeError("检索累计档位的扩样前缀不平衡")
    retrieval_items = [dict(spec, models={}) for spec in specs]
    hotpot = read(root / four["identity_source"])
    groups = expansion_groups(hotpot["expand"])
    source = read(root / "research/manifests/data/hotpotqa.source.json")
    source_path = Path(source["destination"]).resolve()
    project = (root / "models").resolve().parent
    if not source_path.is_relative_to(project):
        raise RuntimeError("Hotpot 数据源超出项目根目录")
    if file_hash(source_path) != source["files"][0]["sha256"]:
        raise RuntimeError("Hotpot 数据源校验失败")
    raw = load_hotpot(source_path)
    indexed = {item["id"]: item for item in raw}
    banned = {item["id"] for split in ("v", "expand", "t") for item in hotpot[split]}
    for offset, n in ((four["pilot_single_seed_offset"], four["pilot_single_n"]),
                      (four["pilot_four_seed_offset"], four["pilot_four_n"])):
        allowed = [row for row in raw if row["id"] not in banned
                   and len(row["support_titles"]) == 2]
        random.Random(four["master_seed"] + offset).shuffle(allowed)
        banned.update(row["id"] for row in allowed[:n])
    pool = [f"{title}: {sentence}" for row in sorted(raw, key=lambda r: r["id"])
            if row["id"] not in banned for title, sentences in sorted(row["docs"].items())
            for sentence in sentences if sentence.strip()]
    four_items = []
    for group, items in enumerate(groups):
        rows = [indexed[item["id"]] for item in items]
        if any(row["answer"] != item["answer"] for row, item in zip(rows, items)):
            raise RuntimeError("四问扩样标准答案与源数据不同")
        four_items.append({
            "id": f"hotpot-four-expand-{group:02d}",
            "question_ids": [item["id"] for item in items],
            "answers": [item["answer"] for item in items], "models": {},
        })
    for model, model_path in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(root / model_path)
        for item in retrieval_items:
            text, identity = build_retrieval(tokenizer, item, model, retrieval)
            path = root / "research/.runtime/data/phase1-text/retrieval_ruler_expand"
            _save_input(root, path / model / f"{item['id']}.txt", text, identity)
            item["models"][model] = identity
        print("FROZEN_RETRIEVAL", model, len(retrieval_items), flush=True)
        for group, item in enumerate(four_items):
            rows = [indexed[item_id] for item_id in item["question_ids"]]
            text, identity = build_input(tokenizer, model, rows, pool, group + 16, four)
            path = root / "research/.runtime/data/phase1-text/hotpot_four_expand"
            _save_input(root, path / model / f"{item['id']}.txt", text, identity)
            item["models"][model] = identity
            print("FROZEN_FOUR", model, group + 1, flush=True)
    manifests = {}
    for config, items in ((retrieval, retrieval_items), (four, four_items)):
        path = root / config["expansion"]["manifest"]
        payload = {
            "protocol": config["protocol"], "generation_config_digest": digest(
                generation_config(config)), "items": items, "n": len(items),
            "identity_source_sha256": file_hash(root / config["identity_source"]),
            "order": config["expansion"],
        }
        _save_manifest(path, payload)
        manifests[path.relative_to(root).as_posix()] = file_hash(path)
    return manifests


def samples(root: Path, model: str, task: str, total: int) -> list[dict[str, Any]]:
    """返回累计冻结前缀，四问以请求计数，单问以题计数。"""
    from phase1.inputs import _inputs

    if task in ("gsm8k", "hotpot"):
        return (_inputs(root, model, task, "v") +
                _inputs(root, model, task, "expand"))[:total]
    config_path = "retrieval.json" if task == "retrieval_ruler" else "long_decode.json"
    config = read(root / "research/phase1/configs" / config_path)
    v_path = "retrieval_ruler.json" if task == "retrieval_ruler" else "hotpot_four.json"
    old = read(root / "research/phase1/splits" / v_path)
    extra = read(root / config["expansion"]["manifest"])
    if extra["generation_config_digest"] != digest(generation_config(config)):
        raise RuntimeError("扩样生成协议已变")
    items = (old["items"] + extra["items"])[:total]
    rows = []
    for item in items:
        identity = item["models"][model]
        common = {"id": item["id"], **identity}
        if task == "retrieval_ruler":
            common.update(answer=identity["value"], depth=item["depth"],
                          interference=item["interference"])
        else:
            common.update(answers=item["answers"], question_ids=item["question_ids"])
        rows.append(common)
    if len(rows) != total or len({row["id"] for row in rows}) != total:
        raise RuntimeError("质量累计前缀长度或身份异常")
    return rows


def totals(root: Path, task: str) -> list[int]:
    """从唯一任务配置读取预先规定的判断档位。"""
    spec = load_plan(root)["v_plan"]["tasks"][task]
    if "totals" in spec:
        return spec["totals"]
    return read(root / spec["expansion_source"])["expansion"]["totals"]


if __name__ == "__main__":
    from kvstudy.paths import project_root

    print(json.dumps(freeze_inputs(project_root()), ensure_ascii=False), flush=True)
