"""共用冻结输入字段核对；历史阶段审核命令转入当前只读审核。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kvstudy.artifacts import read_json as _read
from kvstudy.paths import project_root
from phase1.inputs import _ids_and_source, _inputs
from phase1.inputs import _samples as _long_samples
from phase1.inputs import four_samples as _four_samples
from phase1.models import MODELS
from phase1.result_format import is_quality_result, normalize_quality_result


def _subset_equal(old: Any, new: Any) -> bool:
    """允许新增格式字段，已有字典值、行顺序和逐题分数必须原样保留。"""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(
            key in new and _subset_equal(value, new[key]) for key, value in old.items()
        )
    if isinstance(old, list):
        return isinstance(new, list) and len(old) == len(new) and all(
            _subset_equal(a, b) for a, b in zip(old, new)
        )
    return type(old) is type(new) and old == new


def _quality_plan(root: Path, records: dict[Path, Any]) -> dict[Path, Any]:
    from transformers import AutoTokenizer

    tokenizers = {}
    catalogs = {}
    plan = {}
    for path, value in records.items():
        if not is_quality_result(value):
            continue
        model = value["model"]
        if "extra_n" in value:
            task, split = value["task"], "expand"
        else:
            split, task = value["task"].split("-", 1)
        key = model, task, split
        if key not in catalogs:
            if model not in tokenizers:
                tokenizers[model] = AutoTokenizer.from_pretrained(root / MODELS[model])
            tokenizer = tokenizers[model]
            if task == "hotpot_four":
                items = _four_samples(root, model)
            elif task == "retrieval_ruler":
                items = _long_samples(root, model, task, split)
            else:
                items = _inputs(root, model, task, split)
            catalog = {}
            for item in items:
                if task in ("retrieval_ruler", "hotpot_four"):
                    source = (root / item["path"]).read_text("utf-8")
                    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                    ids = tokenizer.encode(source, add_special_tokens=False)
                    if digest != item["sha256"] or len(ids) != item["tokens"]:
                        raise RuntimeError(f"新输入不符合冻结：{item['id']}")
                else:
                    ids, digest = _ids_and_source(root, tokenizer, model, task, item)
                catalog[item["id"]] = {"input_tokens": len(ids), "input_sha256": digest}
            catalogs[key] = catalog
        catalog = catalogs[key]
        ids = [row["id"] for row in value["rows"]]
        if ids != list(catalog)[:len(ids)] or len(ids) != len(set(ids)):
            raise RuntimeError(f"冻结顺序或样本重复：{path}")
        expected_n = value.get("extra_n", value.get("observed_n", len(catalog)))
        if len(ids) != expected_n:
            raise RuntimeError(f"冻结样本数不一致：{path}")
        normalized = normalize_quality_result(value, catalog)
        if not _subset_equal(value, normalized):
            raise RuntimeError(f"格式整理改写了已有测量：{path}")
        rows = normalized["rows"]
        em = 100 * sum(row.get("correct", row.get("em", False)) for row in rows) / len(rows)
        if "em" in normalized and abs(normalized["em"] - em) > 1e-10:
            raise RuntimeError(f"EM 汇总不符：{path}")
        if "f1" in normalized:
            f1 = sum(row["f1"] for row in rows) / len(rows)
            if abs(normalized["f1"] - f1) > 1e-10:
                raise RuntimeError(f"F1 汇总不符：{path}")
        if "correct" in normalized and normalized["correct"] != sum(
            row["correct"] for row in rows
        ):
            raise RuntimeError(f"答对题数不符：{path}")
        plan[path] = normalized
    return plan


def run(commit: str) -> None:
    """兼容历史审核命令，当前仅复查唯一实体，不再执行旧选择与替换链。"""
    from phase1.audit import audit

    print(json.dumps(audit(project_root()), ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    """在指定 node8 交互环境执行，审核只重建输入与统计，不做模型前向。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.commit)


if __name__ == "__main__":
    main()
