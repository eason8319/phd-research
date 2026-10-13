"""冻结独立随机检索映射、内容 token 和仅改变目标值的配对输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from experiments.static_layer_allocation.code.protocol import digest, load_plan, write_json
from kvstudy.paths import project_root
from phase1.representation import MODELS
from phase1.splits_build import NEEDLE_SENTENCE, build_retrieval, independent_mapping


def answer_positions(tokenizer: Any, answer: str) -> dict[str, Any]:
    """按实际分词的字符跨度选内容位置；跨边界 token 整体计入，先预测再输入。"""
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (left, right) in enumerate(encoded.offset_mapping)
                 if right > len("val-") and left < len(answer)]
    if not answer.startswith("val-") or not positions or positions[0] == 0:
        raise RuntimeError("内容位置未落在可逐步解码的答案前缀之后")
    ids = list(encoded.input_ids)
    if tokenizer.decode(ids, skip_special_tokens=False) != answer:
        raise RuntimeError("目标 token 序列不能准确还原完整答案")
    return {"answer_ids": ids, "content_indices": positions,
            "answer_offsets": [list(pair) for pair in encoded.offset_mapping]}


def freeze(commit: str) -> None:
    """仅生成本实验 C 的两折新输入；已有检索清单及其他实验输入保持历史身份。"""
    root = project_root()
    plan = load_plan(root)
    original = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
    specs = [row for row in original if row["id"].startswith("synth-c-")]
    tokenizers = {name: AutoTokenizer.from_pretrained(root / path)
                  for name, path in MODELS.items()}
    folder = root / "research/.runtime/static-key-content/inputs"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    cal_index = 0
    for original_spec in specs:
        spec = {key: original_spec[key] for key in (
            "id", "seed", "target_tokens", "depth", "interference", "fold"
        )}
        count = 1 if spec["interference"] == "low" else 8
        key, value, others = independent_mapping(spec["seed"], count)
        row = {**spec, "role": "calibration" if spec["fold"] == 0 else "validation",
               "key": key, "value": value, "distractors": others, "models": {}}
        for name, tokenizer in tokenizers.items():
            built = build_retrieval(tokenizer, spec, name == "llama")
            assert built["key"] == key and built["value"] == value
            path = folder / name / (spec["id"] + ".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(built["text"], encoding="utf-8", newline="\n")
            row["models"][name] = {
                key: built[key] for key in ("tokens", "actual_depth", "sha256")
            }
            row["models"][name].update(
                path=path.relative_to(root).as_posix(), **answer_positions(tokenizer, value)
            )
        if spec["fold"] == 0:
            if cal_index in plan["mapping_checks"]["paired_calibration_indices"]:
                rng = np.random.default_rng(spec["seed"] ^ 0x4A97163B)
                for _ in range(10000):
                    alternate = f"val-{int(rng.integers(0, 2**32)):08x}"
                    forbidden = {key[4:], value[4:], *(v[4:] for _, v in others)}
                    if alternate[4:] in forbidden:
                        continue
                    texts = {}
                    for name, tokenizer in tokenizers.items():
                        text = (root / row["models"][name]["path"]).read_text("utf-8")
                        text = text.replace(NEEDLE_SENTENCE.format(key=key, value=value),
                                            NEEDLE_SENTENCE.format(key=key, value=alternate))
                        if len(tokenizer.encode(text, add_special_tokens=False)) != \
                                row["models"][name]["tokens"]:
                            break
                        texts[name] = text
                    if len(texts) == len(tokenizers):
                        break
                else:
                    raise RuntimeError("未能冻结同 token 长度的映射配对")
                row["paired_value"] = alternate
                row["paired_models"] = {}
                for name, text in texts.items():
                    path = folder / name / (spec["id"] + "-swapped.txt")
                    path.write_text(text, encoding="utf-8", newline="\n")
                    row["paired_models"][name] = {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "tokens": row["models"][name]["tokens"],
                        **answer_positions(tokenizers[name], alternate),
                    }
            cal_index += 1
        rows.append(row)
    audit = {
        "old_c_same_key_value_suffix": {
            name: sum(r["models"][name]["key"][4:] == r["models"][name]["value"][4:]
                      for r in specs) for name in MODELS
        },
        "new_key_value_suffix_matches": sum(r["key"][4:] == r["value"][4:] for r in rows),
        "new_distractor_suffix_matches": sum(
            v[4:] == r["value"][4:] for r in rows for _, v in r["distractors"]
        ),
        "paired_n": sum("paired_value" in r for r in rows),
    }
    write_json(root / plan["dataset"], {
        "protocol": plan["input_protocol"], "commit": commit, "spec_source":
        "research/phase1/splits/retrieval.json", "spec_source_sha256":
        hashlib.sha256((root / "research/phase1/splits/retrieval.json").read_bytes()).hexdigest(),
        "counts": plan["counts"], "rows": rows, "audit": audit,
        "mapping_seed_rule": "numpy_default_rng_original_spec_seed_unique_independent_draws",
        "identity_digest": digest(rows),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    freeze(parser.parse_args().commit)
