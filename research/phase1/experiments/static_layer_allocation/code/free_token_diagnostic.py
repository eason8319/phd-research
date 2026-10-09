"""复现内容损失近零但自由生成错误的题，保存首次分词路径分歧。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.generation_error_sources.code.quantized_generation import (
    _config, _decode_step, _prefill_cache,
)
from experiments.generation_error_sources.code.short_baseline import _eos_ids
from experiments.static_layer_allocation.code.content import (
    capture_source, quantized_trace, read_input, reference_trace, save,
)
from experiments.static_layer_allocation.code.content_metrics import token_metrics
from experiments.static_layer_allocation.code.protocol import digest, load_plan
from phase1.representation import MODELS
from phase1.runner import decode_step, load_model


def frame(
    logits: torch.Tensor | np.ndarray, tokenizer: Any, expected: int | None,
    prefix_matches: bool, index: int,
) -> dict[str, Any]:
    """记录三个最大 logit 候选与固定目标支持度；失配后明确前缀身份。"""
    values = logits.detach().float().cpu().numpy().reshape(-1) \
        if isinstance(logits, torch.Tensor) else logits.reshape(-1)
    indices = np.argsort(-values, kind="stable")[:3].tolist()
    return {"answer_index": index, "correct_token_prefix": prefix_matches,
            "expected_id": expected, "expected_text": tokenizer.decode([expected])
            if expected is not None else None,
            "target": token_metrics(values, expected) if expected is not None else None,
            "top": [{"id": i, "text": tokenizer.decode([i]), "logit": float(values[i])}
                    for i in indices]}


def free_trace(
    model: Any, tokenizer: Any, source: dict[str, Any], answer: list[int], eos: set[int],
    config: dict[str, Any] | None, params: dict[str, Any], h: int, limit: int,
) -> dict[str, Any]:
    """保持正式自由生成的缓存规则，附加 token 身份与预测前目标支持度。"""
    if config is None:
        keys, values = list(source["keys"]), list(source["values"])
    else:
        _, keys, values, read_keys, read_values, bad = _prefill_cache(
            model, source["input_ids"], config, params, limit, h, prefill_source=source,
        )
        if bad:
            raise RuntimeError("分词诊断缓存无效")
    produced = [source["token"]]
    frames = [frame(source["logits"], tokenizer, answer[0], True, 0)]
    for step in range(limit - 1):
        if produced[-1] in eos:
            break
        position = source["input_ids"].shape[1] + step
        if config is None:
            logits, keys, values = decode_step(
                model, keys, values, produced[-1], position, consume_cache=True,
            )
            token = int(logits.argmax())
        else:
            captured = []
            token, bad = _decode_step(
                model, keys, values, read_keys, read_values, produced[-1], position,
                config, params, h, observer=lambda event: captured.append(event["logits"])
                if "logits" in event else None,
            )
            if bad or token is None or len(captured) != 1:
                raise RuntimeError("分词诊断解码无效")
            logits = captured[0]
        expected = answer[step + 1] if step + 1 < len(answer) else None
        frames.append(frame(logits, tokenizer, expected, produced == answer[:len(produced)],
                            step + 1))
        produced.append(token)
    return {"token_ids": produced,
            "prediction": tokenizer.decode(produced, skip_special_tokens=True).strip(),
            "frames": frames}


def run(commit: str) -> None:
    """只复核已观察到的 Qwen 失败案例，不改变主矩阵或分配判据。"""
    root = project_root()
    plan = load_plan(root)
    folder = root / "research/.runtime/static-key-content/measurements/qwen"
    read = lambda name: json.loads((folder / (name + ".json")).read_text("utf-8"))
    gate, allocation = read("gate"), read("allocations")
    dataset = json.loads((root / plan["dataset"]).read_text("utf-8"))
    item = next(row for row in dataset["rows"] if row["key"] == "key-909c1bd1")
    row = next(row for row in gate["rows"] if row["id"] == item["id"])
    if torch.cuda.mem_get_info()[0] < plan["memory_estimate_gib"]["qwen"] * 2**30:
        raise RuntimeError("分词诊断显存未达到冻结估计")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS["qwen"])
    model = load_model(str(root / MODELS["qwen"]))
    eos = _eos_ids(tokenizer, False)
    calibration = read("calibration")
    packed = torch.load(root / calibration["path"], weights_only=False, map_location="cpu")
    params = {"key_by_bits": {bits: [tuple(t.to("cuda") for t in tensors)
                                     for tensors in entries]
                               for bits, entries in packed["params"].items()}}
    params["key"] = params["key_by_bits"][8]
    names = ["bf16", "kv8", "online_pre_k8v4", "online_post_k8v4", "offline_pre_k8v4",
             "online_pre_k4_layers_2_sensitivity", "online_post_k4_layers_14_sensitivity",
             "offline_pre_k4_layers_7_sensitivity"]
    schemes = {}
    answer = item["models"]["qwen"]["answer_ids"]
    with torch.inference_mode():
        source = capture_source(model, read_input(root, item, "qwen"))
        for name in names:
            config, h, selected_params = None, 0, {}
            if name == "kv8":
                config = _config(plan["kv8_reference"])
            elif name != "bf16":
                spec = allocation["allocations"][name]
                format_spec = plan["formats"][spec["format"]]
                config = {**_config(format_spec["representation"]),
                          "layer_bits": spec["layer_bits"]}
                h = format_spec["residual_bits"]
                selected_params = params if config["key_mode"] == "per_channel" else {}
            forced = reference_trace(model, source, answer, list(range(len(answer)))) \
                if config is None else quantized_trace(
                    model, source, answer, list(range(len(answer))), config, selected_params, h,
                )
            measured = free_trace(model, tokenizer, source, answer, eos, config, selected_params,
                                  h, plan["generation"]["max_new_tokens"])
            if measured["prediction"] != row["schemes"][name]["prediction"]:
                raise RuntimeError("自由生成案例未复现主矩阵输出")
            measured["teacher_forced"] = [frame(z, tokenizer, answer[i], True, i)
                                         for i, z in enumerate(forced)]
            schemes[name] = measured
    save(folder / "free_token_diagnostic.json", {
        "commit": commit, "model": "qwen", "plan_sha256": digest(plan),
        "selection_role": "post_measurement_explanation_of_observed_discrepancy",
        "main_execution_commit": gate["commit"], "input_sha256": row["input_sha256"],
        "key": item["key"], "answer": item["value"], "answer_ids": answer,
        "answer_offsets": item["models"]["qwen"]["answer_offsets"],
        "content_indices": row["content_indices"], "schemes": schemes,
        "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print("FREE_TOKEN_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    run(parser.parse_args().commit)
