"""运行固定源值的完整路径、关键层保护对照与在线 post 配对精度诊断。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.generation_error_sources.code.quantized_generation import _config
from experiments.generation_error_sources.code.short_baseline import _eos_ids
from experiments.static_layer_allocation.code.allocations import cost, layer_bits
from experiments.static_layer_allocation.code.content import capture_source, read_input, save
from experiments.static_layer_allocation.code.metrics import ranking
from experiments.static_layer_allocation.code.numerics_forward import generate, trace
from experiments.static_layer_allocation.code.numerics_metrics import path_metrics, score_answer
from experiments.static_layer_allocation.code.protocol import code_sources, digest, load_plan
from phase1.representation import MODELS
from phase1.runner import load_model

DESIGN = "research/phase1/configs/static_numerics.json"
RUNTIME = "research/.runtime/static-key-numerics"
FORMAL = "research/phase1/experiments/static_layer_allocation/results"


def read(path: Path) -> dict[str, Any]:
    """读取冻结配置或小型逐题记录。"""
    return json.loads(path.read_text("utf-8"))


def sha(path: Path) -> str:
    """校验保留实体的实际字节。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_allocations(
    base: dict[str, Any], probe: dict[str, Any], plan: dict[str, Any], geometry: dict[str, Any],
    design: dict[str, Any],
) -> dict[str, Any]:
    """最高两层仅由既有校准 S 确定，过滤原随机排列后保持预算嵌套。"""
    result = {"allocations": dict(base["allocations"]), "protected": {},
              "base_allocation_digest": base["allocation_digest"],
              "base_probe_digest": digest(probe), "protected_random_names": []}
    layers = len(next(iter(base["allocations"].values()))["layer_bits"])
    for fmt, spec in plan["formats"].items():
        protected = ranking(probe["formats"][fmt]["layers"], "sensitivity",
                            descending=True)[:design["protected_random"]["protected_count"]]
        result["protected"][fmt] = protected
        for seed, permutation in base["random_permutations"][fmt].items():
            allowed = [layer for layer in permutation if layer not in protected]
            for r in plan["models"][probe["model"]]["k4_layers"]:
                name = f"{fmt}_k4_layers_{r}_protected_random_{seed}"
                bits = layer_bits(layers, allowed[:r])
                result["allocations"][name] = {
                    "format": fmt, "strategy": "protected_random_" + seed, "r": r,
                    "k4_layers": allowed[:r], "layer_bits": bits,
                    "cost": {str(n): cost(bits, geometry, _config(spec["representation"]),
                                          spec["residual_bits"], n)
                             for n in plan["cost_lengths"]},
                }
                result["protected_random_names"].append(name)
    configurations = {}
    result["configuration_aliases"] = {}
    for name, spec in result["allocations"].items():
        identity = (spec["format"], tuple(row["k"] for row in spec["layer_bits"]))
        alias = configurations.setdefault(identity, name)
        result["configuration_aliases"][name] = alias
    result["allocation_digest"] = digest(result)
    return result


def rescore(root: Path, model_key: str, folder: Path, commit: str) -> None:
    """不改输入与原始预测；统一重评全部方案和八组映射控制。"""
    gate_path = root / FORMAL / model_key / "gate.json"
    mapping_path = root / FORMAL / model_key / "mapping.json"
    gate, mapping = read(gate_path), read(mapping_path)
    rows = [{"id": row["id"], "key": row["key"], "answer": row["answer"],
             "schemes": {name: score_answer(value["prediction"], row["answer"])
                         for name, value in row["schemes"].items()}} for row in gate["rows"]]
    controls = []
    for row in mapping["rows"]:
        cases = row["cases"]
        scores = {name: score_answer(cases[name]["prediction"], cases[name]["answer"])
                  for name in ("original", "swapped")}
        controls.append({"id": row["id"], "key": row["key"], "scores": scores,
                         "both_content_follow": all(s["content_correct"] for s in scores.values())})
    save(folder / "rescoring.json", {
        "commit": commit, "model": model_key, "source_gate": {"path": str(
            gate_path.relative_to(root).as_posix()), "sha256": sha(gate_path)},
        "source_mapping": {"path": mapping_path.relative_to(root).as_posix(),
                           "sha256": sha(mapping_path)}, "rows": rows, "mapping_rows": controls,
        "means": {name: {key: float(np.mean([row["schemes"][name][key] for row in rows]))
                          for key in ("content_correct", "format_compliant", "formatted_correct")}
                  for name in gate["means"]},
    })


def source_for_question(
    root: Path, model: Any, item: dict[str, Any], model_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """保留同一 BF16 prefill 和头前 hidden；重复与精度条件均读取这份源值。"""
    path = root / RUNTIME / "sources" / model_key / (item["id"] + ".pt")
    if path.exists():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        if stored["input_sha256"] != item["models"][model_key]["sha256"]:
            raise RuntimeError("固定源值的输入身份不一致")
        source = stored["source"]
        for key, value in source.items():
            source[key] = [x.to("cuda") for x in value] if isinstance(value, list) else \
                value.to("cuda") if isinstance(value, torch.Tensor) else value
    else:
        captured = []
        hook = model.lm_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0][:, -1].detach())
        )
        try:
            source = capture_source(model, read_input(root, item, model_key))
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("没有捕获唯一 prefill 输出头输入")
        source["last_hidden"] = captured[0]
        cpu_source = {key: [x.cpu() for x in value] if isinstance(value, list) else
                      value.cpu() if isinstance(value, torch.Tensor) else value
                      for key, value in source.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"source": cpu_source,
                    "input_sha256": item["models"][model_key]["sha256"]}, path)
        del cpu_source
    return source, {"path": path.relative_to(root).as_posix(), "sha256": sha(path),
                    "bytes": path.stat().st_size, "prefill_dtype": "bfloat16"}


def scheme_config(
    name: str, spec: dict[str, Any], plan: dict[str, Any], params: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], int]:
    """选择冻结的量化参数；精度条件不改变位宽、RoPE 或残留。"""
    if name == "bf16":
        return None, {}, 0
    if name == "kv8":
        return _config(plan["kv8_reference"]), {}, 0
    fmt = plan["formats"][spec["format"]]
    config = {**_config(fmt["representation"]), "layer_bits": spec["layer_bits"]}
    return config, params if config["key_mode"] == "per_channel" else {}, fmt["residual_bits"]


def measure_repeat(
    model: Any, tokenizer: Any, source: dict[str, Any], names: dict[str, Any],
    plan: dict[str, Any], params: dict[str, Any], mode: str, weight: torch.Tensor,
    answer: list[int], positions: list[int], eos: set[int], repeat: int,
    prior: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """首次记录实际贪心输出；严格正路径与贪心一致时直接保留同一实际轨迹。"""
    metrics, logits, configurations = {}, {}, {}
    for name, spec in names.items():
        identity = (spec["format"], tuple(row["k"] for row in spec["layer_bits"])) \
            if spec else (name,)
        if identity in configurations:
            alias = configurations[identity]
            metrics[name], logits[name] = metrics[alias], logits[alias]
            continue
        configurations[identity] = name
        config, selected, h = scheme_config(name, spec, plan, params)
        values = trace(model, source, answer, mode, weight, config, selected, h)
        measured = path_metrics(values, answer, tokenizer, positions)
        if repeat == 0:
            if measured["all_strictly_positive"]:
                tokens = [row["greedy_id"] for row in measured["frames"]]
                generation_method = "same_positive_argmax_path"
            else:
                tokens = generate(model, source, eos, plan["generation"]["max_new_tokens"],
                                  mode, weight, config, selected, h)
                generation_method = "independent_greedy_replay"
            prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            generation = {"prediction": prediction, "token_ids": tokens,
                          "method": generation_method,
                          **score_answer(prediction, tokenizer.decode(answer[:-1]))}
        else:
            generation = prior[name]["generation"]
        metrics[name] = {**measured, "generation": generation}
        logits[name] = torch.from_numpy(values)
    return metrics, logits


def run(model_key: str, commit: str) -> None:
    """逐题恢复，固定输入和层名单；显式关闭 TF32 与 autocast，记录实际计算边界。"""
    root = project_root()
    design, plan = read(root / DESIGN), load_plan(root)
    if sha(root / design["base_plan"]) != design["base_plan_file_sha256"]:
        raise RuntimeError("原完整实验配置身份改变")
    if torch.is_autocast_enabled():
        raise RuntimeError("精度诊断禁止 autocast")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    free, total = torch.cuda.mem_get_info()
    if free < design["memory_estimate_gib"][model_key] * 2**30:
        raise RuntimeError("可用显存未达到本次含余量峰值估计")
    output = root / RUNTIME / "results" / model_key
    output.mkdir(parents=True, exist_ok=True)
    rescore(root, model_key, output, commit)
    dataset = read(root / design["dataset"])
    base_folder = root / FORMAL / model_key
    base, probe, gate = (read(base_folder / (key + ".json"))
                         for key in ("allocations", "probe", "gate"))
    geometry = read(root / MODELS[model_key] / "config.json")
    allocation = protected_allocations(base, probe, plan, geometry, design)
    allocation.update({"model": model_key, "commit": commit, "design_digest": digest(design)})
    save(output / "numerical_allocations.json", allocation)
    metadata = {"commit": commit, "model": model_key, "design_digest": digest(design),
                "code_sha256": code_sources(root), "device": torch.cuda.get_device_name(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "initial_free_bytes": free, "total_device_bytes": total,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "autocast": torch.is_autocast_enabled(),
                "bf16_reduced_precision_reduction":
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    save(output / "execution.json", metadata)
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    weight = model.lm_head.weight.detach().float()
    if weight.dtype != torch.float32 or model.lm_head.weight.dtype != torch.bfloat16:
        raise RuntimeError("输出头实际矩阵 dtype 不符合冻结设计")
    eos = _eos_ids(tokenizer, model_key == "llama")
    end = int(tokenizer.convert_tokens_to_ids(design["termination"][model_key]))
    if end not in eos:
        raise RuntimeError("冻结终止 token 不在当前停止集合")
    calibration = read(base_folder / "calibration.json")
    if sha(root / calibration["path"]) != calibration["sha256"]:
        raise RuntimeError("原离线参数实体身份改变")
    packed = torch.load(root / calibration["path"], map_location="cpu", weights_only=False)
    params = {"key_by_bits": {bits: [tuple(t.to("cuda") for t in tensors) for tensors in rows]
                               for bits, rows in packed["params"].items()}}
    params["key"] = params["key_by_bits"][8]
    all_names = {"bf16": {}, "kv8": {}, **allocation["allocations"]}
    diagnostic_names = {name: spec for name, spec in all_names.items()
                        if name in ("bf16", "kv8") or spec["format"] in design["precision_formats"]}
    single_layers = list(range(len(model.model.layers))) if model_key == "llama" else \
        allocation["protected"][design["single_layer_format"]]
    for layer in single_layers:
        diagnostic_names[f"single_layer_{layer}"] = {
            "format": design["single_layer_format"], "k4_layers": [layer],
            "layer_bits": layer_bits(len(model.model.layers), [layer]),
        }
    validation = [item for item in dataset["rows"] if item["role"] == "validation"]
    prior_rows = {row["id"]: row for row in gate["rows"]}
    with torch.inference_mode():
        for question_index, item in enumerate(validation):
            destination = output / "questions" / (item["id"] + ".json")
            if destination.exists():
                old = read(destination)
                if old["commit"] != commit or old["design_digest"] != digest(design):
                    raise RuntimeError("逐题恢复的提交或设计不同")
                continue
            torch.cuda.reset_peak_memory_stats()
            source, source_entity = source_for_question(root, model, item, model_key)
            record = item["models"][model_key]
            answer = record["answer_ids"] + [end]
            path = {"commit": commit, "design_digest": digest(design), "id": item["id"],
                    "key": item["key"], "answer": item["value"], "question_index": question_index,
                    "input_sha256": record["sha256"], "source_entity": source_entity,
                    "answer_ids": answer, "content_indices": record["content_indices"],
                    "precisions": {}}
            for mode in design["precisions"]:
                measured = []
                previous_logits, previous_entity = None, None
                count = design["repeats_initial"]
                repeat = 0
                while repeat < count:
                    names = {**all_names, **diagnostic_names} if mode == "current" and repeat == 0 \
                        else diagnostic_names
                    metrics, values = measure_repeat(
                        model, tokenizer, source, names, plan, params, mode, weight, answer,
                        record["content_indices"], eos, repeat,
                        measured[0]["schemes"] if measured else None,
                    )
                    identical = previous_logits is not None and all(
                        torch.equal(values[name], previous_logits[name]) for name in values
                    )
                    if repeat == 1 and not identical:
                        count = design["repeats_if_nonidentical"]
                    if identical:
                        entity = {**previous_entity, "same_tensor_bytes_as_previous": True}
                    else:
                        raw_path = root / RUNTIME / "logits" / model_key / item["id"] / \
                            f"{mode}-repeat-{repeat}.pt"
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({"targets": answer, "logits": values,
                                    "input_sha256": record["sha256"]}, raw_path)
                        entity = {"path": raw_path.relative_to(root).as_posix(),
                                  "sha256": sha(raw_path), "bytes": raw_path.stat().st_size,
                                  "dtype": "float32"}
                    measured.append({"repeat": repeat, "schemes": metrics,
                                     "identical_to_previous": identical, "logit_entity": entity})
                    previous_logits, previous_entity = values, entity
                    repeat += 1
                path["precisions"][mode] = measured
                print("precision", model_key, question_index, mode, count, flush=True)
                del previous_logits
            current = path["precisions"]["current"][0]["schemes"]
            historical = {}
            for name, old in prior_rows[item["id"]]["schemes"].items():
                measured = current[name]
                historical[name] = {
                    "same_prediction": measured["generation"]["prediction"] == old["prediction"],
                    "content_nll_difference": measured["content_nll"] - old["nll"],
                    "same_content_nll": bool(np.isclose(measured["content_nll"], old["nll"],
                                                        rtol=2e-12, atol=2e-14)),
                    "old_prediction_score": score_answer(old["prediction"], item["value"]),
                }
            path["historical_device_comparison"] = {
                "historical_gate_sha256": sha(base_folder / "gate.json"),
                "historical_device": gate["device"], "schemes": historical,
                "role": "separate_prefill_capture_device_comparison_not_precision_envelope",
            }
            path["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            path["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            save(destination, path)
            del source, path
            gc.collect()
            torch.cuda.empty_cache()
            print("QUESTION_COMPLETE", model_key, question_index + 1, len(validation), flush=True)
    del model, weight, params
    gc.collect()
    torch.cuda.empty_cache()
    print("NUMERICS_MODEL_COMPLETE", model_key, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "llama"), required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.model, args.commit)
