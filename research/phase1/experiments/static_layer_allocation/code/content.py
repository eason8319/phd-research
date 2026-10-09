"""独立映射的内容位置探针、完整分配验证及单层 logit 叠加诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import AutoTokenizer

from kvstudy.attention import attention_fp32
from kvstudy.paths import project_root
from experiments.generation_error_sources.code.quantized_generation import (
    _config, _decode_step, _inputs, _layer_params, _prefill_cache, _quantize, _recent,
    _rotate_key,
)
from experiments.generation_error_sources.code.short_baseline import _eos_ids
from experiments.static_layer_allocation.code.allocations import build_choices, cost, layer_bits
from experiments.static_layer_allocation.code.content_metrics import (
    additive_metrics, question_metrics, validation_summary,
)
from experiments.static_layer_allocation.code.metrics import grouped_rse, rank_agreement, ranking
from experiments.static_layer_allocation.code.probe import _reconstruction
from experiments.static_layer_allocation.code.protocol import code_sources, digest, load_plan
from phase1.representation import (
    MODELS, _capture, _empty_minmax, _fit_minmax, _render_gsm, _update_minmax,
)
from phase1.runner import cache_lists, decode_step, load_model, modeling_module, prefill

SCHEMA = "independent-retrieval-content-allocation"


def save(path: Path, payload: dict[str, Any]) -> None:
    """原子写入 UTF-8、LF 的正式记录；严格拒绝非有限指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_input(root: Path, item: dict[str, Any], model: str, paired: bool = False) -> list[int]:
    """核对冻结全文身份后读取实际输入 token。"""
    record = item["paired_models" if paired else "models"][model]
    text = (root / record["path"]).read_text("utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["sha256"]:
        raise RuntimeError("冻结检索全文身份改变")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model])
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != record["tokens"]:
        raise RuntimeError("冻结检索 token 长度改变")
    return ids


def capture_source(model: Any, ids: list[int]) -> dict[str, Any]:
    """一次高精度 prefill 捕获主存源值，后续方案各自建立缓存。"""
    module = modeling_module(model)
    original = module.apply_rotary_pos_emb
    pre_keys = []

    def capture(query: torch.Tensor, key: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        pre_keys.append(key.detach())
        return original(query, key, *args, **kwargs)

    module.apply_rotary_pos_emb = capture
    tensor = torch.tensor([ids], device=model.model.embed_tokens.weight.device)
    try:
        output = prefill(model, tensor)
    finally:
        module.apply_rotary_pos_emb = original
    keys, values = cache_lists(output.past_key_values)
    logits = output.logits.reshape(-1, output.logits.shape[-1])[0]
    return {"input_ids": tensor, "keys": keys, "values": values, "pre_keys": pre_keys,
            "logits": logits.detach(), "token": int(logits.argmax())}


def reference_trace(
    model: Any, source: dict[str, Any], answer: list[int], positions: list[int],
    callback: Callable[[dict[str, Any], int], None] | None = None,
) -> np.ndarray:
    """产生 y_t 的 query 使用 y_<t；先回调测量，再由下一步喂入 y_t。"""
    keys, values = list(source["keys"]), list(source["values"])
    pre_keys = [value[0] for value in source["pre_keys"]]
    saved = []
    for index in range(len(answer)):
        if index == 0:
            logits = source["logits"]
        else:
            events = []

            def observe(event: dict[str, Any]) -> None:
                if "layer" in event:
                    events.append(event)

            logits, keys, values = decode_step(
                model, keys, values, answer[index - 1],
                source["input_ids"].shape[1] + index - 1,
                observer=observe, consume_cache=True,
            )
            pre_keys = [torch.cat([old, row["key_pre"][0]], dim=1)
                        for old, row in zip(pre_keys, events)]
            if callback is not None and index in positions:
                callback({"pre": pre_keys, "post": [x[0] for x in keys],
                          "value": [x[0] for x in values],
                          "query": [row["query"][0, :, 0] for row in events]}, index)
        if index in positions:
            saved.append(logits.detach().float().cpu().numpy().reshape(-1))
    return np.stack(saved)


def local_layer(
    model: Any, frame: dict[str, Any], layer: int, config: dict[str, Any],
    calibrated: dict[str, Any], h: int, v_four: torch.Tensor | None = None,
) -> dict[str, Any]:
    """共同源值量化 K8 与 K4，固定 query、输入及 V4，保持残留和格式一致。"""
    key_ref, value_ref = frame["post"][layer].float(), frame["value"][layer].float()
    query = frame["query"][layer].float()
    key_source = frame[config["rope"]][layer].float()
    keys = {}
    invalid = 0
    rope_positions = torch.arange(key_ref.shape[1], device=key_ref.device).view(1, -1)
    for bits in (8, 4):
        restored, bad = _quantize(
            key_source, bits, config["key_mode"], _layer_params(calibrated, "key", layer, bits)
        )
        invalid += int(bad)
        keys[bits] = _rotate_key(model, restored, rope_positions) \
            if config["rope"] == "pre" else restored
    reconstruction = _reconstruction(key_ref, keys[4])
    if v_four is None:
        v_four, bad = _quantize(value_ref, 4, "per_token", None)
        invalid += int(bad)
    if h:
        start = max(0, key_ref.shape[1] - 128)
        recent_key, bad_key = _recent(key_ref[:, start:], h)
        recent_value, bad_value = _recent(value_ref[:, start:], h)
        invalid += int(bad_key) + int(bad_value)
        for key in keys.values():
            key[:, start:] = recent_key
        v_four[:, start:] = recent_value
    if invalid:
        raise RuntimeError("内容位置局部量化出现无效组")
    reference = attention_fp32(query, key_ref, value_ref)[0].cpu().numpy()
    eight = attention_fp32(query, keys[8], v_four)[0].cpu().numpy()
    four = attention_fp32(query, keys[4], v_four)[0].cpu().numpy()
    sensitivity, zeros = grouped_rse(reference, four, eight, key_ref.shape[0])
    return {"layer": layer, "sensitivity": sensitivity, "reconstruction": reconstruction,
            "k8v4_error": grouped_rse(reference, eight, reference, key_ref.shape[0])[0],
            "k4v4_error": grouped_rse(reference, four, reference, key_ref.shape[0])[0],
            "zero_heads": zeros, "invalid_groups": invalid}


def quantized_trace(
    model: Any, source: dict[str, Any], answer: list[int], positions: list[int],
    config: dict[str, Any], calibration: dict[str, Any], h: int,
    local_callback: Callable[[dict[str, Any], int], None] | None = None,
) -> np.ndarray:
    """目标序列完全一致；每方案的量化缓存跨 teacher forcing 步独立更新。"""
    _, keys, values, read_keys, read_values, invalid = _prefill_cache(
        model, source["input_ids"], config, calibration, len(answer), h,
        prefill_source=source,
    )
    if invalid:
        raise RuntimeError("量化 prefill 出现无效组")
    saved = []
    raw_keys = [key[0] for key in source["keys"]]
    raw_values = [value[0] for value in source["values"]]
    for index in range(len(answer)):
        if index == 0:
            logits = source["logits"]
        else:
            captured = []
            events = []

            def observe(event: dict[str, Any]) -> None:
                if "logits" in event:
                    captured.append(event["logits"])
                elif local_callback is not None:
                    events.append(event)

            _, bad = _decode_step(
                model, keys, values, read_keys, read_values, answer[index - 1],
                source["input_ids"].shape[1] + index - 1, config, calibration, h,
                observer=observe,
            )
            if bad or len(captured) != 1:
                raise RuntimeError("量化 decode 无效或未捕获唯一预测")
            logits = captured[0]
            if local_callback is not None:
                raw_keys = [torch.cat([old, row["key_post"][0]], dim=1)
                            for old, row in zip(raw_keys, events)]
                raw_values = [torch.cat([old, row["value"][0]], dim=1)
                              for old, row in zip(raw_values, events)]
                if index in positions:
                    local_callback({"post": raw_keys, "value": raw_values,
                                    "query": [row["query"][0, :, 0] for row in events],
                                    "v4": [value[0, :, :raw_keys[0].shape[1]]
                                           for value in read_values]}, index)
        if index in positions:
            saved.append(logits.detach().float().cpu().numpy().reshape(-1))
    return np.stack(saved)


def generate(
    model: Any, source: dict[str, Any], eos: set[int], limit: int,
    config: dict[str, Any] | None = None, calibration: dict[str, Any] | None = None, h: int = 0,
) -> list[int]:
    """贪心自由生成，完整答案直接评分；不读取正确前缀。"""
    if config is None:
        keys, values = list(source["keys"]), list(source["values"])
    else:
        first, keys, values, read_keys, read_values, bad = _prefill_cache(
            model, source["input_ids"], config, calibration or {}, limit, h,
            prefill_source=source,
        )
        if bad or first != source["token"]:
            raise RuntimeError("自由生成缓存构建失败")
    produced = [source["token"]]
    for step in range(limit - 1):
        if produced[-1] in eos:
            break
        if config is None:
            logits, keys, values = decode_step(
                model, keys, values, produced[-1], source["input_ids"].shape[1] + step,
                consume_cache=True,
            )
            token = int(logits.argmax())
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("自由生成的高精度 logits 非有限")
        else:
            def check_logits(event: dict[str, Any]) -> None:
                if "logits" in event and not bool(torch.isfinite(event["logits"]).all()):
                    raise RuntimeError("自由生成的量化 logits 非有限")

            token, bad = _decode_step(
                model, keys, values, read_keys, read_values, produced[-1],
                source["input_ids"].shape[1] + step, config, calibration or {}, h,
                observer=check_logits,
            )
            if bad or token is None:
                raise RuntimeError("自由生成出现无效量化组")
        produced.append(token)
    return produced


def fit_calibration(
    root: Path, model: Any, tokenizer: Any, model_key: str,
    dataset: dict[str, Any], commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """只用校准折拟合离线 K4/K8；验证折不进入范围、排名或映射配对选择。"""
    geometry = model.config
    accumulator = _empty_minmax(len(model.model.layers), geometry.num_key_value_heads,
                                geometry.hidden_size // geometry.num_attention_heads)
    identities = []
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    folded = {row["id"] for row in gsm["c"] if row["fold"] == 0}
    for item in _inputs(root, model_key, "gsm8k", "c"):
        if item["id"] not in folded:
            continue
        ids = _render_gsm(tokenizer, item["question"], model_key == "llama")
        blob = _capture(model, ids)
        _update_minmax(accumulator, blob["pre"])
        identities.append({"id": item["id"], "kind": "math",
                           "source_sha256": item["question_sha256"]})
        del blob
    for item in dataset["rows"]:
        if item["role"] != "calibration":
            continue
        ids = read_input(root, item, model_key)
        source = capture_source(model, ids)
        _update_minmax(accumulator, [key[0].float().cpu() for key in source["pre_keys"]])

        def update(frame: dict[str, Any], _position: int) -> None:
            _update_minmax(accumulator, [value.float().cpu() for value in frame["pre"]])

        rec = item["models"][model_key]
        reference_trace(model, source, rec["answer_ids"], rec["content_indices"], update)
        identities.append({"id": item["id"], "kind": "retrieval",
                           "source_sha256": rec["sha256"]})
        del source
    params = {bits: _fit_minmax(accumulator, bits) for bits in (4, 8)}
    path = root / f"research/.runtime/static-key-content/calibration/{model_key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"commit": commit, "dataset_digest": dataset["identity_digest"],
                "calibration_inputs": identities, "params": params}, path)
    record = {"path": path.relative_to(root).as_posix(),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "commit": commit,
              "input_identity": identities, "math_n": len(folded), "retrieval_n": 16,
              "source": "calibration_fold_only_prefill_and_correct_prefix"}
    calibrated = {"key_by_bits": {
        bits: [tuple(tensor.to("cuda") for tensor in row) for row in fitted]
        for bits, fitted in params.items()
    }}
    calibrated["key"] = calibrated["key_by_bits"][8]
    return calibrated, record


def probe_layers(
    model: Any, source: dict[str, Any], record: dict[str, Any], plan: dict[str, Any],
    calibrated: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """各格式在同一高精度正确前缀轨迹上测量，先组、再位置、再题等权。"""
    frames = {name: [] for name in plan["formats"]}

    def measure(frame: dict[str, Any], position: int) -> None:
        for name, spec in plan["formats"].items():
            config = _config(spec["representation"])
            params = calibrated if config["key_mode"] == "per_channel" else {}
            frames[name].append({"answer_index": position,
                                 "query_position": source["input_ids"].shape[1] + position - 1,
                                 "layers": [local_layer(model, frame, layer, config, params,
                                                        spec["residual_bits"])
                                            for layer in range(len(model.model.layers))]})

    reference = reference_trace(model, source, record["answer_ids"], record["content_indices"],
                                measure)
    aggregated = {}
    for name, positions in frames.items():
        layer_rows = []
        for layer in range(len(model.model.layers)):
            values = [frame["layers"][layer] for frame in positions]
            layer_rows.append({"layer": layer, "invalid_groups": 0,
                               "zero_heads": sum(row["zero_heads"] for row in values),
                               **{key: float(np.mean([row[key] for row in values]))
                                  for key in ("sensitivity", "reconstruction", "k8v4_error",
                                              "k4v4_error")}})
        aggregated[name] = {"layers": layer_rows, "first_content_layers": positions[0]["layers"],
                            "frames": positions}
    return reference, aggregated


def freeze_allocations(
    root: Path, model: Any, model_key: str, probe: dict[str, Any], plan: dict[str, Any],
) -> dict[str, Any]:
    """校准排名只用 fold=0，六策略按冻结预算生成，相同配置复用测量。"""
    seeds = json.loads((root / plan["random_seeds_source"]).read_text("utf-8"))[
        plan["random_seeds_key"]]
    layers = len(model.model.layers)
    geometry = model.config.to_dict()
    specs = {}
    groups = {}
    permutations = {}
    for name, format_spec in plan["formats"].items():
        config = _config(format_spec["representation"])
        choices, order = build_choices(probe["formats"][name]["layers"],
                                      plan["models"][model_key]["k4_layers"], seeds)
        groups[name] = {}
        permutations[name] = order
        canonical = {}
        for budget, strategies in choices.items():
            groups[name][budget] = []
            for strategy, selected in strategies.items():
                key = f"{name}_{budget}_{strategy}"
                bits = layer_bits(layers, selected)
                canonical.setdefault(digest(bits), key)
                specs[key] = {"format": name, "strategy": strategy, "k4_layers": selected,
                              "r": len(selected), "layer_bits": bits,
                              "alias": canonical[digest(bits)], "actual_int8_units":
                              layers - len(selected),
                              "p_int8": (layers - len(selected)) / (2 * layers),
                              "cost": {str(n): cost(bits, geometry, config,
                                                     format_spec["residual_bits"], n)
                                       for n in plan["cost_lengths"]}}
                groups[name][budget].append(key)
        for control, selected in (("k8v4", []), ("kv4", list(range(layers)))):
            key = name + "_" + control
            bits = layer_bits(layers, selected)
            canonical.setdefault(digest(bits), key)
            specs[key] = {"format": name, "strategy": control, "k4_layers": selected,
                          "r": len(selected), "layer_bits": bits, "alias": canonical[digest(bits)],
                          "cost": {str(n): cost(bits, geometry, config,
                                                 format_spec["residual_bits"], n)
                                   for n in plan["cost_lengths"]}}
    result = {"model": model_key, "random_seeds": seeds, "budget_groups": groups,
              "random_permutations": permutations, "allocations": specs,
              "probe_sha256": digest(probe), "rank_calibration_ids": probe["calibration_ids"]}
    result["allocation_digest"] = digest(result)
    return result


def mapping_checks(
    root: Path, model: Any, tokenizer: Any, model_key: str, dataset: dict[str, Any],
    plan: dict[str, Any], eos: set[int],
) -> dict[str, Any]:
    """固定八对检查答案随映射变化；旧数据删去目标行后直接检查捷径行为。"""
    rows = []
    legacy = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
    old_by_id = {row["id"]: row for row in legacy}
    for item in dataset["rows"]:
        if "paired_value" not in item:
            continue
        cases = {}
        for paired in (False, True):
            ids = read_input(root, item, model_key, paired)
            source = capture_source(model, ids)
            tokens = generate(model, source, eos, plan["generation"]["max_new_tokens"])
            prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            answer = item["paired_value"] if paired else item["value"]
            cases["swapped" if paired else "original"] = {
                "answer": answer, "prediction": prediction, "exact": prediction == answer,
                "input_sha256": item["paired_models" if paired else "models"][model_key]["sha256"],
                "tokens": len(ids),
            }
            del source
        old = old_by_id[item["id"]]["models"][model_key]
        path = root / ("research/.runtime/data/phase1-text/retrieval/" + model_key + "/"
                       + item["id"] + ".txt")
        text = path.read_text("utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != old["sha256"]:
            raise RuntimeError("旧检索控制的全文身份改变")
        needle = f"The register {old['key']} stores {old['value']}."
        if text.count(needle) != 1:
            raise RuntimeError("旧数据目标行不唯一")
        text = text.replace(needle, "Background register has no assigned value.")
        control_path = root / ("research/.runtime/static-key-content/inputs/" + model_key
                               + "/" + item["id"] + "-legacy-target-removed.txt")
        control_path.write_text(text, encoding="utf-8", newline="\n")
        ids = tokenizer.encode(text, add_special_tokens=False)
        source = capture_source(model, ids)
        tokens = generate(model, source, eos, plan["generation"]["max_new_tokens"])
        prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        cases["legacy_target_removed"] = {
            "key": old["key"], "original_answer": old["value"], "prediction": prediction,
            "copies_old_answer": prediction == old["value"], "tokens": len(ids),
            "path": control_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        del source
        rows.append({"id": item["id"], "key": item["key"], "cases": cases,
                     "both_follow_mapping": cases["original"]["exact"]
                     and cases["swapped"]["exact"],
                     "answer_changed": cases["original"]["prediction"]
                     != cases["swapped"]["prediction"]})
        print("mapping", model_key, item["id"], len(rows), flush=True)
    return {"n_pairs": len(rows), "rows": rows,
            "both_follow_mapping": sum(row["both_follow_mapping"] for row in rows),
            "original_exact": sum(row["cases"]["original"]["exact"] for row in rows),
            "swapped_exact": sum(row["cases"]["swapped"]["exact"] for row in rows),
            "legacy_target_removed_copies": sum(
                row["cases"]["legacy_target_removed"]["copies_old_answer"] for row in rows
            )}


def measure_question(
    root: Path, model: Any, tokenizer: Any, model_key: str, item: dict[str, Any],
    allocation: dict[str, Any], plan: dict[str, Any], calibrated: dict[str, Any], eos: set[int],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """同题各配置独立持续缓存；Llama online post 保存全部单层及组合 logits。"""
    ids = read_input(root, item, model_key)
    source = capture_source(model, ids)
    rec = item["models"][model_key]
    positions, answer = rec["content_indices"], rec["answer_ids"]
    targets = [answer[index] for index in positions]
    reference = reference_trace(model, source, answer, positions)
    logits = {"bf16": reference}
    predictions = {}
    predictions["bf16"] = tokenizer.decode(
        generate(model, source, eos, plan["generation"]["max_new_tokens"]),
        skip_special_tokens=True,
    ).strip()
    specs = {"kv8": {"format": None, "alias": "kv8"}, **allocation["allocations"]}
    matched_frames = []
    diagnostic_format = plan["single_layer_diagnostic"]["format"]
    diagnosed = model_key == plan["single_layer_diagnostic"]["model"]
    for name, spec in specs.items():
        if spec["alias"] != name:
            logits[name] = logits[spec["alias"]]
            predictions[name] = predictions[spec["alias"]]
            continue
        if name == "kv8":
            config, params, h = _config(plan["kv8_reference"]), {}, 0
        else:
            format_spec = plan["formats"][spec["format"]]
            config = {**_config(format_spec["representation"]),
                      "layer_bits": spec["layer_bits"]}
            params = calibrated if config["key_mode"] == "per_channel" else {}
            h = format_spec["residual_bits"]

        def matched_local(frame: dict[str, Any], position: int) -> None:
            matched_frames.append({"answer_index": position, "layers": [
                local_layer(model, frame, layer, config, {}, 0, frame["v4"][layer])
                for layer in range(len(model.model.layers))
            ]})

        callback = matched_local if diagnosed and name == diagnostic_format + "_k8v4" else None
        logits[name] = quantized_trace(
            model, source, answer, positions, config, params, h, callback
        )
        produced = generate(model, source, eos, plan["generation"]["max_new_tokens"],
                            config, params, h)
        predictions[name] = tokenizer.decode(produced, skip_special_tokens=True).strip()
    schemes = {}
    for name, values in logits.items():
        format_name = specs.get(name, {}).get("format")
        baseline = logits[format_name + "_k8v4"] if format_name else None
        schemes[name] = {**question_metrics(values, targets, reference, baseline),
                         "prediction": predictions[name],
                         "free_exact": int(predictions[name] == item["value"])}
    diagnostic = None
    if diagnosed:
        singles = {}
        config = _config(plan["formats"][diagnostic_format]["representation"])
        baseline = logits[diagnostic_format + "_k8v4"]
        baseline_loss = question_metrics(baseline, targets, reference)["nll"]
        for layer in range(len(model.model.layers)):
            singles[layer] = quantized_trace(
                model, source, answer, positions,
                {**config, "layer_bits": layer_bits(len(model.model.layers), [layer])}, {}, 0,
            )
        single_metrics = [
            {"layer": layer, **question_metrics(values, targets, reference, baseline)}
            for layer, values in singles.items()
        ]
        additive = {
            name: additive_metrics(baseline, singles, logits[name], spec["k4_layers"],
                                   targets, reference)
            for name, spec in allocation["allocations"].items()
            if spec["format"] == diagnostic_format and spec["strategy"] not in ("k8v4", "kv4")
        }
        stored = {name: torch.from_numpy(values.copy()) for name, values in logits.items()
                  if name == "bf16" or name.startswith(diagnostic_format + "_")}
        stored.update({f"single_layer_{layer}": torch.from_numpy(values.copy())
                       for layer, values in singles.items()})
        path = root / ("research/.runtime/static-key-content/logits/" + model_key
                       + "/" + item["id"] + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_sha256": rec["sha256"], "content_indices": positions,
                    "targets": targets, "logits": stored}, path)
        diagnostic = {"id": item["id"], "baseline_nll": baseline_loss,
                      "single_layers": single_metrics, "matched_local_frames": matched_frames,
                      "additive": additive,
                      "logits_source": {"path": path.relative_to(root).as_posix(),
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "bytes": path.stat().st_size, "dtype": "float32"}}
    return {"id": item["id"], "key": item["key"], "answer": item["value"],
            "input_sha256": rec["sha256"], "input_tokens": len(ids),
            "content_indices": positions, "content_target_ids": targets,
            "query_positions": [len(ids) + index - 1 for index in positions],
            "role": item["role"], "schemes": schemes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}, diagnostic


def run(model_key: str, commit: str) -> None:
    """依次完成配对核验、校准、冻结排名与独立验证，逐题持久化并保留失败。"""
    root = project_root()
    plan = load_plan(root)
    dataset = json.loads((root / plan["dataset"]).read_text("utf-8"))
    if dataset["protocol"] != plan["input_protocol"] or digest(dataset["rows"]) != \
            dataset["identity_digest"]:
        raise RuntimeError("修正数据协议或身份不一致")
    folder = root / ("research/.runtime/static-key-content/measurements/" + model_key)
    folder.mkdir(parents=True, exist_ok=True)
    free, total = torch.cuda.mem_get_info()
    estimate = plan["memory_estimate_gib"][model_key] * 2**30
    if free < estimate:
        raise RuntimeError(f"该模型峰值估计及余量为 {estimate} 字节，实际空闲 {free}")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    model = load_model(str(root / MODELS[model_key]))
    if len(model.model.layers) != plan["models"][model_key]["layers"]:
        raise RuntimeError("实际模型层数与冻结预算不符")
    eos = _eos_ids(tokenizer, model_key == "llama")
    provenance = {
        "schema": SCHEMA, "protocol": plan["input_protocol"], "model": model_key,
        "commit": commit, "plan_sha256": digest(plan), "dataset_digest": dataset["identity_digest"],
        "dataset_sha256": hashlib.sha256((root / plan["dataset"]).read_bytes()).hexdigest(),
        "code_sha256": code_sources(root),
        "model_config_sha256": digest(model.config.to_dict()),
        "model_manifest_sha256": hashlib.sha256((root / (
            "research/manifests/models/" + Path(MODELS[model_key]).name + ".source.json"
        )).read_bytes()).hexdigest(),
        "device": {"node": os.uname().nodename, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                   "gpu": torch.cuda.get_device_name(), "free_before_load": free,
                   "total_bytes": total, "estimated_peak_with_margin_bytes": estimate},
    }
    with torch.inference_mode():
        mapping_path = folder / "mapping.json"
        if not mapping_path.exists():
            save(mapping_path, {**provenance, **mapping_checks(
                root, model, tokenizer, model_key, dataset, plan, eos
            )})
        calibration_path = folder / "calibration.json"
        if calibration_path.exists():
            calibration_record = json.loads(calibration_path.read_text("utf-8"))
            packed = torch.load(root / calibration_record["path"], weights_only=False,
                                map_location="cpu")
            if packed["dataset_digest"] != dataset["identity_digest"]:
                raise RuntimeError("离线校准恢复实体与输入不同")
            calibrated = {"key_by_bits": {bits: [tuple(t.to("cuda") for t in row)
                                                 for row in values]
                                          for bits, values in packed["params"].items()}}
            calibrated["key"] = calibrated["key_by_bits"][8]
        else:
            calibrated, calibration_record = fit_calibration(
                root, model, tokenizer, model_key, dataset, commit
            )
            calibration_record = {**provenance, **calibration_record}
            save(calibration_path, calibration_record)
        calibration_rows = [item for item in dataset["rows"] if item["role"] == "calibration"]
        progress_path = folder / "probe_progress.json"
        if progress_path.exists():
            prior = json.loads(progress_path.read_text("utf-8"))
            if prior["plan_sha256"] != provenance["plan_sha256"] or \
                    prior["code_sha256"] != provenance["code_sha256"]:
                raise RuntimeError("内容探针恢复源码或协议改变")
            samples = prior["samples"]
        else:
            samples = []
        for item in calibration_rows[len(samples):]:
            source = capture_source(model, read_input(root, item, model_key))
            _, measured = probe_layers(model, source, item["models"][model_key], plan, calibrated)
            samples.append({"id": item["id"], "role": item["role"],
                            "input_sha256": item["models"][model_key]["sha256"],
                            "formats": measured})
            save(progress_path, {**provenance, "samples": samples})
            del source
            print("probe", model_key, item["id"], len(samples), flush=True)
        formats = {}
        for name in plan["formats"]:
            rows = [{"layer": layer, "invalid_groups": 0,
                     **{key: float(np.mean([item["formats"][name]["layers"][layer][key]
                                            for item in samples]))
                        for key in ("sensitivity", "reconstruction", "k8v4_error", "k4v4_error")}}
                    for layer in range(len(model.model.layers))]
            formats[name] = {**plan["formats"][name], "layers": rows,
                             "rankings": {key: ranking(rows, key, descending=True)
                                          for key in ("sensitivity", "reconstruction")}}
        probe = {**provenance, "calibration_source": calibration_record,
                 "calibration_ids": [row["id"] for row in samples], "n": len(samples),
                 "formats": formats, "samples": samples,
                 "aggregation": "valid_kv_query_groups_then_content_tokens_then_questions",
                 "reconstruction_location": "post_rope_main_before_residual_override"}
        save(folder / "probe.json", probe)
        allocation = {**provenance, **freeze_allocations(root, model, model_key, probe, plan)}
        save(folder / "allocations.json", allocation)
        validation = [item for item in dataset["rows"] if item["role"] == "validation"]
        gate_progress = folder / "gate_progress.json"
        if gate_progress.exists():
            prior = json.loads(gate_progress.read_text("utf-8"))
            if (prior["allocation_digest"] != allocation["allocation_digest"]
                    or prior["code_sha256"] != provenance["code_sha256"]):
                raise RuntimeError("验证恢复时冻结位宽表或源码改变")
            rows, diagnostics = prior["rows"], prior["diagnostics"]
        else:
            rows, diagnostics = [], []
        for item in validation[len(rows):]:
            torch.cuda.reset_peak_memory_stats()
            row, diagnosis = measure_question(
                root, model, tokenizer, model_key, item, allocation, plan, calibrated, eos
            )
            rows.append(row)
            if diagnosis is not None:
                diagnostics.append(diagnosis)
            save(gate_progress, {**provenance, "allocation_digest": allocation["allocation_digest"],
                                 "rows": rows, "diagnostics": diagnostics})
            print("validation", model_key, item["id"], len(rows), flush=True)
        save(folder / "gate.json", {**provenance, "n": len(rows),
                                    "allocation_digest": allocation["allocation_digest"],
                                    "validation_ids": [row["id"] for row in rows], "rows": rows,
                                    "primary_metric": "question_mean_content_target_nll",
                                    **validation_summary(rows, allocation["budget_groups"],
                                                         allocation["random_seeds"]),
                                    "quality_pass_threshold": None, "v_plan_frozen": False})
        if diagnostics:
            layer_rows = []
            for layer in range(len(model.model.layers)):
                measures = [row["single_layers"][layer] for row in diagnostics]
                matched = [float(np.mean([frame["layers"][layer]["sensitivity"]
                                          for frame in row["matched_local_frames"]]))
                           for row in diagnostics]
                layer_rows.append({"layer": layer, "invalid_groups": 0,
                                   "matched_local_sensitivity": float(np.mean(matched)),
                                   **{key: float(np.mean([row[key] for row in measures]))
                                      for key in ("delta_nll", "kl_baseline", "kl_bf16")}})
            format_name = plan["single_layer_diagnostic"]["format"]
            reference_rank = probe["formats"][format_name]["rankings"]["sensitivity"]
            agreement = {
                "calibration_local_vs_" + key: rank_agreement(
                    reference_rank, ranking(layer_rows, key, descending=True), len(layer_rows)//2
                ) for key in ("delta_nll", "kl_baseline", "matched_local_sensitivity")
            }
            matched_rank = ranking(layer_rows, "matched_local_sensitivity", descending=True)
            agreement.update({
                "matched_local_vs_" + key: rank_agreement(
                    matched_rank, ranking(layer_rows, key, descending=True), len(layer_rows)//2
                ) for key in ("delta_nll", "kl_baseline")
            })
            save(folder / "single_layer.json", {
                **provenance, "format": format_name, "n": len(diagnostics),
                "baseline": format_name + "_k8v4", "layers": layer_rows,
                "rank_agreement": agreement, "rows": diagnostics,
                "additive_means": {name: {key: float(np.mean([
                    row["additive"][name][key] for row in diagnostics
                ])) for key in ("actual_to_additive_kl", "sum_single_layer_kl",
                                "kl_sum_minus_actual", "additive_nll_minus_actual",
                                "centered_logit_residual_energy_fraction")}
                                   for name in diagnostics[0]["additive"]},
            })
    print("CONTENT_EXPERIMENT_COMPLETE", model_key, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.model, args.commit)
