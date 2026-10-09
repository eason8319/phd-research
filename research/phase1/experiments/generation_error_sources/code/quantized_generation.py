"""真实生成与误差来源的 H=0 假量化质量运行；缓存反量化为 FP32，不代表压缩后的成本。"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from phase1.result_format import write_quality_result
from kvstudy.quantize import STATUS_INVALID, dequantize, fit_group_params, quantize_with_params
from experiments.generation_error_sources.code.short_baseline import MODELS, _eos_ids
from experiments.generation_error_sources.code.long_baseline import _samples
from experiments.generation_error_sources.code.sources import _protocol_matches, _runtime_path
from phase1.runner import (
    attention_from_batch,
    cache_lists,
    load_model,
    modeling_module,
    prefill,
    project_qkv,
)
from phase1.score import (
    answer_exact, gsm8k_exact, retrieval_exact, retrieval_number_exact, retrieval_numbers,
    token_f1,
)

G_C = 32
MAX_CONTEXT = 32768
RESIDUAL_TOKENS = 128


def _config(name: str) -> dict[str, Any]:
    """从位宽与表示交互的配置名解析 K/V 位宽、量化轴及 RoPE 顺序。"""
    parts = name.split("-")
    if len(parts) not in (3, 4) or parts[0] not in ("kv8", "kv4", "k8v4", "k4v8"):
        raise ValueError(f"未知配置：{name}")
    if parts[1] not in ("per_channel", "per_token") or parts[2] not in ("pre", "post"):
        raise ValueError(f"未知配置：{name}")
    if len(parts) == 4 and parts[3] != "vchannel":
        raise ValueError(f"未知配置：{name}")
    k_bits, v_bits = {
        "kv8": (8, 8), "kv4": (4, 4), "k8v4": (8, 4), "k4v8": (4, 8)
    }[parts[0]]
    return {
        "name": name,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "key_mode": parts[1],
        "rope": parts[2],
        "value_mode": "per_channel" if len(parts) == 4 else "per_token",
    }


def _calibration(
    root: Path, model_key: str, config: dict[str, Any], *, device: str = "cuda",
) -> dict[str, Any]:
    """仅离线轴读取全部 C 拟合的参数，并核对 C 的冻结 ID。"""
    if config["key_mode"] != "per_channel" and config["value_mode"] != "per_channel":
        return {}
    path = _runtime_path(
        root, f"generation-calibration-{model_key}.pt", f"e2-calibration-{model_key}.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    retrieval = json.loads((root / "research/phase1/splits/retrieval.json").read_text("utf-8"))
    expected = [item["id"] for item in gsm["c"]]
    expected.extend(item["id"] for item in retrieval if item["id"].startswith("synth-c-"))
    if (not _protocol_matches(payload["protocol"], "phase1-full-key-retrieval")
            or payload["ids"] != expected):
        raise RuntimeError("全部 C 的离线参数与当前冻结划分不一致")
    result: dict[str, Any] = {"commit": payload["commit"]}
    if config["key_mode"] == "per_channel":
        bits_needed = {config["k_bits"]}
        bits_needed.update(row["k"] for row in config.get("layer_bits", []))
        result["key_by_bits"] = {
            bits: [tuple(t.to(device) for t in row)
                   for row in payload["params"][f"k_{config['rope']}{bits}"]]
            for bits in bits_needed
        }
        result["key"] = result["key_by_bits"][config["k_bits"]]
    if config["value_mode"] == "per_channel":
        value = f"v{config['v_bits']}"
        result["value"] = [tuple(t.to(device) for t in row) for row in payload["params"][value]]
    return result


def _layer_params(
    calibrated: dict[str, Any], side: str, layer: int, bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """混合层按实际位宽读取离线参数；未校准的在线轴返回 None。"""
    by_bits = calibrated.get(f"{side}_by_bits")
    if by_bits is not None:
        return by_bits[bits][layer]
    params = calibrated.get(side)
    return params[layer] if params is not None else None


def _rotate_key(model: Any, key: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """量化后按模型 RoPE 旋转 [heads, seq, dim] Key。"""
    hidden = torch.zeros(
        1, key.shape[1], model.config.hidden_size, device=key.device, dtype=torch.bfloat16
    )
    cos, sin = model.model.rotary_emb(hidden, positions)
    packed = key.unsqueeze(0)
    _, rotated = modeling_module(model).apply_rotary_pos_emb(packed, packed, cos, sin)
    return rotated[0]


def _quantize(
    values: torch.Tensor,
    bits: int,
    mode: str,
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 FP32 还原值与无效组数，始终从高精度输入量化。"""
    if mode == "per_token":
        heads, seq, dim = values.shape
        if dim % G_C:
            raise RuntimeError("真实生成与误差来源模型的 head_dim 不能被 G_C 整除")
        grouped = values.reshape(heads, seq, dim // G_C, G_C)
        scale, offset, status = fit_group_params(grouped, bits)
        codes = quantize_with_params(grouped, scale, offset, status, bits)
        restored = dequantize(codes, scale, offset, status).reshape_as(values)
    else:
        if params is None:
            raise RuntimeError("逐通道量化缺少全部 C 校准参数")
        heads, seq, dim = values.shape
        scale, offset, status = params
        scale = scale.unsqueeze(1).expand(heads, seq, dim)
        offset = offset.unsqueeze(1).expand(heads, seq, dim)
        status = status.unsqueeze(1).expand(heads, seq, dim)
        grouped = values.unsqueeze(-1)
        codes = quantize_with_params(grouped, scale, offset, status, bits)
        restored = dequantize(codes, scale, offset, status).squeeze(-1)
    return restored, (status == STATUS_INVALID).sum()


def _recent(values: torch.Tensor, h_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """从高精度原值生成最近窗口的 H=8 或 H=16 副本。"""
    if h_bits == 16:
        return values.float(), torch.zeros((), dtype=torch.int64, device=values.device)
    if h_bits == 8:
        return _quantize(values, 8, "per_token", None)
    raise ValueError(f"未知残留位宽：{h_bits}")


def _layer_bits(config: dict[str, Any], index: int) -> tuple[int, int]:
    """静态层分配可按层覆盖 K/V 位宽；真实生成与误差来源保持全层同一配置。"""
    if "layer_bits" not in config:
        return config["k_bits"], config["v_bits"]
    row = config["layer_bits"][index]
    if row["k"] not in (4, 8) or row["v"] not in (4, 8):
        raise ValueError("静态层分配层位宽只允许 INT4 或 INT8")
    return row["k"], row["v"]


def _prefill_cache(
    model: Any,
    input_ids: torch.Tensor,
    config: dict[str, Any],
    calibrated: dict[str, Any],
    limit: int,
    h_bits: int,
    *,
    offload_layers: int = 0,
) -> tuple[
    int, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int
]:
    """高精度 prefill 后量化；前 offload_layers 层以同 dtype 暂存 CPU。"""
    pre_keys: list[torch.Tensor] = []
    module = modeling_module(model)
    original = module.apply_rotary_pos_emb
    if config["rope"] == "pre":
        def capture(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor, *args: object, **kwargs: object) -> Any:
            pre_keys.append(key.detach().cpu() if offload_layers else key.detach())
            return original(query, key, cos, sin, *args, **kwargs)
        module.apply_rotary_pos_emb = capture
    try:
        outputs = prefill(model, input_ids)
    finally:
        module.apply_rotary_pos_emb = original
    first = int(outputs.logits.reshape(-1, outputs.logits.shape[-1])[0].argmax())
    original_k, original_v = cache_lists(outputs.past_key_values)
    del outputs
    if config["rope"] == "pre" and len(pre_keys) != len(original_k):
        raise RuntimeError("未捕获所有层的 pre-RoPE Key")
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    read_keys: list[torch.Tensor] = []
    read_values: list[torch.Tensor] = []
    invalid = torch.zeros((), dtype=torch.int64, device=input_ids.device)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
    for index in range(len(original_k)):
        k_bits, v_bits = _layer_bits(config, index)
        source = pre_keys[index][0] if config["rope"] == "pre" else original_k[index][0]
        source = source.to(input_ids.device)
        k_params = _layer_params(calibrated, "key", index, k_bits)
        k_hat, bad_k = _quantize(source, k_bits, config["key_mode"], k_params)
        if config["rope"] == "pre":
            k_hat = _rotate_key(model, k_hat, positions)
        v_params = _layer_params(calibrated, "value", index, v_bits)
        v_hat, bad_v = _quantize(
            original_v[index][0], v_bits, config["value_mode"], v_params
        )
        capacity = int(input_ids.shape[1]) + limit
        key_buffer = torch.empty(
            (1, *k_hat.shape[:1], capacity, k_hat.shape[-1]),
            device=k_hat.device, dtype=torch.float32,
        )
        value_buffer = torch.empty_like(key_buffer)
        key_buffer[:, :, :input_ids.shape[1], :] = k_hat.unsqueeze(0)
        value_buffer[:, :, :input_ids.shape[1], :] = v_hat.unsqueeze(0)
        invalid += bad_k + bad_v
        if h_bits:
            start = max(0, int(input_ids.shape[1]) - RESIDUAL_TOKENS)
            high_k, bad_high_k = _recent(original_k[index][0][:, start:, :], h_bits)
            high_v, bad_high_v = _recent(original_v[index][0][:, start:, :], h_bits)
            ring_shape = (1, k_hat.shape[0], RESIDUAL_TOKENS, k_hat.shape[-1])
            low_key_ring = torch.empty(ring_shape, device=k_hat.device, dtype=torch.float32)
            low_value_ring = torch.empty_like(low_key_ring)
            ring_indices = torch.arange(start, int(input_ids.shape[1]),
                                        device=k_hat.device) % RESIDUAL_TOKENS
            low_key_ring[:, :, ring_indices, :] = k_hat[:, start:, :].unsqueeze(0)
            low_value_ring[:, :, ring_indices, :] = v_hat[:, start:, :].unsqueeze(0)
            key_buffer[:, :, start:input_ids.shape[1], :] = high_k.unsqueeze(0)
            value_buffer[:, :, start:input_ids.shape[1], :] = high_v.unsqueeze(0)
            keys.append(low_key_ring)
            values.append(low_value_ring)
            read_keys.append(key_buffer)
            read_values.append(value_buffer)
            invalid += bad_high_k + bad_high_v
        else:
            keys.append(key_buffer)
            values.append(value_buffer)
            read_keys.append(key_buffer)
            read_values.append(value_buffer)
        original_k[index] = None
        original_v[index] = None
        if pre_keys:
            pre_keys[index] = None
        if index < offload_layers:
            buffers = (keys, values, read_keys, read_values)
            copied = {}
            for group in buffers:
                if id(group[-1]) not in copied:
                    copied[id(group[-1])] = group[-1].cpu()
            for group in buffers:
                group[-1] = copied[id(group[-1])]
    return first, keys, values, read_keys, read_values, int(invalid.item())


def _write_residual_token(
    low_key_ring: torch.Tensor,
    low_value_ring: torch.Tensor,
    read_key: torch.Tensor,
    read_value: torch.Tensor,
    k_hat: torch.Tensor,
    v_hat: torch.Tensor,
    high_k: torch.Tensor,
    high_v: torch.Tensor,
    position: int,
) -> None:
    """先恢复滑出的低位宽值，再写当前高精度副本，避免全历史重复缓存。"""
    if position >= RESIDUAL_TOKENS:
        leaving = position - RESIDUAL_TOKENS
        slot = leaving % RESIDUAL_TOKENS
        read_key[:, :, leaving:leaving + 1, :] = low_key_ring[:, :, slot:slot + 1, :]
        read_value[:, :, leaving:leaving + 1, :] = low_value_ring[:, :, slot:slot + 1, :]
    slot = position % RESIDUAL_TOKENS
    low_key_ring[:, :, slot:slot + 1, :] = k_hat.unsqueeze(0)
    low_value_ring[:, :, slot:slot + 1, :] = v_hat.unsqueeze(0)
    read_key[:, :, position:position + 1, :] = high_k.unsqueeze(0)
    read_value[:, :, position:position + 1, :] = high_v.unsqueeze(0)


def _decode_step(
    model: Any,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    read_keys: list[torch.Tensor],
    read_values: list[torch.Tensor],
    token: int,
    position: int,
    config: dict[str, Any],
    calibrated: dict[str, Any],
    h_bits: int,
    *,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int | None, int]:
    """追加 KV 并按共同 FP32 注意力解码；CPU 缓存原样逐层搬运，不转换精度。"""
    device = model.model.embed_tokens.weight.device
    hidden = model.model.embed_tokens(torch.tensor([[token]], device=device))
    positions = torch.tensor([[position]], device=device)
    cos, sin = model.model.rotary_emb(hidden, positions)
    apply_rope = modeling_module(model).apply_rotary_pos_emb
    invalid = torch.zeros((), dtype=torch.int64, device=device)
    for index, layer in enumerate(model.model.layers):
        buffers = (keys, values, read_keys, read_values)
        originals = [group[index] for group in buffers]
        offloaded = read_keys[index].device.type == "cpu"
        if offloaded:
            resident = {}
            for value in originals:
                if id(value) not in resident:
                    resident[id(value)] = value.to(device)
            for group, original in zip(buffers, originals):
                group[index] = resident[id(original)]
        k_bits, v_bits = _layer_bits(config, index)
        residual = hidden
        query, key_pre, value = project_qkv(layer.self_attn, layer.input_layernorm(hidden))
        query, key_post = apply_rope(query, key_pre, cos, sin)
        source = key_pre[0] if config["rope"] == "pre" else key_post[0]
        k_params = _layer_params(calibrated, "key", index, k_bits)
        k_hat, bad_k = _quantize(source, k_bits, config["key_mode"], k_params)
        if config["rope"] == "pre":
            _, rotated = apply_rope(k_hat.unsqueeze(0), k_hat.unsqueeze(0), cos, sin)
            k_hat = rotated[0]
        v_params = _layer_params(calibrated, "value", index, v_bits)
        v_hat, bad_v = _quantize(value[0], v_bits, config["value_mode"], v_params)
        invalid += bad_k + bad_v
        if h_bits:
            high_k, bad_high_k = _recent(key_post[0], h_bits)
            high_v, bad_high_v = _recent(value[0], h_bits)
            _write_residual_token(
                keys[index], values[index], read_keys[index], read_values[index],
                k_hat, v_hat, high_k, high_v, position,
            )
            invalid += bad_high_k + bad_high_v
        else:
            keys[index][:, :, position:position + 1, :] = k_hat.unsqueeze(0)
            values[index][:, :, position:position + 1, :] = v_hat.unsqueeze(0)
        attended = attention_from_batch(
            query, read_keys[index][:, :, :position + 1, :],
            read_values[index][:, :, :position + 1, :],
        )
        hidden = residual + layer.self_attn.o_proj(attended)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        if observer is not None:
            observer({"layer": index, "attention": attended, "hidden": hidden,
                      "key_pre": key_pre, "key_post": key_post, "value": value})
        if offloaded:
            for original in {id(value): value for value in originals}.values():
                original.copy_(resident[id(original)])
            for group, original in zip(buffers, originals):
                group[index] = original
            del resident
    bad = int(invalid.item())
    if bad:
        return None, bad
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, -1, :])
    if observer is not None:
        observer({"logits": logits})
    return int(logits.reshape(-1, logits.shape[-1])[0].argmax()), 0


def _generate(
    model: Any,
    ids: list[int],
    eos: set[int],
    limit: int,
    config: dict[str, Any],
    calibrated: dict[str, Any],
    h_bits: int = 0,
    *,
    offload_layers: int = 0,
) -> tuple[list[int], str, int]:
    """返回生成 token、停止原因和累计无效组数。"""
    tensor = torch.tensor([ids], device="cuda")
    first, keys, values, read_keys, read_values, invalid = _prefill_cache(
        model, tensor, config, calibrated, limit, h_bits, offload_layers=offload_layers
    )
    if invalid:
        return [first], "invalid", invalid
    produced = [first]
    if first in eos:
        return produced, "eos", 0
    token = first
    for step in range(limit - 1):
        next_token, bad = _decode_step(
            model, keys, values, read_keys, read_values,
            token, len(ids) + step, config, calibrated, h_bits,
        )
        invalid += bad
        if next_token is None:
            return produced, "invalid", invalid
        token = next_token
        produced.append(token)
        if token in eos:
            return produced, "eos", invalid
    return produced, "limit", invalid


def _inputs(root: Path, model_key: str, task: str, split: str) -> list[dict[str, Any]]:
    """按冻结顺序加载题目索引；GSM8K 正文延迟到运行时构造。"""
    if task == "hotpot_four":
        if split != "v":
            raise ValueError("四问联答只使用 V")
        from experiments.generation_error_sources.code.four_question_baseline import samples

        return samples(root, model_key)
    if task != "gsm8k":
        return _samples(root, model_key, task, split)
    if split not in ("c", "v", "expand"):
        raise ValueError(f"不支持的数学题划分：{split}")
    frozen = json.loads((root / "research/phase1/splits/gsm8k.json").read_text("utf-8"))
    manifest = json.loads((root / "research/manifests/data/gsm8k.source.json").read_text("utf-8"))
    table = pq.read_table(manifest["destination"])
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
    from experiments.generation_error_sources.code.four_question_baseline import score_request

    scored = score_request(prediction, item["answers"])
    if invalid:
        scored.update(
            f1=0.0, em=False, answer_em_percent=0.0,
            answer_f1=[0.0, 0.0, 0.0, 0.0], answer_em=[False, False, False, False],
        )
    return scored


def run(
    model_key: str, task: str, split: str, name: str, commit: str, h_bits: int = 0,
    *, save_predictions: bool = False, guard_severe_failure: bool = False,
    execution_source: dict[str, Any] | None = None,
) -> None:
    """按一个配置和任务逐题运行；只回传小型评分与输入来源索引。"""
    if task not in (
        "gsm8k", "retrieval", "hotpot", "hotpot_long", "retrieval_ruler", "hotpot_four",
    ) or split not in ("v", "expand"):
        raise ValueError(f"不支持的任务与划分：{task}/{split}")
    if task in ("hotpot_long", "retrieval_ruler", "hotpot_four") and split != "v":
        raise ValueError("长任务只使用 V")
    if h_bits not in (0, 8, 16) or (h_bits and split != "v"):
        raise ValueError(f"残留条件不支持：H={h_bits}，{split}")
    root = project_root()
    config = _config(name)
    items = _inputs(root, model_key, task, split)
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    calibration = _calibration(root, model_key, config)
    ids_order = [item["id"] for item in items]
    if task == "gsm8k":
        template = json.loads(
            (root / "research/phase1/splits/templates.json").read_text("utf-8")
        )["gsm8k"]
        template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
        source_index = [template_hash + item["question_sha256"] for item in items]
    else:
        source_index = [item["sha256"] for item in items]
    suffix = f"-residual{h_bits}" if h_bits else ""
    legacy_suffix = f"-h{h_bits}" if h_bits else ""
    progress = _runtime_path(
        root,
        f"generation-{model_key}-{task}-{split}-{name}{suffix}.json",
        f"e2-{model_key}-{task}-{split}-{name}{legacy_suffix}.json",
    )
    if progress.exists():
        state = json.loads(progress.read_text("utf-8"))
        if (state["commit"] != commit or state["ids"] != ids_order
                or state.get("source_index") != source_index):
            raise RuntimeError("已有量化进度与代码或冻结输入不一致")
        rows = state["rows"]
        predictions = state.get("predictions", [])
        if save_predictions and (len(predictions) != len(rows)
                                 or state.get("execution_source") != execution_source):
            raise RuntimeError("恢复预测缺失或执行来源变化，不能静默重新生成")
    else:
        rows = []
        predictions = []
    out = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    if guard_severe_failure:
        from experiments.generation_error_sources.code.early_stop import severe_failure
        references = [json.loads((out / f"{split}_{task}_{ref}.json").read_text("utf-8"))
                      for ref in ("bf16", "kv8-per_token-post")]
        failure = severe_failure(rows, references, task)
        if failure is not None:
            raise RuntimeError(f"已有进度已触发停止：{failure}")
    model = load_model(str(root / MODELS[model_key]))
    if save_predictions:
        execution_device = {
            "name": torch.cuda.get_device_name(),
            "total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        }
        torch.cuda.reset_peak_memory_stats()
    eos = _eos_ids(tokenizer, model_key == "llama")
    limit = 128 if task in ("retrieval", "retrieval_ruler") else 1024
    for item in items[len(rows):]:
        ids, source_hash = _ids_and_source(root, tokenizer, model_key, task, item)
        if len(ids) + limit > MAX_CONTEXT:
            raise RuntimeError(f"输入超出总窗口：{item['id']}")
        with torch.inference_mode():
            produced, stop, invalid = _generate(
                model, ids, eos, limit, config, calibration, h_bits
            )
        prediction = tokenizer.decode(produced, skip_special_tokens=True)
        row: dict[str, Any] = {
            "id": item["id"], "input_tokens": len(ids), "input_sha256": source_hash,
            "new_tokens": len(produced), "stop": stop, "invalid_groups": invalid,
        }
        if save_predictions:
            row["memory_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["memory_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        if task == "gsm8k":
            row["correct"] = not invalid and gsm8k_exact(prediction, item["answer"])
        elif task == "retrieval":
            row["correct"] = not invalid and retrieval_exact(prediction, item["answer"])
        elif task in ("retrieval_ruler", "hotpot_four"):
            row.update(_score_new_input(task, prediction, item, invalid))
        else:
            row["em"] = not invalid and answer_exact(prediction, item["answer"])
            row["f1"] = 0.0 if invalid else token_f1(prediction, item["answer"])
        if save_predictions:
            predictions.append({"id": item["id"], "text": prediction, "token_ids": produced})
            row["prediction_source"] = {
                "path": progress.relative_to(root).as_posix(), "index": len(rows),
                "text_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
            }
        rows.append(row)
        state = {
            "commit": commit, "ids": ids_order, "source_index": source_index,
            "rows": rows,
        }
        if save_predictions:
            state.update(predictions=predictions, execution_source=execution_source,
                         execution_device=execution_device,
                         model=model_key, config=name, residual_bits=h_bits,
                         residual_tokens=RESIDUAL_TOKENS if h_bits else 0,
                         task=f"{split}-{task}", generation={
                             "do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                             "max_new_tokens": limit, "eos_token_ids": sorted(eos),
                         })
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(progress)
        print(model_key, task, split, name, len(rows), row, flush=True)
        if task in ("retrieval_ruler", "hotpot_four"):
            torch.cuda.empty_cache()
        if guard_severe_failure:
            failure = severe_failure(rows, references, task)
            if failure is not None:
                partial = {
                    "protocol": "phase1-full-key-retrieval", "commit": commit,
                    "model": model_key, "config": name, "task": f"{split}-{task}",
                    "planned_n": len(items), "observed_n": len(rows), "rows": rows,
                    "residual_bits": h_bits, "residual_tokens": RESIDUAL_TOKENS if h_bits else 0,
                    "stopped_early": True, "reason": failure,
                    "calibration_commit": calibration.get("commit"),
                    "execution_source": execution_source,
                }
                write_quality_result(
                    out / f"{split}_{task}_{name}{suffix}_early_stop.json", partial,
                )
                raise RuntimeError(f"配置触发严重失效停止：{failure}")
    payload: dict[str, Any] = {
        "protocol": _task_protocol(root, task), "commit": commit, "model": model_key,
        "config": name, "task": f"{split}-{task}", "n": len(rows), "rows": rows,
        "calibration_commit": calibration.get("commit"),
        "invalid_samples": sum(row["invalid_groups"] > 0 for row in rows),
    }
    if h_bits:
        payload["residual_bits"] = h_bits
        payload["residual_tokens"] = RESIDUAL_TOKENS
    if save_predictions:
        payload["execution_source"] = execution_source
        payload["execution_device"] = execution_device
        payload["prediction_source"] = {
            "path": progress.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(progress.read_bytes()).hexdigest(),
        }
    if task == "hotpot_four":
        payload.update(
            statistical_unit="request", questions_per_request=4, n_questions=4 * len(rows),
            em=100 * sum(row["em"] for row in rows) / len(rows),
            f1=sum(row["f1"] for row in rows) / len(rows),
            answer_em=sum(row["answer_em_percent"] for row in rows) / len(rows),
            missing_answers=sum(row["missing_answers"] for row in rows),
        )
    elif task.startswith("hotpot"):
        payload["em"] = 100 * sum(row["em"] for row in rows) / len(rows)
        payload["f1"] = sum(row["f1"] for row in rows) / len(rows)
    else:
        payload["correct"] = sum(row["correct"] for row in rows)
        payload["em"] = 100 * payload["correct"] / len(rows)
        if task == "retrieval_ruler":
            payload["strict_correct"] = sum(row["strict_correct"] for row in rows)
    out = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    out.mkdir(parents=True, exist_ok=True)
    write_quality_result(out / f"{split}_{task}_{name}{suffix}.json", payload)
    print(model_key, task, split, name, payload.get("f1", payload.get("em")), flush=True)


if __name__ == "__main__":
    run(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
        int(sys.argv[6]) if len(sys.argv) > 6 else 0,
    )
