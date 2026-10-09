"""加载模型，并用 FP32 注意力做单步 decode。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from kvstudy.attention import attention_fp32, full_history_mask


def load_model(model_dir: str) -> Any:
    """以 BF16 和 SDPA 加载因果语言模型，放到当前 CUDA 设备。"""
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    model.requires_grad_(False)
    return model


def modeling_module(model: Any) -> Any:
    """返回定义 apply_rotary_pos_emb 的建模模块。"""
    name = model.config.model_type
    if name == "llama":
        import transformers.models.llama.modeling_llama as module
    elif name == "qwen2":
        import transformers.models.qwen2.modeling_qwen2 as module
    else:
        raise ValueError(f"未支持的模型类型: {name}")
    return module


def cache_lists(past: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """从 Transformers cache 取出每层的 K、V，形状 [batch, kv_heads, seq, dim]。"""
    if hasattr(past, "key_cache") and past.key_cache:
        return list(past.key_cache), list(past.value_cache)
    if hasattr(past, "layers"):
        keys = [layer.keys for layer in past.layers]
        values = [layer.values for layer in past.layers]
        return keys, values
    keys = [item[0] for item in past]
    values = [item[1] for item in past]
    return keys, values


def project_qkv(
    attention: Any, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """投影并整理成 [batch, heads, seq, dim]。"""
    query = attention.q_proj(hidden)
    key = attention.k_proj(hidden)
    value = attention.v_proj(hidden)
    batch, seq, _ = query.shape
    query = query.view(batch, seq, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch, seq, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
    value = value.view(
        batch, seq, attention.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)
    return query, key, value


def attention_from_batch(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """batch=1、query 长度=1 的 FP32 注意力，返回与 query 同 dtype 的 [1, 1, hidden]。"""
    output, _, _ = attention_fp32(
        query[0, :, 0, :].float(),
        key[0].float(),
        value[0].float(),
        valid=full_history_mask(key.shape[2], device=key.device),
    )
    hidden = query.shape[1] * query.shape[3]
    return output.view(1, 1, hidden).to(dtype=query.dtype)


def prefill(model: Any, input_ids: torch.Tensor) -> Any:
    """标准 SDPA prefill。只保留最后一个位置的词表 logits。"""
    try:
        return model(input_ids=input_ids, use_cache=True, num_logits_to_keep=1)
    except TypeError:
        outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=False)
        outputs.logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
        return outputs


def decode_step(
    model: Any,
    past_k: list[torch.Tensor],
    past_v: list[torch.Tensor],
    token_id: int,
    position: int,
    *,
    observer: Callable[[dict[str, Any]], None] | None = None,
    consume_cache: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """用全历史掩码解码；consume_cache=True 时逐层释放传入列表中的旧缓存。"""
    apply_rope = modeling_module(model).apply_rotary_pos_emb
    device = past_k[0].device
    hidden = model.model.embed_tokens(torch.tensor([[token_id]], device=device))
    position_ids = torch.tensor([[position]], device=device)
    cos, sin = model.model.rotary_emb(hidden, position_ids)
    updated_k: list[torch.Tensor] = []
    updated_v: list[torch.Tensor] = []
    for index, layer in enumerate(model.model.layers):
        residual = hidden
        query, key, value = project_qkv(layer.self_attn, layer.input_layernorm(hidden))
        key_pre, new_value = key, value
        query, key = apply_rope(query, key, cos, sin)
        key_post = key
        key = torch.cat([past_k[index], key], dim=2)
        value = torch.cat([past_v[index], value], dim=2)
        attended = attention_from_batch(query, key, value)
        hidden = residual + layer.self_attn.o_proj(attended)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        if observer is not None:
            observer({"layer": index, "attention": attended, "hidden": hidden,
                      "query": query, "key_pre": key_pre,
                      "key_post": key_post, "value": new_value})
        updated_k.append(key)
        updated_v.append(value)
        if consume_cache:
            past_k[index] = None
            past_v[index] = None
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden[:, -1, :])
    if observer is not None:
        observer({"logits": logits})
    return logits, updated_k, updated_v


def synchronize() -> None:
    """等待当前 CUDA 流结束，便于计时。"""
    torch.cuda.synchronize()


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    """返回函数结果和耗时秒数。"""
    synchronize()
    start = time.perf_counter()
    result = function()
    synchronize()
    return result, time.perf_counter() - start
