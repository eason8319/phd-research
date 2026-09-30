"""第一阶段共用的冻结样本、QKV 捕获、RoPE 与量化参数辅助逻辑。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pyarrow.parquet as pq
import torch

from kvstudy.groups import QuantStats, fit_per_channel, quantize_per_channel, quantize_per_token
from kvstudy.paths import project_root
from phase1.runner import cache_lists, modeling_module, prefill

MODELS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "llama": "models/Llama-3.1-8B-Instruct",
}
G_C = 32


def _render_gsm(tokenizer: Any, question: str, llama: bool) -> torch.Tensor:
    """按冻结 GSM8K 模板生成 CUDA 上的输入 token；Llama 使用固定日期。"""
    template_path = project_root() / "research/phase1/splits/templates.json"
    templates = json.loads(template_path.read_text(encoding="utf-8"))
    user = templates["gsm8k"].format(question=question)
    kwargs = {"date_string": "26 Jul 2024"} if llama else {}
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    return torch.tensor([ids], device="cuda")


def _load_c_items(model_key: str) -> list[dict[str, Any]]:
    """读取冻结 C 样本，并核对数学原题与检索全文的校验值。"""
    root = project_root()
    gsm = json.loads((root / "research/phase1/splits/gsm8k.json").read_text(encoding="utf-8"))
    meta_path = root / "research/manifests/data/gsm8k.source.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    questions = pq.read_table(root / meta["destination"]).column("question").to_pylist()
    items = []
    for row in gsm["c"]:
        question = questions[row["row"]]
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != row["question_sha256"]:
            raise RuntimeError(f"数学原题校验失败：{row['id']}")
        items.append({
            "id": row["id"], "fold": int(row["fold"]),
            "kind": "math", "question": question,
            "source_sha256": row["question_sha256"],
        })
    retrieval = json.loads(
        (root / "research/phase1/splits/retrieval.json").read_text(encoding="utf-8")
    )
    for row in retrieval:
        if not row["id"].startswith("synth-c-"):
            continue
        path = (
            root / "research/.runtime/data/phase1-text/retrieval"
            / model_key / f"{row['id']}.txt"
        )
        source = path.read_text(encoding="utf-8")
        frozen = row["models"][model_key]
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != frozen["sha256"]:
            raise RuntimeError(f"检索全文校验失败：{row['id']}")
        items.append({
            "id": row["id"], "fold": int(row["fold"]), "kind": "retrieval",
            "text": source, "tokens": frozen["tokens"],
            "source_sha256": frozen["sha256"],
        })
    del questions
    return items


def _capture(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    """捕获 prefill 和首个生成 token 的 QKV；返回 CPU FP32 张量及首 token。

    临时替换模型模块的 RoPE 函数，正常结束和异常退出时均恢复原函数。
    """
    module = modeling_module(model)
    original = module.apply_rotary_pos_emb
    pre: list[torch.Tensor] = []

    def wrapped(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pre.append(key.detach())
        return original(query, key, cos, sin, *args, **kwargs)

    module.apply_rotary_pos_emb = wrapped
    try:
        outputs = prefill(model, input_ids)
        token = int(outputs.logits.view(-1, outputs.logits.shape[-1])[0].argmax())
        query_pre: list[torch.Tensor] = []

        def wrapped_query(
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            *args: object,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            query_pre.append(query.detach())
            pre.append(key.detach())
            return original(query, key, cos, sin, *args, **kwargs)

        module.apply_rotary_pos_emb = wrapped_query
        step = model(
            input_ids=torch.tensor([[token]], device=input_ids.device),
            past_key_values=outputs.past_key_values,
            use_cache=True,
            num_logits_to_keep=1,
        )
    finally:
        module.apply_rotary_pos_emb = original
    keys, values = cache_lists(step.past_key_values)
    n_layers = len(query_pre)
    history_pre = pre[:n_layers]
    new_pre = pre[n_layers:]
    return {
        "pre": [
            torch.cat([old[0], new[0]], dim=1).float().cpu()
            for old, new in zip(history_pre, new_pre)
        ],
        "post": [item[0].float().cpu() for item in keys],
        "value": [item[0].float().cpu() for item in values],
        "query": [item[0, :, 0, :].float().cpu() for item in query_pre],
        "token": token,
    }


def _quantize_side(
    values: torch.Tensor,
    bits: int,
    mode: str,
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, QuantStats]:
    """返回反量化值和量化统计；逐通道量化需要已有校准参数。"""
    if mode == "per_token":
        return quantize_per_token(values, bits, G_C)
    assert params is not None
    return quantize_per_channel(values, bits, *params)


def _empty_minmax(layers: int, kv: int, dim: int) -> dict[str, torch.Tensor]:
    """创建逐层、逐 KV head、逐通道的最小值和最大值累积器。"""
    return {
        "min": torch.full((layers, kv, dim), float("inf")),
        "max": torch.full((layers, kv, dim), float("-inf")),
    }


def _update_minmax(acc: dict[str, torch.Tensor], tensors: list[torch.Tensor]) -> None:
    """沿历史位置轴更新校准范围，原地修改累积器。"""
    for layer, values in enumerate(tensors):
        acc["min"][layer] = torch.minimum(acc["min"][layer], values.amin(dim=1))
        acc["max"][layer] = torch.maximum(acc["max"][layer], values.amax(dim=1))


def _fit_minmax(
    acc: dict[str, torch.Tensor], bits: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """按通道最小值和最大值拟合各层的 scale、offset 与状态。"""
    fitted = []
    for layer in range(acc["min"].shape[0]):
        # 用最小/最大两个值代表该通道的校准范围。
        pair = torch.stack([acc["min"][layer], acc["max"][layer]], dim=1)
        scale, offset, status = fit_per_channel(pair, bits)
        fitted.append((scale, offset, status))
    return fitted


def _key_params(
    scale_pack: dict[str, Any], cfg: dict[str, Any], layer: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """按 Key 的表示位置和位宽取离线参数；在线逐 token 模式返回 None。"""
    if cfg["key_mode"] != "per_channel":
        return None
    kind = "k_pre" if cfg["rope"] == "pre" else "k_post"
    if cfg["k_bits"] == 4:
        kind = kind + "4"
    return scale_pack[kind][layer]


def _rotate_key(model: Any, key: torch.Tensor) -> torch.Tensor:
    """在 CUDA 旋转 [kv, seq, dim] 的 Key，返回 CPU FP32 张量。"""
    module = modeling_module(model)
    seq = key.shape[1]
    hidden = torch.zeros(
        1, seq, model.config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.arange(seq, device="cuda").view(1, -1)
    cos, sin = model.model.rotary_emb(hidden, positions)
    packed = key.unsqueeze(0).to("cuda")
    _query, rotated = module.apply_rotary_pos_emb(packed, packed, cos, sin)
    return rotated[0].float().cpu()


def _rotate_query(model: Any, query: torch.Tensor, position: int) -> torch.Tensor:
    """在 CUDA 旋转 [heads, dim] 的指定位置 query，返回 CPU FP32 张量。"""
    module = modeling_module(model)
    hidden = torch.zeros(
        1, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.tensor([[position]], device="cuda")
    cos, sin = model.model.rotary_emb(hidden, positions)
    packed = query.view(1, query.shape[0], 1, query.shape[1]).to("cuda")
    rotated, _key = module.apply_rotary_pos_emb(packed, packed[:, :1], cos, sin)
    return rotated[0, :, 0, :].float().cpu()
