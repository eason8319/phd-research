"""第一阶段模型位置与冻结的 EOS token 规则。"""

from __future__ import annotations

from typing import Any


MODELS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "llama": "models/Llama-3.1-8B-Instruct",
}


def _eos_ids(tokenizer: Any, llama: bool) -> set[int]:
    """返回既定贪心生成采用的 EOS token 集合。"""
    if llama:
        return {128001, 128008, 128009}
    values = []
    for token in ("<|im_end|>", "<|endoftext|>"):
        values.append(int(tokenizer.convert_tokens_to_ids(token)))
    return set(values)
