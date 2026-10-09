"""GSM8K、检索和长问答的抽取与评分。"""

from __future__ import annotations

import re
import string
from collections import Counter

_HASH_NUMBER = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
_FINAL_ANSWER = re.compile(r"Final answer:\s*(.+)", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({char: " " for char in string.punctuation})
_SEVEN_DIGITS = re.compile(r"(?<![0-9])[0-9]{7}(?![0-9])")


def extract_gsm8k(text: str) -> str | None:
    """按官方规则取最后一个 #### 后的数值，并去掉逗号。"""
    matches = _HASH_NUMBER.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").strip()


def gsm8k_exact(prediction: str, gold: str) -> bool:
    """抽取后的数值字符串完全一致才算对。"""
    got = extract_gsm8k(prediction)
    expected = extract_gsm8k(gold)
    return got is not None and got == expected


def normalize_retrieval(text: str) -> str:
    """检索目标：去首尾空白、合并空白并忽略大小写。"""
    return " ".join(text.strip().casefold().split())


def retrieval_exact(prediction: str, target: str) -> bool:
    """规范化后的整段预测与预登记目标相同。"""
    return normalize_retrieval(prediction) == normalize_retrieval(target)


def retrieval_numbers(prediction: str) -> list[str]:
    """抽取不同的七位 ASCII 数字值；更长数字中的子串不参与评分。"""
    return sorted(set(_SEVEN_DIGITS.findall(prediction)))


def retrieval_number_exact(prediction: str, target: str) -> bool:
    """只有唯一七位数字值与目标一致才算正确；相同值重复出现允许。"""
    if not re.fullmatch(r"[1-9][0-9]{6}", target):
        raise ValueError("检索目标必须是无前导零的七位数字")
    return retrieval_numbers(prediction) == [target]


def extract_final_answer(text: str) -> str:
    """取最后一行 Final answer。没有这一行时用最后一段非空文本。"""
    matches = _FINAL_ANSWER.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_f1(text: str) -> list[str]:
    """SQuAD 式归一化后的 token。"""
    lowered = text.lower().translate(_PUNCT)
    without_articles = _ARTICLES.sub(" ", lowered)
    return without_articles.split()


def token_f1(prediction: str, gold: str) -> float:
    """答案 token F1，范围 0–100。两边都为空时为 100。"""
    predicted = normalize_f1(extract_final_answer(prediction))
    reference = normalize_f1(gold)
    if not predicted and not reference:
        return 100.0
    if not predicted or not reference:
        return 0.0
    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 100.0 * 2 * precision * recall / (precision + recall)


def answer_exact(prediction: str, gold: str) -> bool:
    """长问答的 EM：归一化后的 token 序列相同。"""
    return normalize_f1(extract_final_answer(prediction)) == normalize_f1(gold)
