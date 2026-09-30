"""验证共同候选范围不会因模型内排名或历史名额而缩减。"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from experiments.representation_interactions.code.select import (
    K8V4_CANDIDATES,
    KV8_BASELINE,
    select_representations,
)


def summary() -> dict[str, dict[str, Any]]:
    """构造容量低于 KV8、局部误差排序不同的四种表示。"""
    cells = {
        name: {"error": 0.005 + index * 0.00001, "bytes": 70 + index, "invalid": 0}
        for index, name in enumerate(K8V4_CANDIDATES)
    }
    cells[KV8_BASELINE] = {"error": 0.0001, "bytes": 100, "invalid": 0}
    return cells


class RepresentationSelectionTest(unittest.TestCase):
    """检查共同范围、完整测量和容量约束。"""

    def test_model_rankings_do_not_cut_candidates(self) -> None:
        """反转误差与输入顺序，仍保留同一完整因素对照。"""
        first = summary()
        second = copy.deepcopy(first)
        for index, name in enumerate(K8V4_CANDIDATES):
            second[name]["error"] = first[K8V4_CANDIDATES[-index - 1]]["error"]
        second = dict(reversed(list(second.items())))
        self.assertEqual(select_representations(first).kept, K8V4_CANDIDATES)
        self.assertEqual(select_representations(second).kept, K8V4_CANDIDATES)

    def test_reviewed_models_support_the_same_four(self) -> None:
        """直接核对冻结结果；历史选择字段不需要改写。"""
        root = Path(__file__).resolve().parents[2]
        folder = root / "research/phase1/experiments/representation_interactions/results"
        for model in ("qwen", "llama"):
            payload = json.loads((folder / model / "summary.json").read_text("utf-8"))
            original = copy.deepcopy(payload)
            self.assertEqual(select_representations(payload["summary"]).kept, K8V4_CANDIDATES)
            self.assertEqual(payload, original)

    def test_missing_measurement_does_not_make_partial_selection(self) -> None:
        """缺少一种表示或基线时必须报错。"""
        for name in (*K8V4_CANDIDATES, KV8_BASELINE):
            cells = summary()
            del cells[name]
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "缺少"):
                select_representations(cells)

    def test_complete_bytes_must_be_below_kv8(self) -> None:
        """按实际完整字节检查，不能只靠名义位宽。"""
        for value in (100, 101):
            cells = summary()
            cells[K8V4_CANDIDATES[0]]["bytes"] = value
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "容量"):
                select_representations(cells)

    def test_invalid_measurements_cannot_be_registered(self) -> None:
        """非有限误差、非法计数和缺失容量不能成为候选依据。"""
        for metric, value in (("error", float("nan")), ("invalid", 0.5), ("bytes", None)):
            cells = summary()
            cells[K8V4_CANDIDATES[0]][metric] = value
            with self.subTest(metric=metric), self.assertRaises(RuntimeError):
                select_representations(cells)


if __name__ == "__main__":
    unittest.main()
