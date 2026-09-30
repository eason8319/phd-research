"""验证真实生成候选不会沿用历史名额，缺测时停止重新选择。"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.generation_error_sources.code.candidates import (
    KV4,
    KV8,
    generation_candidates,
    require_generation_results,
)
from experiments.representation_interactions.code.select import K8V4_CANDIDATES


class GenerationSelectionTest(unittest.TestCase):
    """覆盖候选扩大、缺测保护与提前停止后的结果范围。"""

    def setUp(self) -> None:
        """构造历史只选一种、当前四种测量齐备的主矩阵。"""
        self.matrix = {
            name: {"error": 0.005, "bytes": 70 + index, "invalid": 0}
            for index, name in enumerate(K8V4_CANDIDATES)
        }
        self.matrix[KV8] = {"error": 0.0001, "bytes": 100, "invalid": 0}
        self.matrix[KV4] = {"error": 0.02, "bytes": 90, "invalid": 0}
        self.payload = {"summary": self.matrix, "kept": [K8V4_CANDIDATES[-1]]}

    def test_all_four_despite_historical_kept(self) -> None:
        """旧结果中的 kept 不限制当前生成实验的候选。"""
        chosen = generation_candidates(self.payload)
        self.assertEqual(chosen, (KV8, KV4, *K8V4_CANDIDATES))
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertEqual(self.payload["kept"], [K8V4_CANDIDATES[-1]])

    def test_missing_candidate_blocks_selection(self) -> None:
        """任一新候选在判断单元上缺测，都不能开始重新选择。"""
        for task in ("gsm8k", "hotpot"):
            missing = f"v_{task}_{K8V4_CANDIDATES[0]}.json"
            with patch.object(
                Path, "exists", autospec=True, side_effect=lambda path: path.name != missing
            ):
                with self.subTest(task=task), self.assertRaisesRegex(RuntimeError, missing):
                    require_generation_results(
                        Path("unused"), list(generation_candidates(self.payload))
                    )

    def test_missing_reference_also_blocks_selection(self) -> None:
        """完整配对需要相同单元的高精度和 KV8 参照。"""
        for missing in ("v_hotpot_bf16.json", f"v_gsm8k_{KV8}.json"):
            with patch.object(
                Path, "exists", autospec=True, side_effect=lambda path: path.name != missing
            ):
                with self.subTest(missing=missing), self.assertRaisesRegex(RuntimeError, missing):
                    require_generation_results(
                        Path("unused"), list(generation_candidates(self.payload))
                    )

    def test_checked_early_stop_can_exclude_remaining_results(self) -> None:
        """调用方核验严重失效后，仅检查仍需运行的配置。"""
        stopped = K8V4_CANDIDATES[0]
        active = [name for name in generation_candidates(self.payload) if name != stopped]
        with patch.object(
            Path, "exists", autospec=True, side_effect=lambda path: stopped not in path.name
        ):
            require_generation_results(Path("unused"), active)


if __name__ == "__main__":
    unittest.main()
