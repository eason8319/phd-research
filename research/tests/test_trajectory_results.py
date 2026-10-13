"""保护传播诊断的共同轨迹、完整性及既有自由评分的输入身份。"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.generation_error_sources.code import review_trajectories as results

ROOT = Path(__file__).resolve().parents[2]


class TrajectoryResultsTest(unittest.TestCase):
    """从正式回传结果构造破坏性变体，验证审核会拒绝失配。"""

    def setUp(self) -> None:
        path = ROOT / "research/phase1/experiments/generation_error_sources/results"
        self.payload = json.loads((path / "qwen/trajectories.json").read_text("utf-8"))

    def test_reuse_is_read_only(self) -> None:
        original = copy.deepcopy(self.payload)
        results.reuse_quality(self.payload, ROOT)
        self.assertEqual(self.payload, original)

    def test_mismatched_input_is_rejected(self) -> None:
        self.payload["rows"][0]["input_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "实际输入不一致"):
            results.reuse_quality(self.payload, ROOT)

    def test_missing_step_is_rejected(self) -> None:
        self.payload["rows"][0]["candidates"][0]["steps"].pop()
        with self.assertRaisesRegex(ValueError, "回放缺步"):
            results.validate_trajectories(self.payload)

    def test_changed_reference_is_rejected(self) -> None:
        self.payload["rows"][0]["candidates"][1]["steps"][0]["bf16_argmax"] += 1
        with self.assertRaisesRegex(ValueError, "共同 BF16"):
            results.validate_trajectories(self.payload)

    def test_missing_candidate_is_rejected(self) -> None:
        self.payload["rows"][0]["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "三个共同候选"):
            results.validate_trajectories(self.payload)

    def test_wrong_reset_position_is_rejected(self) -> None:
        candidate = next(c for r in self.payload["rows"] for c in r["candidates"]
                         if c["history_reset"] is not None)
        candidate["history_reset"]["history_generated_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "历史重建位置"):
            results.validate_trajectories(self.payload)


if __name__ == "__main__":
    unittest.main()
