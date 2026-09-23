"""区间、掩码和单 query 注意力。"""

import math
import unittest

import torch

from kvstudy.attention import (
    attention_fp32,
    full_history_mask,
    probability_kl,
    relative_output_error,
    square_causal_mask,
)
from kvstudy.intervals import (
    binary_drop_interval,
    bootstrap_drop_interval,
    class_unit_passes,
    quality_state,
)


class IntervalTest(unittest.TestCase):
    def test_zero_disagreement_bounds(self) -> None:
        correct = [True] * 64
        interval = binary_drop_interval(correct, correct)
        # 0 次事件的 Clopper-Pearson 上界：1-(alpha/2)^(1/n)，这里 alpha=0.025。
        upper = 100.0 * (1.0 - (0.0125 ** (1.0 / 64)))
        self.assertAlmostEqual(interval.estimate, 0.0)
        self.assertAlmostEqual(interval.upper or 0.0, upper, places=6)
        self.assertGreater(interval.upper or 0.0, 6.0)
        self.assertLess(interval.upper or 0.0, 7.0)
        self.assertEqual(
            quality_state(interval, interval, format_ok=True),
            "证据不足",
        )

    def test_n256_zero_disagreement_can_be_kept(self) -> None:
        correct = [True] * 256
        interval = binary_drop_interval(correct, correct)
        upper = 100.0 * (1.0 - (0.0125 ** (1.0 / 256)))
        self.assertAlmostEqual(interval.upper or 0.0, upper, places=6)
        self.assertAlmostEqual(upper, 1.7, delta=0.05)
        self.assertEqual(quality_state(interval, interval, format_ok=True), "保留")

    def test_lower_bound_eliminates(self) -> None:
        reference = [True] * 64
        candidate = [False] * 64
        interval = binary_drop_interval(reference, candidate)
        kept = binary_drop_interval(reference, reference)
        self.assertEqual(quality_state(interval, kept, format_ok=True), "淘汰")
        self.assertFalse(class_unit_passes(interval, kept, format_ok=True))

    def test_all_zero_continuous_has_no_width(self) -> None:
        scores = [1.0, 1.0, 1.0]
        interval = bootstrap_drop_interval(scores, scores, seed=1)
        self.assertEqual(interval.kind, "all_zero")
        self.assertIsNone(interval.lower)
        self.assertIsNone(interval.upper)
        self.assertTrue(class_unit_passes(interval, interval, format_ok=True))


class AttentionMaskTest(unittest.TestCase):
    def test_single_query_sees_full_history(self) -> None:
        mask = full_history_mask(32768)
        self.assertEqual(int(mask.sum()), 32768)
        square = square_causal_mask(1)
        self.assertEqual(int(square.sum()), 1)

    def test_hand_attention(self) -> None:
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        value = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]], dtype=torch.float32)
        output, scores, probs = attention_fp32(query, key, value)
        scale = 1.0 / math.sqrt(2.0)
        expected_scores = torch.tensor([scale, 0.0])
        self.assertTrue(torch.allclose(scores[0], expected_scores))
        weight = torch.softmax(expected_scores, dim=0)
        expected = weight[0] * value[0, 0] + weight[1] * value[0, 1]
        self.assertTrue(torch.allclose(output[0], expected))
        self.assertTrue(torch.allclose(probs[0], weight))
        error, zero = relative_output_error(output, output)
        self.assertEqual(error, 0.0)
        self.assertFalse(zero)

    def test_zero_output_is_flagged(self) -> None:
        reference = torch.zeros(2)
        error, zero = relative_output_error(reference, reference + 1)
        self.assertTrue(zero)
        self.assertEqual(error, 0.0)

    def test_kl_zero_when_identical(self) -> None:
        probs = torch.tensor([[0.25, 0.75]])
        kl = probability_kl(probs, probs)
        self.assertAlmostEqual(float(kl), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
