"""内容 token 的目标支持度、等权汇总和 logit 叠加归因的长期回归。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.static_layer_allocation.code.content import quantized_trace
from experiments.static_layer_allocation.code.content_data import answer_positions
from experiments.static_layer_allocation.code.content_metrics import (
    additive_metrics, question_metrics, token_metrics,
)
from phase1.splits_build import independent_mapping


class ContentAllocationTests(unittest.TestCase):
    """覆盖错误自信、实际分词、跨步缓存和纯指标非线性。"""

    def test_independent_mapping_is_unique_and_reproducible(self) -> None:
        for seed in range(32):
            key, value, others = independent_mapping(seed, 8)
            suffixes = [key[4:], value[4:]] + [x[4:] for pair in others for x in pair]
            self.assertEqual(len(set(suffixes)), 18)
            self.assertEqual((key, value, others), independent_mapping(seed, 8))

    def test_confident_wrong_token_has_negative_margin(self) -> None:
        row = token_metrics(np.array([10.0, 1.0, -4.0]), 1)
        self.assertEqual(row['margin'], -9.0)
        self.assertGreater(row['nll'], 9.0)

    def test_actual_token_offsets_include_boundary_overlap(self) -> None:
        class Tokens:
            input_ids = [0, 1, 2]
            offset_mapping = [(0, 3), (3, 6), (6, 12)]

        class Tokenizer:
            def __call__(self, *args: object, **kwargs: object) -> Tokens:
                return Tokens()

            def decode(self, *args: object, **kwargs: object) -> str:
                return 'val-12345678'

        measured = answer_positions(Tokenizer(), 'val-12345678')
        self.assertEqual(measured['content_indices'], [1, 2])

    def test_question_content_tokens_are_averaged(self) -> None:
        z = np.array([[1.0, 0.0], [0.0, 2.0]])
        row = question_metrics(z, [0, 0], z)
        expected = (token_metrics(z[0], 0)['nll'] + token_metrics(z[1], 0)['nll']) / 2
        self.assertEqual(row['nll'], expected)
        self.assertEqual(row['first_content']['nll'], token_metrics(z[0], 0)['nll'])

    def test_linear_logits_can_have_nonadditive_kl(self) -> None:
        base = np.array([[0.0, 0.0, 0.0]])
        singles = {0: np.array([[2.0, 0.0, 0.0]]), 1: np.array([[0.0, 2.0, 0.0]])}
        actual = singles[0] + singles[1]
        row = additive_metrics(base, singles, actual, [0, 1], [2], base)
        self.assertEqual(row['actual_to_additive_kl'], 0.0)
        self.assertEqual(row['centered_logit_residual_energy_fraction'], 0.0)
        self.assertGreater(abs(row['kl_sum_minus_actual']), 0.1)

    def test_teacher_forcing_preserves_state_and_measures_before_target_input(self) -> None:
        source = {'input_ids': torch.zeros((1, 10), dtype=torch.long),
                  'keys': [], 'values': [], 'logits': torch.tensor([1.0, 0.0])}
        cache = [0]
        calls = []

        def decode(*args: object, **kwargs: object) -> tuple[int, int]:
            token, position = args[5:7]
            cache[0] += 1
            calls.append((token, position, cache[0]))
            kwargs['observer']({'logits': torch.tensor([float(cache[0]), -float(token)])})
            return 0, 0

        module = 'experiments.static_layer_allocation.code.content'
        with patch(module + '._prefill_cache', return_value=(0, [], [], [], [], 0)) as init:
            with patch(module + '._decode_step', side_effect=decode):
                values = quantized_trace(None, source, [3, 4, 5, 6], [1, 3], {}, {}, 0)
        self.assertEqual(init.call_count, 1)
        self.assertEqual(calls, [(3, 10, 1), (4, 11, 2), (5, 12, 3)])
        np.testing.assert_array_equal(values, [[1.0, -3.0], [3.0, -5.0]])


if __name__ == '__main__':
    unittest.main()
