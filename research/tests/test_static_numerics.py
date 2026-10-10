"""覆盖严格规范化、首次分歧、数值包络及增量归因的长期回归。"""

from __future__ import annotations

import unittest

import numpy as np

from experiments.static_layer_allocation.code.numerics_metrics import (
    incremental_comparison, paired_reliability, path_metrics, score_answer, spearman,
)


class NumericalDiagnosticTests(unittest.TestCase):
    """检验预测路径与文本评分的区别，不把重复性等同于数值准确。"""

    def test_normalization_accepts_bare_content_but_rejects_extraction(self) -> None:
        self.assertEqual(score_answer(' 1234abcd\n', 'val-1234abcd')['content_correct'], 1)
        self.assertEqual(score_answer('1234abcd', 'val-1234abcd')['format_compliant'], 0)
        for wrong in ('answer: val-1234abcd', 'val-1234abcd val-deadbeef', '1234abcd-d0'):
            self.assertEqual(score_answer(wrong, 'val-1234abcd')['content_correct'], 0)
        scored = score_answer('val-deadbeef', 'val-1234abcd')
        self.assertEqual(scored['format_compliant'], 1)
        self.assertEqual(scored['formatted_correct'], 0)

    def test_first_nonpositive_differs_from_global_minimum(self) -> None:
        class Tokenizer:
            def decode(self, ids: list[int]) -> str:
                return str(ids[0])
        z = np.array([[3., 0., 1.], [1., 2., 0.], [-4., 8., 0.]])
        row = path_metrics(z, [0, 0, 0], Tokenizer(), [1])
        self.assertEqual(row['first_nonpositive']['answer_index'], 1)
        self.assertEqual(row['minimum']['answer_index'], 2)
        self.assertEqual(row['first_nonpositive']['competitor_id'], 1)

    def test_consistent_repeats_do_not_remove_precision_uncertainty(self) -> None:
        effects = np.tile([-1e-6, -1e-6, 1e-6, 1e-6], (16, 1))
        row = paired_reliability(effects, 17, 1000)
        self.assertEqual(row['status'], 'unresolved')
        stable = paired_reliability(np.tile([-2., -1.], (16, 1)), 17, 1000)
        self.assertEqual(stable['status'], 'reliably_lower')

    def test_paired_effect_ignores_common_baseline_displacement(self) -> None:
        base = np.array([[2., 1., 0.]])
        actual = base + np.array([[1., 0., -1.]])
        offset = np.array([[0., 5., -8.]])
        row = incremental_comparison(base + offset, actual + offset, base, actual)
        self.assertEqual(row['increment_difference_centered_rms'], 0.)
        self.assertAlmostEqual(row['increment_direction_cosine'], 1.)

    def test_average_ties_do_not_create_layer_order_evidence(self) -> None:
        self.assertIsNone(spearman([0., 0., 0.], [1., 2., 3.]))
        self.assertEqual(spearman([1., 1., 2.], [3., 3., 4.]), 1.)

    def test_protected_random_keeps_budget_and_nested_sets(self) -> None:
        from kvstudy.paths import project_root
        from experiments.static_layer_allocation.code.numerics import (
            DESIGN, FORMAL, protected_allocations, read,
        )
        from experiments.static_layer_allocation.code.protocol import load_plan

        root = project_root()
        plan, design = load_plan(root), read(root / DESIGN)
        for model, hidden, heads, kv_heads in (('qwen', 3584, 28, 4), ('llama', 4096, 32, 8)):
            base = read(root / FORMAL / model / 'allocations.json')
            probe = read(root / FORMAL / model / 'probe.json')
            geometry = {'hidden_size': hidden, 'num_attention_heads': heads,
                        'num_key_value_heads': kv_heads}
            result = protected_allocations(base, probe, plan, geometry, design)
            self.assertEqual(len(result['protected_random_names']), 27)
            for fmt in plan['formats']:
                for seed in base['random_permutations'][fmt]:
                    previous = set()
                    for r in plan['models'][model]['k4_layers']:
                        row = result['allocations'][f'{fmt}_k4_layers_{r}_protected_random_{seed}']
                        selected = set(row['k4_layers'])
                        self.assertEqual(len(selected), r)
                        self.assertTrue(previous <= selected)
                        self.assertFalse(selected & set(result['protected'][fmt]))
                        self.assertTrue(all(bits['v'] == 4 for bits in row['layer_bits']))
                        reference = base['allocations'][f'{fmt}_k4_layers_{r}_sensitivity']
                        self.assertEqual(row['cost'], reference['cost'])
                        previous = selected


if __name__ == '__main__':
    unittest.main()
