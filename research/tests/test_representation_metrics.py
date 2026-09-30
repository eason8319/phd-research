"""H1 配对指标的平移不变性、精确分解和小扰动传播回归测试。"""

import unittest

import torch

from experiments.representation_interactions.code.metrics import (
    competition_metrics,
    output_metrics,
    paired_head_metrics,
)


class PairedHeadMetricsTest(unittest.TestCase):
    """检查逐 head 指标与直接注意力计算一致。"""

    def setUp(self) -> None:
        self.score8 = torch.tensor([[0.3, -0.4, 0.8]], dtype=torch.float64)
        self.score4 = self.score8 + torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)
        self.value8 = torch.tensor([[[1.0, 2.0], [-2.0, 1.0], [3.0, -1.0]]])
        self.value4 = self.value8 + torch.tensor([[[0.1, -0.2], [0.3, 0.1], [-0.1, 0.2]]])
        self.reference = torch.tensor([[1.0, 0.5]], dtype=torch.float64)

    def metrics(self, score4: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """在同一基准下返回测试用指标。"""
        return paired_head_metrics(
            self.score8, self.score4 if score4 is None else score4,
            self.value8, self.value4, self.reference,
        )

    def test_common_score_shift_changes_no_probability_or_output_metrics(self) -> None:
        original = self.metrics()
        shifted = self.metrics(self.score4 + 17.0)
        for name in original:
            if original[name].dtype == torch.bool:
                self.assertTrue(torch.equal(original[name], shifted[name]), name)
            else:
                torch.testing.assert_close(
                    original[name], shifted[name], rtol=1e-11, atol=1e-12, equal_nan=True
                )
        unchanged = self.metrics(self.score8 + 17.0)
        for name in ("weighted_score_var", "paired_probability_kl", "key_increment_error"):
            self.assertAlmostEqual(float(unchanged[name][0]), 0.0, places=12)

    def test_equal_values_make_key_output_change_zero(self) -> None:
        equal_values = torch.tensor([[[1.5, -2.0]]], dtype=torch.float64).expand(1, 3, 2)
        result = paired_head_metrics(
            self.score8, self.score4, equal_values, equal_values, self.reference
        )
        self.assertGreater(float(result["paired_probability_kl"][0]), 0.0)
        self.assertAlmostEqual(float(result["key_increment_error"][0]), 0.0, places=24)
        self.assertAlmostEqual(float(result["key_prediction_error"][0]), 0.0, places=24)

    def test_output_errors_and_cross_term_match_direct_calculation(self) -> None:
        result = self.metrics()
        probability8 = torch.softmax(self.score8, dim=-1)
        probability4 = torch.softmax(self.score4, dim=-1)
        output88 = probability8 @ self.value8.double()[0]
        output48 = probability4 @ self.value8.double()[0]
        output84 = probability8 @ self.value4.double()[0]
        denominator = self.reference.square().sum(-1)
        expected = {
            "kv8_error": (output88 - self.reference).square().sum(-1) / denominator,
            "k4v8_error": (output48 - self.reference).square().sum(-1) / denominator,
            "k8v4_error": (output84 - self.reference).square().sum(-1) / denominator,
            "key_increment_error": (output48 - output88).square().sum(-1) / denominator,
            "value_increment_error": (output84 - output88).square().sum(-1) / denominator,
        }
        for name, value in expected.items():
            torch.testing.assert_close(result[name], value)
        torch.testing.assert_close(
            result["difference_residual"], torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0
        )
        torch.testing.assert_close(result["probability_tv"], result["probability_l1"] / 2)

    def test_small_perturbation_prediction_converges_in_length_and_direction(self) -> None:
        direction = torch.tensor([[0.9, -0.7, 0.2]], dtype=torch.float64)
        residuals = []
        for scale in (0.1, 0.01, 0.001):
            result = self.metrics(self.score8 + scale * direction)
            residuals.append(float(result["key_prediction_relative_error"][0]))
        self.assertLess(residuals[1], residuals[0] / 50)
        self.assertLess(residuals[2], residuals[1] / 50)
        self.assertAlmostEqual(float(result["key_prediction_norm_ratio"][0]), 1.0, delta=0.001)
        self.assertGreater(float(result["key_prediction_cosine"][0]), 0.999999)
        ratio = float(result["paired_probability_kl"][0] / (result["weighted_score_var"][0] / 2))
        self.assertAlmostEqual(ratio, 1.0, delta=0.001)

    def test_zero_reference_and_zero_prediction_are_explicit(self) -> None:
        score = torch.zeros((1, 1), dtype=torch.float64)
        value = torch.ones((1, 1, 2), dtype=torch.float64)
        result = paired_head_metrics(score, score, value, value, torch.zeros((1, 2)))
        self.assertTrue(bool(result["zero_reference"][0]))
        self.assertTrue(bool(result["key_increment_zero"][0]))
        self.assertTrue(bool(result["key_prediction_zero"][0]))
        for name in (
            "kv8_error", "key_increment_error", "value_increment_error",
            "key_prediction_relative_error", "key_prediction_norm_ratio", "key_prediction_cosine",
        ):
            self.assertTrue(bool(torch.isnan(result[name][0])), name)
        self.assertEqual(float(result["weighted_score_var"][0]), 0.0)
        self.assertEqual(float(result["probability_l1"][0]), 0.0)

    def test_pair_norms_do_not_add_up_to_total_first_order_error(self) -> None:
        score8 = torch.zeros((1, 3), dtype=torch.float64)
        score4 = torch.tensor([[0.1, 0.0, -0.1]], dtype=torch.float64)
        value = torch.tensor([[[0.0], [1.0], [0.0]]], dtype=torch.float64)
        result = paired_head_metrics(score8, score4, value, value, torch.ones((1, 1)))
        pairs = competition_metrics(score8, score4, value, top_k=3)
        self.assertGreater(float(pairs["pair_first_order_norm"].sum()), 0.0)
        self.assertFalse(bool(result["key_increment_zero"][0]))
        # 对称项在数学上抵消；浮点矩阵乘法仍可能留下微小且非零的舍入残差。
        self.assertAlmostEqual(float(result["key_prediction_norm_ratio"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(result["key_prediction_relative_error"][0]), 1.0, places=12)
        if bool(result["key_prediction_zero"][0]):
            self.assertTrue(bool(torch.isnan(result["key_prediction_cosine"][0])))
        else:
            self.assertTrue(bool(torch.isfinite(result["key_prediction_cosine"][0])))

    def test_head_normalization_is_independent(self) -> None:
        result = paired_head_metrics(
            self.score8.expand(2, -1), self.score4.expand(2, -1),
            self.value8.expand(2, -1, -1), self.value4.expand(2, -1, -1),
            torch.cat((self.reference, self.reference * 2)),
        )
        self.assertAlmostEqual(
            float(result["key_increment_error"][0] / result["key_increment_error"][1]), 4.0
        )
        self.assertTrue(all(tensor.shape == (2,) for tensor in result.values()))

    def test_layer_metrics_use_total_reference_energy(self) -> None:
        reference = torch.tensor([[1.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
        key_delta = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        result = output_metrics(
            reference, reference, reference + key_delta, reference, key_delta
        )
        self.assertAlmostEqual(float(result["key_increment_error"]), 2 / 101)
        self.assertNotAlmostEqual(float(result["key_increment_error"]), (1 + 0.01) / 2)
        self.assertEqual(float(result["key_prediction_error"]), 0.0)
        self.assertAlmostEqual(float(result["key_prediction_norm_ratio"]), 1.0)
        self.assertAlmostEqual(float(result["key_prediction_cosine"]), 1.0)
        self.assertTrue(all(tensor.ndim == 0 for tensor in result.values()))

    def test_invalid_inputs_fail_instead_of_broadcasting(self) -> None:
        with self.assertRaisesRegex(ValueError, "float32"):
            paired_head_metrics(
                self.score8.half(), self.score4, self.value8, self.value4, self.reference
            )
        with self.assertRaisesRegex(ValueError, "同为"):
            self.metrics(torch.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "非有限"):
            self.metrics(torch.full_like(self.score4, torch.nan))


class CompetitionMetricsTest(unittest.TestCase):
    """检查固定基准 top-k 的竞争配对及退化历史情况。"""

    def test_selection_stays_fixed_when_an_unselected_position_overtakes(self) -> None:
        score8 = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
        score4 = torch.tensor([[1.0, 2.0, 10.0]], dtype=torch.float64)
        value = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [100.0, 100.0]]])
        result = competition_metrics(score8, score4, value, top_k=2)
        torch.testing.assert_close(result["selected_positions"], torch.tensor([[0, 1]]))
        torch.testing.assert_close(result["pair_positions"], torch.tensor([[[0, 1]]]))
        self.assertEqual(float(result["pair_margin"][0, 0]), 1.0)
        self.assertEqual(float(result["pair_differential"][0, 0]), -2.0)
        self.assertEqual(float(result["pair_quantized_margin"][0, 0]), -1.0)
        self.assertEqual(float(result["pair_value_distance2"][0, 0]), 25.0)
        self.assertEqual(int(result["pair_flip_count"][0]), 1)
        self.assertEqual(float(result["pair_flip_rate"][0]), 1.0)
        probability = torch.softmax(score8, dim=-1)
        expected = probability[0, 0] * probability[0, 1] * 2 * 5
        torch.testing.assert_close(result["pair_first_order_norm"][0, 0], expected)
        torch.testing.assert_close(
            result["selected_probability_coverage"], probability[:, :2].sum(-1)
        )

    def test_baseline_ties_use_position_order_and_are_not_strict_flips(self) -> None:
        score8 = torch.tensor([[1.0, 1.0, 0.0]])
        score4 = torch.tensor([[0.0, 2.0, 0.0]])
        result = competition_metrics(score8, score4, torch.zeros((1, 3, 2)), top_k=2)
        torch.testing.assert_close(result["selected_positions"], torch.tensor([[0, 1]]))
        self.assertEqual(int(result["pair_tie_count"][0]), 1)
        self.assertEqual(int(result["pair_flip_count"][0]), 0)
        self.assertEqual(float(result["pair_first_order_norm"][0, 0]), 0.0)

    def test_single_position_has_empty_pairs_and_explicit_zero_rate(self) -> None:
        result = competition_metrics(
            torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.ones((1, 1, 2))
        )
        self.assertEqual(result["pair_positions"].shape, (1, 0, 2))
        self.assertEqual(result["pair_margin"].shape, (1, 0))
        self.assertEqual(int(result["pair_count"][0]), 0)
        self.assertEqual(float(result["pair_flip_rate"][0]), 0.0)
        self.assertEqual(float(result["selected_probability_coverage"][0]), 1.0)
        with self.assertRaisesRegex(ValueError, "正整数"):
            competition_metrics(
                torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.ones((1, 1, 2)), top_k=0
            )


if __name__ == "__main__":
    unittest.main()
