"""回归保障固定 V4、K 层预算、残留成本与共同参考的误差和 KL。"""

import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.static_layer_allocation.code.allocations import build_choices, cost, layer_bits
from experiments.static_layer_allocation.code.metrics import (
    distribution_summary, grouped_rse, rank_agreement, ranking, vocabulary_kl,
)
from experiments.static_layer_allocation.code.protocol import load_plan, require_v_plan


class StaticKeyAllocationTest(unittest.TestCase):
    """独立公式、边界与跨预算不变量防止恢复旧 K/V 分配语义。"""

    def test_group_first_common_denominator(self) -> None:
        """能量差异很大的组仍等权；分母既不取 K8，也不随候选变化。"""
        reference = np.array([[1.0], [10.0]])
        eight = np.array([[2.0], [20.0]])
        four = np.array([[3.0], [20.0]])
        value, zeros = grouped_rse(reference, four, eight, 2)
        self.assertEqual(value, 0.5)
        self.assertEqual(zeros, 0)
        # 若按层总能量归一化是 1/101，若分母取 K8 则是 1/8。
        self.assertNotAlmostEqual(value, 1 / 101)
        self.assertNotAlmostEqual(value, 1 / 8)

    def test_zero_energy_is_undefined(self) -> None:
        """全零参考不能冒充零敏感度，部分零组只计数。"""
        zero = np.zeros((2, 1))
        self.assertEqual(grouped_rse(zero, zero, zero, 2), (None, 2))
        reference = np.array([[0.0], [2.0]])
        four = np.array([[4.0], [3.0]])
        self.assertEqual(grouped_rse(reference, four, reference, 2), (0.25, 1))

    def test_local_change_can_move_towards_reference(self) -> None:
        """降位宽增量为正并不说明对高精度的误差增加。"""
        reference, eight, four = np.array([[2.0]]), np.array([[4.0]]), np.array([[3.0]])
        self.assertEqual(grouped_rse(reference, four, eight, 1)[0], 0.25)
        self.assertLess(grouped_rse(reference, four, reference, 1)[0],
                        grouped_rse(reference, eight, reference, 1)[0])

    def test_ties_and_invalid_layers(self) -> None:
        """正向及反向并列都取层号升序；无效量化不能冻结排名。"""
        rows = [{"sensitivity": value, "reconstruction": value, "invalid_groups": 0}
                for value in (1.0, 1.0, 2.0, 2.0)]
        self.assertEqual(ranking(rows, "sensitivity", descending=False), [0, 1, 2, 3])
        self.assertEqual(ranking(rows, "sensitivity", descending=True), [2, 3, 0, 1])
        rows[0]["invalid_groups"] = 1
        with self.assertRaisesRegex(RuntimeError, "无效量化"):
            ranking(rows, "sensitivity", descending=False)

    def test_nested_random_and_fixed_v_four(self) -> None:
        """三格式随机集合相同，三预算嵌套，只改变 K 层且 INT8 数量准确。"""
        plan = load_plan()
        seeds = [202609231, 202609232, 202609233]
        for model_spec in plan["models"].values():
            layers = model_spec["layers"]
            rows = [{"sensitivity": float(i), "reconstruction": float(layers - i),
                     "invalid_groups": 0} for i in range(layers)]
            choices, permutations = build_choices(rows, model_spec["k4_layers"], seeds)
            changed = [{**row, "sensitivity": -row["sensitivity"]} for row in rows]
            other, other_permutations = build_choices(changed, model_spec["k4_layers"], seeds)
            self.assertEqual(permutations, other_permutations)
            previous = {seed: set() for seed in seeds}
            for r in model_spec["k4_layers"]:
                strategies = choices[f"k4_layers_{r}"]
                self.assertEqual(len(strategies), 6)
                self.assertEqual(strategies["sensitivity"], list(range(r)))
                self.assertEqual(strategies["inverse"], list(range(layers - 1, layers - r - 1, -1)))
                for selected in strategies.values():
                    bits = layer_bits(layers, selected)
                    self.assertTrue(all(row["v"] == 4 for row in bits))
                    self.assertEqual(sum(row["k"] == 4 for row in bits), r)
                    self.assertEqual(sum(row["k"] == 8 for row in bits), layers - r)
                for seed in seeds:
                    selected = strategies[f"random_{seed}"]
                    self.assertEqual(selected, other[f"k4_layers_{r}"][f"random_{seed}"])
                    self.assertLessEqual(previous[seed], set(selected))
                    previous[seed] = set(selected)

    def test_complete_cost_and_residual_write(self) -> None:
        """手算一层 payload/参数/副本，验证驻留不增长时仍有残留写入。"""
        model = {"hidden_size": 128, "num_attention_heads": 1, "num_key_value_heads": 1}
        config = {"key_mode": "per_channel", "rope": "pre"}
        value = cost([{"k": 4, "v": 4}], model, config, 8, 4096)
        self.assertEqual(value["components_raw_bytes"], {
            "k_payload": 262144, "v_payload": 262144,
            "k_params": 512, "v_params": 65536, "residual": 36864,
        })
        self.assertEqual(value["resident_bytes"], 627200)
        self.assertEqual(value["decode_main_write_bytes"], 144)
        self.assertEqual(value["decode_residual_write_bytes"], 288)
        self.assertEqual(value["decode_write_bytes"], 432)
        self.assertEqual(value["decode_read_full_bytes"] - value["decode_read_lower_bytes"],
                         18432)
        self.assertEqual(value["pre_rope_mul_adds_per_read"], 3 * 4096 * 128)

    def test_budget_equal_cost_and_cross_format_difference(self) -> None:
        """相同 K4 数量的层名单成本相同；离线参数及 H8 改变跨格式字节。"""
        model = {"hidden_size": 512, "num_attention_heads": 4, "num_key_value_heads": 2}
        first, second = layer_bits(8, [0, 1]), layer_bits(8, [6, 7])
        online = {"key_mode": "per_token", "rope": "pre"}
        offline = {"key_mode": "per_channel", "rope": "pre"}
        for config, residual in ((online, 0), (offline, 8)):
            self.assertEqual(cost(first, model, config, residual, 4096),
                             cost(second, model, config, residual, 4096))
        self.assertNotEqual(cost(first, model, online, 0, 4096)["resident_bytes"],
                            cost(first, model, offline, 8, 4096)["resident_bytes"])

    def test_kl_identity_shift_tiny_and_large(self) -> None:
        """自比较与常数平移为零；微小扰动保持正值，极端尾部不溢出。"""
        logits = np.array([0.0, 0.0])
        self.assertEqual(vocabulary_kl(logits, logits), 0.0)
        self.assertEqual(vocabulary_kl(logits, logits + 100.0), 0.0)
        value = vocabulary_kl(logits, np.array([1e-8, -1e-8]))
        self.assertAlmostEqual(value / 5e-17, 1.0, places=7)
        extreme = vocabulary_kl(np.array([1000.0, -1000.0]),
                                np.array([-1000.0, 1000.0]))
        self.assertEqual(extreme, 2000.0)
        with self.assertRaisesRegex(ValueError, "非有限"):
            vocabulary_kl(logits, np.array([math.nan, 0.0]))

    def test_kl_matches_independent_definition(self) -> None:
        """一般尺度下与直接归一化概率的 KL 定义一致。"""
        rng = np.random.default_rng(8319)
        for _ in range(20):
            old, new = rng.normal(size=(2, 100))
            p, q = np.exp(old), np.exp(new)
            p, q = p / p.sum(), q / q.sum()
            expected = float(np.sum(p * np.log(p / q)))
            self.assertAlmostEqual(vocabulary_kl(old, new), expected, places=13)

    def test_each_budget_strict_gate_and_paired_contribution(self) -> None:
        """独立预算分别判断；与随机并列不通过，均值不掩盖题级方向。"""
        seeds = [1, 2, 3]
        strategies = ["sensitivity", "reconstruction", "inverse", "random_1", "random_2",
                      "random_3"]
        groups = {f"k4_layers_{r}": [f"online_pre_k4_layers_{r}_{s}" for s in strategies]
                  for r in (2, 7, 14)}
        formats = {"online_pre": {"budget_groups": groups}}
        rows = []
        for kind in ("math", "retrieval"):
            for index in range(4):
                kl = {"bf16": 0.0, "kv8": 0.1, "online_pre_k8v4": 0.2}
                for r in (2, 7, 14):
                    for strategy in strategies:
                        kl[f"online_pre_k4_layers_{r}_{strategy}"] = float(r)
                    if r != 7:
                        kl[f"online_pre_k4_layers_{r}_sensitivity"] = float(r - 1)
                rows.append({"id": f"{kind}_{index}", "kind": kind, "kl": kl})
        result = distribution_summary(rows, formats, seeds)
        checks = result["budget_comparisons"]["online_pre"]
        self.assertTrue(checks["k4_layers_2"]["sensitivity_strictly_below_all_random"])
        self.assertFalse(checks["k4_layers_7"]["sensitivity_strictly_below_all_random"])
        self.assertTrue(checks["k4_layers_14"]["sensitivity_below_reconstruction"])
        paired = result["paired_differences"]["retrieval"]
        value = paired["online_pre_k4_layers_2_sensitivity_minus_online_pre_k4_layers_2_random_1"]
        self.assertEqual(value["negative_questions"], 4)
        self.assertEqual(value["top_three_absolute_share"], 0.75)
        self.assertEqual(len(result["budget_trends"]["online_pre"]["sensitivity"]), 3)
        rows[0]["kl"]["online_pre_k4_layers_2_random_1"] = None
        result = distribution_summary(rows, formats, seeds)
        self.assertIsNone(result["mean_next_token_kl"]["math"]["online_pre_k4_layers_2_random_1"])

    def test_high_layer_agreement(self) -> None:
        """相关性按固定层序计算，高敏感集合大小固定。"""
        result = rank_agreement([0, 1, 2, 3], [0, 2, 1, 3], 2)
        self.assertEqual(result["spearman_fixed_ties"], 0.8)
        self.assertEqual(result["high_set_overlap"], 1)

    def test_v_is_not_automatically_authorized_by_kl(self) -> None:
        """开发生成与 V 计划都未冻结时，旧真实生成入口必须停止。"""
        self.assertFalse(load_plan()["generation_screen"]["frozen"])
        with patch("experiments.static_layer_allocation.code.protocol.load_plan",
                   return_value={"v_plan": {"frozen": False}}):
            with self.assertRaisesRegex(RuntimeError, "尚未冻结"):
                require_v_plan(Path.cwd())


if __name__ == "__main__":
    unittest.main()
