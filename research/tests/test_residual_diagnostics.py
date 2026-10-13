"""核对条件补测、独立执行和配对方向，防止诊断改写下游选择。"""

import unittest
from unittest.mock import patch

from experiments.generation_error_sources.code.residual_diagnostics import (
    ResidualDiagnostics, h8_decision, paired_gain,
)
from experiments.generation_error_sources.code.select_residual import _capacity, _rank


def cells(passes: bool, estimate: float, lower: float | None) -> dict:
    """构造两个有效任务的最小决策输入。"""
    return {task: {
        "0": {"class_unit_passes": False},
        "16": {"class_unit_passes": passes,
               "gain_vs_h0": {"estimate": estimate, "lower": lower}},
    } for task in ("gsm8k", "hotpot")}


class ResidualDiagnosticsTest(unittest.TestCase):
    """检查预登记触发边界与当前选择保护。"""

    def test_class_rescue_triggers_without_positive_interval(self) -> None:
        self.assertTrue(h8_decision(cells(True, 1, -3), False)["run"])

    def test_supported_gain_triggers_without_class_rescue(self) -> None:
        self.assertTrue(h8_decision(cells(False, 5, 0.1), False)["run"])

    def test_point_only_or_touching_zero_does_not_trigger(self) -> None:
        for lower in (None, -1, 0):
            decision = h8_decision(cells(False, 1, lower), False)
            self.assertFalse(decision["run"])
            self.assertIn("待确认", decision["reason"])

    def test_existing_negative_result_is_retained(self) -> None:
        decision = h8_decision(cells(False, -3, -8), True)
        self.assertFalse(decision["run"])
        self.assertTrue(decision["existing_retained"])

    def test_gain_reverses_interval_and_keeps_paired_disagreements(self) -> None:
        old = {"n": 4, "rows": [{"id": str(i), "correct": x}
                                for i, x in enumerate([True, False, False, False])]}
        new = {"n": 4, "rows": [{"id": str(i), "correct": x}
                                for i, x in enumerate([True, True, True, False])]}
        gain = paired_gain(old, new, "gsm8k", 20260923, 10000)
        self.assertEqual(gain["estimate"], 50)
        self.assertEqual(gain["detail"], {"degrade": 0, "improve": 2})
        self.assertLess(gain["lower"], gain["upper"])

    def test_cli_diagnostic_mode_never_calls_full_completion(self) -> None:
        from phase1.complete_experiments import main

        with patch("sys.argv", ["complete", "--commit", "source", "--residual-diagnostics"]), \
                patch("experiments.generation_error_sources.code.residual_diagnostics.run") \
                as diagnostic, patch("phase1.complete_experiments.ExperimentCompletion.run") \
                as full:
            main()
        diagnostic.assert_called_once_with("source", ["qwen", "llama"])
        full.assert_not_called()

    def test_residual_run_contains_no_downstream_calls(self) -> None:
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(ResidualDiagnostics.run_diagnostics)))
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"run_model", "_layers", "_diagnostics", "_expansion", "_run"})

    def test_joint_quality_precedes_capacity_and_capacity_breaks_ties(self) -> None:
        entries = [
            {"representation": "small", "H": 0, "maximum_drop": 1, "complete_bytes": 10},
            {"representation": "residual", "H": 16, "maximum_drop": 0,
             "complete_bytes": 30},
            {"representation": "plain", "H": 0, "maximum_drop": 0, "complete_bytes": 20},
        ]
        self.assertEqual([row["representation"] for row in sorted(entries, key=_rank)],
                         ["plain", "residual", "small"])

    def test_joint_capacity_clips_short_windows_and_counts_copy(self) -> None:
        model = {"num_hidden_layers": 1, "num_key_value_heads": 1,
                 "hidden_size": 32, "num_attention_heads": 1}
        name = "k8v4-per_token-pre"
        base = _capacity(name, 0, model, [64, 256])
        self.assertEqual(_capacity(name, 16, model, [64, 256]) - base,
                         2 * 32 * 2 * (64 + 128))
        self.assertGreater(_capacity(name, 8, model, [64, 256]), base)
        self.assertLess(_capacity(name, 8, model, [64, 256]),
                        _capacity(name, 16, model, [64, 256]))

    def test_cli_joint_mode_updates_selection_without_generation(self) -> None:
        from phase1.complete_experiments import main

        with patch("sys.argv", ["complete", "--commit", "source", "--joint-selection"]), \
                patch("experiments.generation_error_sources.code.select_residual.run") as choose, \
                patch("phase1.complete_experiments.ExperimentCompletion.run") as full:
            main()
        self.assertEqual(choose.call_args_list, [
            unittest.mock.call("qwen", "source"),
            unittest.mock.call("llama", "source"),
        ])
        full.assert_not_called()


if __name__ == "__main__":
    unittest.main()
