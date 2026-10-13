"""核验默认收尾入口不会重跑已经结束的 C 实验。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.complete_experiments import main


class StaticEntryTest(unittest.TestCase):
    """常规入口按当前协议路由，未冻结时在任何生成前停止。"""

    def test_default_closeout_does_not_run_c(self) -> None:
        """未额外指定模式时直接恢复独立生成，不调用旧实验完成器。"""
        root = Path("project")
        plan = {"v_plan": {"protocol": "static-allocation-independent-generation"}}
        with patch.object(sys, "argv", ["complete", "--commit", "source", "--models", "qwen"]), \
                patch("phase1.complete_experiments.project_root", return_value=root), \
                patch("experiments.static_layer_allocation.code.protocol.load_plan",
                      return_value=plan), \
                patch("experiments.static_layer_allocation.code.protocol.require_v_plan"), \
                patch("experiments.static_layer_allocation.code.quality.run") as generation, \
                patch("phase1.complete_experiments.ExperimentCompletion") as legacy:
            main()
        generation.assert_called_once_with(root, "qwen")
        legacy.assert_not_called()

    def test_unfrozen_closeout_stops_before_generation(self) -> None:
        """准备中的新协议不能跌回旧流程或启动未经冻结的候选。"""
        plan = {"v_plan": {"protocol": "static-allocation-independent-generation"}}
        with patch.object(sys, "argv", ["complete", "--commit", "source", "--models", "qwen"]), \
                patch("phase1.complete_experiments.project_root", return_value=Path("project")), \
                patch("experiments.static_layer_allocation.code.protocol.load_plan",
                      return_value=plan), \
                patch("experiments.static_layer_allocation.code.protocol.require_v_plan",
                      side_effect=RuntimeError("not frozen")), \
                patch("experiments.static_layer_allocation.code.quality.run") as generation, \
                patch("phase1.complete_experiments.ExperimentCompletion") as legacy:
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                main()
        generation.assert_not_called()
        legacy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
