"""检查候选补测的配对保护与中断后的依赖重新执行。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.complete_experiments import ExperimentCompletion
from experiments.static_layer_allocation.code.protocol import SCHEMA, digest


class ExperimentCompletionTest(unittest.TestCase):
    """防止复用错误输入，以及在新分配写完后复用旧门槛。"""

    def test_reused_candidate_must_have_same_input(self) -> None:
        """相同题号但不同全文的旧结果不能跳过补测。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "research/phase1/experiments/generation_error_sources/results/qwen"
            folder.mkdir(parents=True)
            config = "k8v4-per_channel-pre"
            candidate = {
                "model": "qwen", "config": config, "n": 1,
                "rows": [{"id": "question", "input_sha256": "wrong", "input_tokens": 8,
                          "correct": True}],
            }
            reference = {
                "n": 1,
                "rows": [{"id": "question", "input_sha256": "correct", "input_tokens": 8,
                          "correct": True}],
            }
            (folder / f"v_gsm8k_{config}.json").write_text(json.dumps(candidate))
            (folder / "v_gsm8k_kv8-per_token-post.json").write_text(json.dumps(reference))
            seeds = root / "research/phase1/splits/seeds.json"
            seeds.parent.mkdir(parents=True)
            seeds.write_text('{"bootstrap": 1, "bootstrap_draws": 100}')
            with patch("phase1.complete_experiments.project_root", return_value=root):
                completion = ExperimentCompletion("new")
            with self.assertRaisesRegex(RuntimeError, "配对输入不一致"):
                completion._generation("qwen", "gsm8k", config)

    def test_old_allocation_cannot_reuse_gate_or_trigger_v(self) -> None:
        """旧 K/V 位宽表必须重算三格式 C，但不能继续调用 V。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "research/phase1/experiments/static_layer_allocation/results/qwen"
            folder.mkdir(parents=True)
            name = "k8v4-per_channel-pre"
            fixtures = {
                "probe.json": {"selected_representation": name},
                "allocations.json": {"representation": name, "residual_bits": 0,
                                     "commit": "new"},
                "gate.json": {"allocation_commit": "old", "commit": "old",
                              "selected_nonrandom_for_v": ["p50_sensitivity"],
                              "random_for_v": []},
            }
            for filename, payload in fixtures.items():
                (folder / filename).write_text(json.dumps(payload))
            with patch("phase1.complete_experiments.project_root", return_value=root):
                completion = ExperimentCompletion("new")
            with patch.object(completion, "_run") as run, patch.object(
                completion, "_fresh_progress"
            ), patch("experiments.static_layer_allocation.code.protocol.load_plan",
                     return_value={"v_plan": {"frozen": False}}):
                completion._layers("qwen", {"e3_representation": name,
                                            "e3_residual_bits": 0})
            modules = [call.args[0] for call in run.call_args_list]
            self.assertIn("experiments.static_layer_allocation.code.gate", modules)
            self.assertNotIn("experiments.static_layer_allocation.code.quality", modules)
            self.assertNotIn("experiments.static_layer_allocation.code.select", modules)
            self.assertEqual(completion.state["downstream_status"],
                             "await_frozen_key_allocation_v_plan")

    def test_current_probe_and_bits_resume_only_kl(self) -> None:
        """已冻结的探针和位宽表有效时，不因 KL 中断重算排名或触发 V。"""
        plan = {"v_plan": {"frozen": False}}
        sources = {"code": "current"}
        probe = {
            "schema": SCHEMA, "plan_sha256": digest(plan), "code_sha256": sources,
            "frozen_sources": {}, "calibration_source": {},
        }
        allocation = {"schema": SCHEMA, "plan_sha256": digest(plan),
                      "probe_sha256": digest(probe)}
        allocation["allocation_digest"] = digest(allocation)
        gate = {"schema": SCHEMA, "plan_sha256": digest(plan),
                "allocation_digest": "old", "code_sha256": sources}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "research/phase1/experiments/static_layer_allocation/results/qwen"
            folder.mkdir(parents=True)
            for name, payload in (("probe", probe), ("allocations", allocation), ("gate", gate)):
                (folder / f"{name}.json").write_text(json.dumps(payload))
            with patch("phase1.complete_experiments.project_root", return_value=root):
                completion = ExperimentCompletion("new")
            completion.state["completed_steps"] = [
                "experiments.static_layer_allocation.code.gate qwen --commit new"
            ]
            prefix = "experiments.static_layer_allocation.code.protocol"
            with patch.object(completion, "_run") as run, patch.object(
                completion, "_fresh_progress"
            ), patch(prefix + ".load_plan", return_value=plan), patch(
                prefix + ".code_sources", return_value=sources
            ), patch(prefix + ".frozen_sources", return_value={}), patch(
                prefix + ".calibration_source", return_value={}
            ):
                completion._layers("qwen", {})
            run.assert_called_once_with(
                "experiments.static_layer_allocation.code.gate", "qwen", "--commit", "new"
            )
            self.assertEqual(completion.state["completed_steps"], [])


if __name__ == "__main__":
    unittest.main()
