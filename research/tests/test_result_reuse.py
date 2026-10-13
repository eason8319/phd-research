"""核验跨实验复用、来源不足、重复参考归并和累计输出恢复。"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from phase1.reuse import (
    _Computation, ReuseDecision, assess_reuse, function_signature, require_missing_result,
    validate_reference_prefix,
)
from experiments.static_layer_allocation.code.quality import _saved_predictions
from experiments.static_layer_allocation.code.quality_review import _outputs


class ResultReuseTest(unittest.TestCase):
    """覆盖可能造成重复生成或错误混用已有结果的分支。"""

    def setUp(self) -> None:
        self.rows = [{"id": "a", "input_sha256": "input", "input_tokens": 100,
                      "new_tokens": 20, "correct": True, "invalid_groups": 0,
                      "stop": None}]
        self.result = {"model": "qwen", "config": "bf16-h0", "n": 1,
                       "rows": self.rows, "residual_bits": 0}
        self.evidence = {"status": "passed", "source_sha256": "entity",
                         "computation_signature": "computation", "device": {"gpu": "Ada"}}

    def decide(self, **changes: object) -> ReuseDecision:
        arguments = {"model": "qwen", "configuration": "bf16", "evidence": self.evidence,
                     "source_sha256": "entity", "signature": "computation"}
        arguments.update(changes)
        return assess_reuse(self.result, self.rows, **arguments)

    def test_missing_provenance_needs_review_and_never_requests_rerun(self) -> None:
        decision = self.decide(evidence=None)
        self.assertEqual(decision.action, "review")
        self.assertEqual(decision.missing_n, 0)

    def test_verified_prefix_only_requests_missing_samples(self) -> None:
        extra = {**self.rows[0], "id": "b", "input_sha256": "next"}
        decision = assess_reuse(
            self.result, [*self.rows, extra], model="qwen", configuration="bf16",
            evidence=self.evidence, source_sha256="entity", signature="computation",
        )
        self.assertEqual((decision.action, decision.completed_n, decision.missing_n),
                         ("append", 1, 1))

    def test_same_sample_id_with_different_text_cannot_be_reused(self) -> None:
        changed = [{**self.rows[0], "input_sha256": "different"}]
        decision = assess_reuse(
            self.result, changed, model="qwen", configuration="bf16",
            evidence=self.evidence, source_sha256="entity", signature="computation",
        )
        self.assertEqual(decision.action, "remeasure")

    def test_old_entity_validation_does_not_validate_changed_file_or_arithmetic(self) -> None:
        self.assertEqual(self.decide(source_sha256="changed").action, "review")
        self.assertEqual(self.decide(signature="changed").action, "review")

    def test_node_job_and_device_number_do_not_force_reference_rerun(self) -> None:
        self.evidence["device"].update(node="node8", slurm_job="old", visible_device="0")
        decision = self.decide(device={"gpu": "Ada", "node": "node8", "slurm_job": "new",
                                       "visible_device": "1"})
        self.assertEqual(decision.action, "reuse")
        self.assertEqual(self.decide(device={"gpu": "other"}).action, "review")

    def test_duplicate_prefix_keeps_unknown_stop_and_rejects_length_difference(self) -> None:
        fresh = {"rows": [{**self.rows[0], "stop": "eos"}]}
        record = validate_reference_prefix(self.result, fresh)
        self.assertEqual(record["historical_unknown_stops"], 1)
        self.assertIsNone(self.result["rows"][0]["stop"])
        fresh["rows"][0]["new_tokens"] = 21
        with self.assertRaisesRegex(RuntimeError, "测量差异"):
            validate_reference_prefix(self.result, fresh)

    def test_function_moves_and_docstrings_keep_signature_but_arithmetic_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.py"
            new = Path(directory) / "new.py"
            old.write_text('def before(x):\n    """旧说明。"""\n    return x * 4\n', "utf-8")
            new.write_text('def after(x):\n    """新说明。"""\n    return x * 4\n', "utf-8")
            self.assertEqual(function_signature(old, "before"), function_signature(new, "after"))
            new.write_text('def after(x):\n    return x * 8\n', "utf-8")
            self.assertNotEqual(function_signature(old, "before"),
                                function_signature(new, "after"))

    def test_empty_parser_version_fields_do_not_invalidate_measurements(self) -> None:
        first = _Computation().visit(ast.parse("def calculate(x):\n    return x * 4\n"))
        second = ast.parse("def calculate(x):\n    return x * 4\n")
        node = second.body[0]
        if "type_params" not in node._fields:
            node._fields = (*node._fields, "type_params")
        node.type_params = []
        self.assertEqual(ast.dump(first, include_attributes=False),
                         ast.dump(_Computation().visit(second), include_attributes=False))

    def test_reference_suffix_reads_only_its_original_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.json"
            predictions = [{"id": "prefix", "text": "old", "token_ids": [1]},
                           {"id": "extra", "text": "new", "token_ids": [2]}]
            raw.write_text(json.dumps({"predictions": predictions}), "utf-8")
            payload = {"n": 1, "rows": [{"id": "extra"}], "raw_selection": {"ids": ["extra"]},
                       "raw_source": {"path": "raw.json", "sha256": hashlib.sha256(
                           raw.read_bytes()).hexdigest()}}
            self.assertEqual(_outputs(root, payload), {"extra": predictions[1]})
            self.assertEqual(_saved_predictions(root, payload), [predictions[1]])
            raw.write_text("{}", "utf-8")
            with self.assertRaisesRegex(RuntimeError, "校验"):
                _outputs(root, payload)

    def test_independent_entry_does_not_overwrite_existing_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            require_missing_result(path)
            path.write_text("historical measurement", "utf-8")
            with self.assertRaisesRegex(RuntimeError, "不能重复生成"):
                require_missing_result(path)
            self.assertEqual(path.read_text("utf-8"), "historical measurement")

    def test_unverified_existing_generation_is_reviewed_instead_of_rerun(self) -> None:
        from phase1.entry import plan

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ("research/phase1/experiments/generation_error_sources/results/"
                           "qwen/v_gsm8k_bf16.json")
            path.parent.mkdir(parents=True)
            path.write_text("{}", "utf-8")
            proposal = plan(root, "generation_error_sources", ["qwen"], "gsm8k", "bf16")
            self.assertEqual(proposal["cells"]["qwen"]["action"], "review")

    def test_missing_checkpoint_keeps_old_predictions_and_actual_source_segment(self) -> None:
        from contextlib import ExitStack
        from experiments.static_layer_allocation.code import quality

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            path = quality.result_path(root, "qwen", "gsm8k", "mixed")
            path.parent.mkdir(parents=True)
            config = {"name": "mixed"}
            old_publication = {"actual_source": "old"}
            new_publication = {"actual_source": "new"}
            device = {"gpu": "same", "node": "same"}
            row = {"id": "old", "correct": True, "new_tokens": 1, "invalid_groups": 0}
            value = {"rows": [row], "n": 1, "config_digest": quality.digest(config),
                     "device": device, "publication": old_publication}
            path.write_text(json.dumps(value), "utf-8")
            tokenizer = MagicMock()
            tokenizer.decode.return_value = "#### 42"
            overrides = {
                "load_plan": {"v_plan": {"frozen": True, "protocol": "test",
                                         "tasks": {"gsm8k": {"max_new_tokens": 1024}}}},
                "totals": [1, 2], "samples": [{"id": "old"}, {"id": "new", "answer": "42"}],
                "config_for": config, "_device": device,
                "current_publication": new_publication,
                "_saved_predictions": [{"id": "old", "text": "saved", "token_ids": [1]}],
            }
            for name, returned in overrides.items():
                stack.enter_context(patch.object(quality, name, return_value=returned))
            stack.enter_context(patch.object(quality, "verify_publication"))
            stack.enter_context(patch.object(quality, "validate_segments"))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained",
                                      return_value=tokenizer))
            stack.enter_context(patch("phase1.runner.load_model"))
            stack.enter_context(patch("phase1.models._eos_ids", return_value={9}))
            stack.enter_context(patch("phase1.inputs._ids_and_source", return_value=([2], "new")))
            stack.enter_context(patch("phase1.generation._generate", return_value=([9], "eos", 0)))
            stack.enter_context(patch("phase1.score.gsm8k_exact", return_value=True))
            for name, returned in (("mem_get_info", (100 * 1024 ** 3, 100 * 1024 ** 3)),
                                   ("reset_peak_memory_stats", None), ("max_memory_allocated", 0),
                                   ("max_memory_reserved", 0), ("empty_cache", None)):
                stack.enter_context(patch(f"torch.cuda.{name}", return_value=returned))
            quality.measure(root, "qwen", "gsm8k", "mixed", 2)
            measured = json.loads(path.read_text("utf-8"))
            self.assertEqual(measured["reused_prefix"]["n"], 0)
            self.assertEqual([segment["publication"] for segment in measured["source_segments"]],
                             [old_publication, new_publication])
            raw = json.loads((root / measured["raw_source"]["path"]).read_text("utf-8"))
            self.assertEqual([output["id"] for output in raw["predictions"]], ["old", "new"])


if __name__ == "__main__":
    unittest.main()
