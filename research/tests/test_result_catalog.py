"""核对四实验统一复用、审核失效与引用来源冲突的长期回归。"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kvstudy.artifacts import file_hash, read_json, write_json
from phase1.entry import _published_run, plan
from phase1.execution import artifact_files, verify_recorded_code
from phase1.results import (
    BASE, EXPERIMENTS, INDEX, ResultCatalog, project_path, quality_path,
    reference_view, reviewed_receipt,
)


class ResultCatalogTest(unittest.TestCase):
    """已有测量只在实体与审核依赖均相同时进入复用，不隐式重测。"""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bindings = patch("phase1.results.review_bindings", return_value={"config": "same"})
        self.signature = patch("phase1.results.computation_signature", return_value="same")
        self.bindings.start()
        self.signature.start()
        self.addCleanup(self.bindings.stop)
        self.addCleanup(self.signature.stop)
        entries = []
        for name in EXPERIMENTS:
            filename = "qwen/v_gsm8k_bf16.json" if name == "generation_error_sources" else (
                "review.json"
            )
            relative = f"{BASE}/{name}/results/{filename}"
            write_json(self.root / relative, {"evidence": name})
            entries.append({"current_path": relative,
                            "current_sha256": file_hash(self.root / relative)})
        self.index = {"files": entries, "shared_capacity_inputs": {"qwen": {"rows": []}}}
        write_json(self.root / INDEX, self.index)
        self.audit = {"status": "passed", "experiments": {name: {} for name in EXPERIMENTS},
                      "formal_result_files": 4, "generation_forward_calls": 0}
        self.index["reuse_review"] = reviewed_receipt(self.root, self.audit)
        write_json(self.root / INDEX, self.index)

    def test_all_completed_experiments_and_generation_cell_reuse(self) -> None:
        catalog = ResultCatalog(self.root)
        for name in EXPERIMENTS:
            self.assertEqual(catalog.decision(name)["action"], "reuse")
        result = plan(self.root, "generation_error_sources", ["qwen"], "gsm8k", "bf16")
        self.assertEqual(result["cells"]["qwen"]["action"], "reuse")
        missing = plan(self.root, "generation_error_sources", ["qwen"], "hotpot", "bf16")
        self.assertEqual(missing["cells"]["qwen"]["action"], "measure")

    def test_dependency_or_shared_input_change_requires_review(self) -> None:
        with patch("phase1.results.review_bindings", return_value={"config": "changed"}):
            self.assertEqual(ResultCatalog(self.root).decision(EXPERIMENTS[0])["action"], "review")
        self.index["shared_capacity_inputs"]["qwen"]["rows"].append({"input_tokens": 200})
        write_json(self.root / INDEX, self.index)
        self.assertEqual(ResultCatalog(self.root).decision(EXPERIMENTS[0])["action"], "review")

    def test_changed_or_extra_upstream_entity_invalidates_transitive_reuse(self) -> None:
        path = self.root / self.index["files"][0]["current_path"]
        write_json(path, {"changed": True})
        self.assertEqual(ResultCatalog(self.root).decision(EXPERIMENTS[-1])["action"], "review")
        write_json(path, {"evidence": EXPERIMENTS[0]})
        write_json(path.parent / "extra.json", {})
        self.assertEqual(ResultCatalog(self.root).decision(EXPERIMENTS[-1])["action"], "review")

    def test_partial_or_generated_audit_cannot_issue_receipt(self) -> None:
        with self.assertRaises(RuntimeError):
            reviewed_receipt(self.root, {**self.audit, "generation_forward_calls": 1})
        with self.assertRaises(RuntimeError):
            reviewed_receipt(self.root, {**self.audit, "formal_result_files": 3})

    def test_completed_run_does_not_load_model_or_rewrite_results(self) -> None:
        write_json(self.root / ".execution.json", {})
        before = {item["current_path"]: file_hash(self.root / item["current_path"])
                  for item in self.index["files"]}
        args = argparse.Namespace(experiment=EXPERIMENTS[0], task=None, condition=None)
        with patch("phase1.execution.verify_publication"), \
                patch("builtins.print"), patch("phase1.runner.load_model") as load:
            _published_run(args, self.root)
        load.assert_not_called()
        self.assertEqual(before, {name: file_hash(self.root / name) for name in before})

    def test_source_paths_and_incomplete_cell_requests_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            project_path(self.root, "../other.json")
        with self.assertRaises(ValueError):
            plan(self.root, EXPERIMENTS[1], ["qwen"], "gsm8k")

    def test_reference_view_rejects_changed_origin_and_wrong_task(self) -> None:
        base = self.root / "base.json"
        rows = [{"id": key, "input_sha256": key, "input_tokens": 10} for key in "ab"]
        write_json(base, {"n": 1, "model": "qwen", "task": "v-gsm8k", "rows": rows[:1]})
        extra = quality_path(self.root, "qwen", "gsm8k", "bf16")
        value = {"n": 1, "model": "qwen", "task": "gsm8k", "condition": "bf16",
                 "rows": rows[1:], "reference_source": {"path": "base.json", "n": 1,
                                                        "sha256": file_hash(base)}}
        write_json(extra, value)
        with patch("phase1.results.reference_path", return_value=base):
            self.assertEqual(reference_view(self.root, "qwen", "gsm8k", "bf16", 2, {})["rows"],
                             rows)
            value["task"] = "hotpot"
            write_json(extra, value)
            with self.assertRaisesRegex(RuntimeError, "身份异常"):
                reference_view(self.root, "qwen", "gsm8k", "bf16", 2, {})
            value["task"] = "gsm8k"
            value["reference_source"]["sha256"] = "different"
            write_json(extra, value)
            with self.assertRaisesRegex(RuntimeError, "来源发生改变"):
                reference_view(self.root, "qwen", "gsm8k", "bf16", 2, {})

    def test_recorded_source_survives_helper_extraction_but_not_arithmetic_change(self) -> None:
        relative = "research/phase1/experiments/storage_service_cost/code/service.py"
        current = self.root / relative
        current.parent.mkdir(parents=True)
        historical_root = self.root / "research/.runtime/recorded/execution_root"
        historical = historical_root / relative
        historical.parent.mkdir(parents=True)
        helper = ("def digest(path):\n"
                  "    return hashlib.sha256(path.read_bytes()).hexdigest()\n")
        calculation = "def calculate(x):\n    return x * 2\n"
        historical.write_text("import hashlib\n" + helper + calculation, "utf-8")
        current.write_text("from kvstudy.artifacts import file_hash as digest\n" + calculation,
                           "utf-8")
        shared = self.root / "research/src/kvstudy/artifacts.py"
        shared.parent.mkdir(parents=True)
        shared.write_text("import hashlib\n" + helper.replace("def digest", "def file_hash"),
                          "utf-8")
        publication = {"source_root": historical_root.relative_to(self.root).as_posix(),
                       "files": {relative: file_hash(historical)}}
        self.index["execution_alignment"] = {"publication": publication}
        write_json(self.root / INDEX, self.index)
        expected = file_hash(historical)
        with patch("phase1.execution.workspace_root", return_value=self.root):
            self.assertEqual(verify_recorded_code(self.root, relative, expected), historical)
            current.write_text(current.read_text().replace("x * 2", "x * 3"), "utf-8")
            with self.assertRaisesRegex(RuntimeError, "不兼容"):
                verify_recorded_code(self.root, relative, expected)


class ArtifactTest(unittest.TestCase):
    """非法新汇总不能覆盖已有有效测量。"""

    def test_nonfinite_write_preserves_original_entity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"value": 1})
            before = file_hash(path)
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertEqual(file_hash(path), before)
            self.assertEqual(read_json(path), {"value": 1})

    def test_publication_contains_result_index_reports_and_cited_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / BASE / EXPERIMENTS[0] / "report/index.md"
            report.parent.mkdir(parents=True)
            report.write_text("来源：`docs/literature-survey.bib`\n", "utf-8")
            bibliography = root / "docs/literature-survey.bib"
            bibliography.parent.mkdir(parents=True)
            bibliography.write_text("@article{source}\n", "utf-8")
            index = root / INDEX
            write_json(index, {"files": []})
            guide = root / "research/phase1/guide.md"
            guide.write_text("指导\n", "utf-8")
            files = set(artifact_files(root))
            self.assertTrue({report, bibliography, index, guide} <= files)
