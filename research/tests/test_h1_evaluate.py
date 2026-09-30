"""H1 统一入口在合成 CPU GQA 轨迹上的集成回归测试。"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import torch

from kvstudy.groups import fit_per_channel
from experiments.representation_interactions.code import evaluate


class H1EvaluateTest(unittest.TestCase):
    """用真实量化器、两折和 GQA 验证诊断、原矩阵与默认保护路径。"""

    def setUp(self) -> None:
        """构造两题、两层、两个 KV head 与四个 query head 的 CPU 轨迹。"""
        generator = torch.Generator().manual_seed(71)
        self.captured = []
        for fold in (0, 1):
            keys = [torch.randn(2, 7, 8, generator=generator) for _ in range(2)]
            self.captured.append({
                "id": f"sample-{fold}", "kind": "math", "fold": fold,
                "source_sha256": str(fold + 1) * 64, "seq": 6, "token": 1,
                "pre": keys, "post": [key.clone() for key in keys],
                "value": [torch.randn(2, 7, 8, generator=generator) for _ in range(2)],
                "query": [torch.randn(4, 8, generator=generator) for _ in range(2)],
            })
        self.settings = {
            "measurement_version": "h1-paired-v2", "top_k": 3,
            "bootstrap_seed": 17, "bootstrap_resamples": 20,
        }
        self.expected = {blob["id"] for blob in self.captured}

    def _rows(self, refresh_matrix: bool) -> list[dict[str, Any]]:
        """只替换模型 RoPE 操作；校准、量化、分数和指标均使用真实实现。"""
        scales, _full = evaluate._fit_scales(self.captured, refresh_matrix)
        with patch.object(evaluate, "_rotate_key", side_effect=lambda model, key: key), \
                patch.object(
                    evaluate, "_rotate_query", side_effect=lambda model, query, position: query
                ):
            return [
                row for blob in self.captured
                for row in evaluate._score_item(
                    blob, None, scales, self.settings, refresh_matrix
                )
            ]

    def _matrix(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """从内存中的完整明细汇总矩阵，测试不写临时文件。"""
        with patch.object(evaluate, "_read_rows", return_value=iter(rows)):
            return evaluate._matrix_payload(Path("unused.jsonl"), "measured", self.expected)

    def test_calibration_fold_and_downstream_keys(self) -> None:
        """校准参数保持按折分离，all-C 包维持下游原键名。"""
        scales, no_full = evaluate._fit_scales(self.captured, False)
        self.assertIsNone(no_full)
        self.assertEqual(set(scales[0]), {"k_pre", "k_pre4", "k_post", "k_post4"})
        for fold in (0, 1):
            source = self.captured[fold]["pre"][0]
            extrema = torch.stack((source.amin(dim=1), source.amax(dim=1)), dim=1)
            expected = fit_per_channel(extrema, 4)
            for actual, reference in zip(scales[fold]["k_pre4"][0], expected):
                self.assertTrue(torch.equal(actual, reference))
        refreshed, full = evaluate._fit_scales(self.captured, True)
        self.assertIn("v4", refreshed[0])
        self.assertEqual(set(full), {"k_pre8", "k_pre4", "k_post8", "k_post4", "v8", "v4"})

    def test_paired_schema_and_output_decomposition(self) -> None:
        """默认四种配对只存核心分数指标，整层误差分解和竞争记录完整。"""
        rows = self._rows(False)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertNotIn("matrix", row)
            self.assertEqual(len(row["layers"]), 2)
            for layer in row["layers"]:
                self.assertEqual(len(layer["heads"]), 4)
                for head in layer["heads"]:
                    self.assertEqual(set(head), set(evaluate.HEAD_FIELDS))
                    self.assertGreaterEqual(head["paired_probability_kl"], 0)
                self.assertAlmostEqual(layer["output"]["difference_residual"], 0, places=12)
                self.assertIn("key_prediction_relative_error", layer["output"])
                self.assertEqual(len(layer["competition"]["pair_rows"]), 12)

    def test_full_matrix_has_17_cells_and_cumulative_bytes(self) -> None:
        """主矩阵仍为 17 格，并累加各题容量而不改成题均容量。"""
        rows = self._rows(True)
        payload = self._matrix(rows)
        self.assertEqual(len(payload["summary"]), 17)
        self.assertEqual(payload["commit"], "measured")
        for cfg in evaluate._matrix_configs():
            expected = sum(evaluate._matrix_bytes(cfg, blob) for blob in self.captured)
            entry = payload["summary"][cfg["name"]]
            self.assertEqual(entry["bytes"], expected)
            self.assertEqual(entry["zero_reference_outputs"], 0)
            self.assertGreaterEqual(entry["error"], 0)

    def test_missing_duplicate_or_incomplete_matrix_fails(self) -> None:
        """缺格、重复题目和缺少冻结题目都不能生成正式主矩阵。"""
        rows = self._rows(True)
        missing_cell = copy.deepcopy(rows)
        del missing_cell[0]["matrix"][next(iter(missing_cell[0]["matrix"]))]
        duplicate = rows + [copy.deepcopy(rows[0])]
        missing_sample = [row for row in rows if row["id"] != "sample-1"]
        for broken in (missing_cell, duplicate, missing_sample):
            with self.subTest(kind=len(broken)), self.assertRaises(RuntimeError):
                self._matrix(broken)

    def test_nonfinite_errors_or_invalid_counts_fail(self) -> None:
        """非有限数、负误差和非整数无效组计数不会进入矩阵或选择器。"""
        rows = self._rows(True)
        for field, value in (
            ("error", float("nan")), ("error", float("inf")), ("error", -1),
            ("invalid", float("nan")), ("invalid", 0.5),
        ):
            broken = copy.deepcopy(rows)
            matrix = next(iter(broken[0]["matrix"].values()))
            if field == "error":
                matrix["layer_errors"][0] = value
            else:
                matrix["invalid"] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                self._matrix(broken)

    def test_zero_reference_layers_are_counted_and_all_zero_fails(self) -> None:
        """部分零参考层跳过并计数；整题无可定义误差时不能伪装为完美零误差。"""
        rows = self._rows(True)
        partial = copy.deepcopy(rows)
        name = next(iter(partial[0]["matrix"]))
        partial[0]["matrix"][name]["layer_errors"][0] = None
        partial[0]["matrix"][name]["zero_reference_outputs"] = 1
        payload = self._matrix(partial)
        self.assertEqual(payload["summary"][name]["zero_reference_outputs"], 1)
        all_zero = copy.deepcopy(rows)
        all_zero[0]["matrix"][name]["layer_errors"] = [None, None]
        all_zero[0]["matrix"][name]["zero_reference_outputs"] = 2
        with self.assertRaises(RuntimeError):
            self._matrix(all_zero)

    def test_default_preserves_frozen_matrix_and_calibration(self) -> None:
        """默认诊断不刷新矩阵、不改冻结提交，也不写下游 all-C 校准包。"""
        frozen = {
            "protocol": "phase1-protocol v7", "commit": "frozen",
            "summary": {
                cfg["name"]: {"error": 1.0, "invalid": 0.0, "bytes": 12.0}
                for cfg in evaluate._matrix_configs()
            },
            "kept": ["k8v4-per_token-post"], "reason": "frozen-selection",
        }
        expected_items = {
            blob["id"]: {name: blob[name] for name in ("kind", "fold", "source_sha256")}
            for blob in self.captured
        }
        with patch.object(evaluate, "project_root", return_value=Path("virtual-project")), \
                patch.object(evaluate, "_settings", return_value=self.settings), \
                patch.object(evaluate.AutoTokenizer, "from_pretrained", return_value=object()), \
                patch.object(evaluate, "load_model", return_value=object()), \
                patch.object(evaluate, "_capture_items", return_value=self.captured), \
                patch.object(evaluate, "_rotate_key", side_effect=lambda model, key: key), \
                patch.object(
                    evaluate, "_rotate_query", side_effect=lambda model, query, position: query
                ), \
                patch.object(evaluate, "_expected_items", return_value=expected_items), \
                patch.object(evaluate, "_sha256", return_value="detail-hash"), \
                patch.object(evaluate, "_code_hashes", return_value={"module": "code-hash"}), \
                patch.object(evaluate, "_write_json") as write_manifest, \
                patch.object(evaluate, "_finalize") as finalize, \
                patch.object(Path, "read_text", return_value=json.dumps(frozen)), \
                patch.object(Path, "open", mock_open()), \
                patch.object(Path, "replace"), patch.object(torch, "save") as save, \
                patch.object(torch.cuda, "empty_cache"), patch("builtins.print"):
            evaluate.run_model("qwen", "diagnostic-only")
        manifest = write_manifest.call_args.args[1]
        self.assertEqual(manifest["matrix_payload"], frozen)
        self.assertEqual(manifest["provenance"]["source_summary_commit"], "frozen")
        self.assertFalse(manifest["provenance"]["main_matrix_refreshed"])
        self.assertEqual(manifest["measurement_status"]["main_matrix"]["commit"], "frozen")
        save.assert_not_called()
        finalize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
