"""位宽与表示交互题目级汇总、配对完整性及冻结主矩阵的回归检查。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.representation_interactions.code.results import (
    MATRIX_CONFIGS,
    PAIRS,
    SCHEMA_VERSION,
    summarize_diagnostics,
    summarize_rows,
    summarize_samples,
    write_results,
)


def make_rows(folds: tuple[int, ...] = (0, 0, 1, 1)) -> list[dict]:
    """生成同题同层配对记录，head 波动不作为额外统计样本。"""
    rows = []
    for pair in PAIRS:
        for index, fold in enumerate(folds):
            heads = [
                {
                    "weighted_score_var": index * 2.0 + offset,
                    "paired_probability_kl": (index * 2.0 + offset) / 2,
                    "optional": None,
                }
                for offset in (0.0, 100.0)
            ]
            key = index + 2.0
            value = 1.0
            rows.append({
                "id": f"sample-{index}", "kind": "math", "fold": fold,
                "source_sha256": f"{index:064x}", "pair": pair,
                "layers": [{
                    "layer": 0, "heads": heads,
                    "output": {
                        "key_increment_error": key,
                        "value_increment_error": value,
                        "difference_residual": 0.0,
                        "zero_reference_output": False,
                    },
                }],
            })
    return rows


def make_matrix() -> dict:
    """构造完整主矩阵及冻结选择，验证保存时数值不会被诊断覆盖。"""
    return {
        "protocol": "phase1-full-key-retrieval", "commit": "6ca63fd",
        "summary": {
            name: {"error": index / 100.0, "invalid": 0.0, "bytes": float(index + 10)}
            for index, name in enumerate(MATRIX_CONFIGS)
        },
        "kept": ["k8v4-per_channel-post", "kv4-per_channel-pre"],
        "reason": "6bit+4bit",
    }


class SummarizeResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = make_rows()
        self.ids = {f"sample-{index}" for index in range(4)}

    def test_statistics_use_sample_means_and_keep_pairing(self) -> None:
        metrics, samples = summarize_rows(self.rows, self.ids, resamples=100)
        self.assertEqual(len(samples), 16)
        result = metrics[PAIRS[0]]
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["layers"], 4)
        self.assertEqual(result["query_heads"], 8)
        self.assertEqual(result["head_metric_means"]["weighted_score_var"], 53.0)
        self.assertEqual(result["key_increment_greater_samples"], 4)
        self.assertEqual(result["key_increment_comparable_samples"], 4)
        self.assertEqual(result["key_increment_minus_value_ci95"]["samples"], 4)
        self.assertEqual(result["key_increment_minus_value_ci95"]["estimate"], 2.5)
        self.assertEqual(result["paired_kl_over_half_weighted_variance"], 1.0)
        self.assertNotIn("layers", samples[0])

    def test_overall_mean_and_bootstrap_keep_fold_weights(self) -> None:
        rows = make_rows((0, 1, 1, 1))
        metrics, _samples = summarize_rows(rows, self.ids, bootstrap_seed=4, resamples=50)
        result = metrics[PAIRS[0]]
        self.assertEqual(result["head_metric_means"]["weighted_score_var"], 52.0)
        ci = result["key_increment_minus_value_ci95"]
        self.assertEqual(ci["estimate"], 2.0)
        self.assertEqual(ci["seed"], 4)
        self.assertEqual(ci["draws"], 50)
        repeated, _samples = summarize_rows(rows, self.ids, bootstrap_seed=4, resamples=50)
        self.assertEqual(ci, repeated[PAIRS[0]]["key_increment_minus_value_ci95"])

    def test_rejects_duplicate_and_missing_pairs(self) -> None:
        for rows in (self.rows + [self.rows[0]], self.rows[:-1]):
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                summarize_rows(rows, self.ids, resamples=5)

    def test_rejects_changed_input_metadata(self) -> None:
        changed = copy.deepcopy(self.rows)
        changed[4]["source_sha256"] = "different"
        with self.assertRaises(ValueError):
            summarize_rows(changed, self.ids, resamples=5)
        frozen = {sample_id: {"fold": 1} for sample_id in self.ids}
        with self.assertRaises(ValueError):
            summarize_rows(self.rows, frozen, resamples=5)

    def test_rejects_missing_layers_and_nonfinite_data(self) -> None:
        changed = copy.deepcopy(self.rows)
        changed[0]["layers"][0]["layer"] = 1
        with self.assertRaises(ValueError):
            summarize_rows(changed, self.ids, resamples=5)
        for group in ("heads", "output"):
            changed = copy.deepcopy(self.rows)
            if group == "heads":
                changed[0]["layers"][0]["heads"][0]["weighted_score_var"] = math.nan
            else:
                changed[0]["layers"][0]["output"]["key_increment_error"] = math.inf
            with self.subTest(group=group), self.assertRaises(ValueError):
                summarize_rows(changed, self.ids, resamples=5)

    def test_null_metrics_remain_undefined_and_report_valid_counts(self) -> None:
        result = summarize_rows(self.rows, self.ids, resamples=5)[0][PAIRS[0]]
        self.assertIsNone(result["head_metric_means"]["optional"])
        self.assertEqual(result["head_metric_valid_samples"]["optional"], 0)

    def test_sample_summary_does_not_fabricate_layer_or_head_counts(self) -> None:
        _metrics, samples = summarize_rows(self.rows, self.ids, resamples=5)
        result = summarize_samples(samples, self.ids, resamples=5)[PAIRS[0]]
        self.assertNotIn("layers", result)
        self.assertNotIn("query_heads", result)
        self.assertEqual(result["head_metric_means"]["weighted_score_var"], 53.0)
        with self.assertRaises(ValueError):
            summarize_samples(samples[:-1], self.ids, resamples=5)

    def test_residual_errors_cannot_cancel_between_samples_or_layers(self) -> None:
        changed = copy.deepcopy(self.rows)
        changed[0]["layers"][0]["output"]["difference_residual"] = 1e-4
        changed[1]["layers"][0]["output"]["difference_residual"] = -1e-4
        with self.assertRaises(ValueError):
            summarize_rows(changed, self.ids, resamples=5)
        _metrics, samples = summarize_rows(self.rows, self.ids, resamples=5)
        samples[0]["layer_output_means"]["difference_residual"] = 1e-4
        samples[1]["layer_output_means"]["difference_residual"] = -1e-4
        with self.assertRaises(ValueError):
            summarize_samples(samples, self.ids, resamples=5)

    def test_zero_reference_keeps_valid_independently_normalized_metrics(self) -> None:
        rows = copy.deepcopy(self.rows)
        for row in rows:
            output = row["layers"][0]["output"]
            output.pop("zero_reference_output")
            output.update({
                "zero_reference": True, "reference_energy": 0.0,
                "key_increment_error": None, "value_increment_error": None,
                "difference_residual": None, "key_prediction_relative_error": 0.5,
            })
        result = summarize_rows(rows, self.ids, resamples=5)[0][PAIRS[0]]
        self.assertEqual(result["zero_reference_outputs"], 4)
        self.assertIsNone(result["layer_output_means"]["key_increment_error"])
        self.assertEqual(result["layer_output_means"]["key_prediction_relative_error"], 0.5)
        self.assertIsNone(result["key_increment_minus_value_ci95"]["lower"])
        self.assertEqual(result["key_increment_comparable_samples"], 0)


class LayerStatisticsTest(unittest.TestCase):
    """逐层统计保留折权重、未定义值和完整配对，不借用历史审核字段。"""

    def setUp(self) -> None:
        self.rows = make_rows((0, 1, 1, 1))
        self.ids = {f"sample-{index}" for index in range(4)}
        for row in self.rows:
            index = int(row["id"].split("-")[-1])
            output = row["layers"][0]["output"]
            r2 = index / 2.0
            output.update({
                "key_prediction_error": output["key_increment_error"] * r2,
                "key_prediction_relative_error": r2, "key_prediction_cosine": 0.5,
            })
            second = copy.deepcopy(row["layers"][0])
            second["layer"] = 1
            second["output"]["key_increment_error"] *= 2
            second["output"]["key_prediction_error"] *= 2
            row["layers"].append(second)

    def test_layer_statistics_use_equal_fold_weights_and_actual_extremes(self) -> None:
        metrics, samples, checks = summarize_diagnostics(iter(self.rows), self.ids, resamples=5)
        self.assertEqual((metrics, samples), summarize_rows(self.rows, self.ids, resamples=5))
        self.assertEqual(checks["rows"], 16)
        pair = checks["pairs"][PAIRS[0]]
        self.assertEqual(pair["layer_means"], [[0, 3.0, 13 / 6, 0.5, 0.5],
                                               [1, 6.0, 13 / 3, 0.5, 0.5]])
        self.assertEqual(pair["layer_valid_samples"], [[0, 4, 4, 4, 4], [1, 4, 4, 4, 4]])
        self.assertEqual(pair["r2_gt_one_layers"], 2)
        self.assertEqual(pair["max_r2_location"]["id"], "sample-3")
        self.assertEqual(pair["max_r2_location"]["r2"], 1.5)
        self.assertEqual(pair["max_output_decomposition_error"], 0.0)

    def test_undefined_fold_is_not_reweighted_or_counted_as_zero(self) -> None:
        self.rows[0]["layers"][0]["output"]["key_prediction_cosine"] = None
        checks = summarize_diagnostics(self.rows, self.ids, resamples=5)[2]
        pair = checks["pairs"][PAIRS[0]]
        self.assertIsNone(pair["layer_means"][0][-1])
        self.assertEqual(pair["layer_valid_samples"][0][-1], 3)

    def test_prediction_identity_and_required_fields_are_checked(self) -> None:
        for broken in ("identity", "missing"):
            rows = copy.deepcopy(self.rows)
            output = rows[0]["layers"][0]["output"]
            if broken == "identity":
                output["key_prediction_error"] = 10.0
            else:
                for row in rows:
                    for layer in row["layers"]:
                        del layer["output"]["key_prediction_cosine"]
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                summarize_diagnostics(rows, self.ids, resamples=5)

    def test_incomplete_pairs_cannot_publish_layer_statistics(self) -> None:
        with self.assertRaises(ValueError):
            summarize_diagnostics(self.rows[:-1], self.ids, resamples=5)


class WriteResultsTest(unittest.TestCase):
    def test_preserves_all_original_matrix_fields_and_sample_list(self) -> None:
        matrix = make_matrix()
        before = copy.deepcopy(matrix)
        metrics, samples = summarize_rows(make_rows(), {f"sample-{i}" for i in range(4)},
                                           resamples=5)
        diagnostic = {
            "model": "qwen", "measurement_version": "paired-output-diagnostics",
            "provenance": {"detail_path": "research/.runtime/e1-paired-qwen.jsonl"},
            "metrics": metrics,
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            folder = Path(temporary)
            write_results(folder, matrix, diagnostic, samples,
                          measurement_status={"first_order_output": "pending"})
            stored = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            for key, value in before.items():
                self.assertEqual(stored[key], value)
            self.assertEqual(matrix, before)
            self.assertEqual(stored["schema_version"], SCHEMA_VERSION)
            self.assertEqual(stored["measurement_status"]["first_order_output"], "pending")
            self.assertEqual(stored["paired_diagnostic"]["measurement_version"],
                             "paired-output-diagnostics")
            self.assertEqual(json.loads((folder / "samples.json").read_text("utf-8")),
                             samples)
            sample_index = stored["samples_file"]
            self.assertEqual(sample_index["path"], "samples.json")
            self.assertEqual(sample_index["rows"], len(samples))
            self.assertEqual(sample_index["sha256"],
                             hashlib.sha256((folder / "samples.json").read_bytes()).hexdigest())
            self.assertEqual(sorted(path.name for path in folder.iterdir()),
                             ["samples.json", "summary.json"])

    def test_invalid_payload_does_not_modify_existing_files(self) -> None:
        matrix = make_matrix()
        diagnostic = {
            "model": "qwen", "measurement_version": "paired-propagation-diagnostics",
            "provenance": {}, "metrics": {"bad": math.nan},
            "measurement_status": {"paired": "measured"},
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temporary:
            folder = Path(temporary)
            target = folder / "summary.json"
            target.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_results(folder, matrix, diagnostic, [])
            self.assertEqual(target.read_text("utf-8"), "original")
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["summary.json"])
        diagnostic["metrics"] = {}
        diagnostic.pop("measurement_status")
        with self.assertRaises(ValueError):
            write_results(Path("unused"), matrix, diagnostic, [])


if __name__ == "__main__":
    unittest.main()
