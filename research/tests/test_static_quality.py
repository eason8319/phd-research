"""独立质量收尾的请求统计、零差异与候选停止规则回归。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.static_layer_allocation.code.quality import (
    judge, next_round, reference_result, result_path, run,
)
from experiments.static_layer_allocation.code.quality_data import expansion_groups


class StaticQualityTest(unittest.TestCase):
    """核对会改变样本配对和最终候选资格的行为。"""

    def test_reference_prefix_is_composed_without_copy(self) -> None:
        """追加实体只含新行，早期前缀继续读原参考，重复身份被拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            rows = [{"id": key, "input_sha256": key, "input_tokens": 10} for key in "abcd"]
            base.write_text(json.dumps({"rows": rows[:2], "n": 2, "model": "qwen",
                                        "task": "v-gsm8k"}), "utf-8")
            extra = result_path(root, "qwen", "gsm8k", "bf16")
            extra.parent.mkdir(parents=True)
            value = {"rows": rows[2:], "n": 2, "model": "qwen", "task": "gsm8k",
                     "condition": "bf16", "reference_source": {
                "path": "base.json", "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "n": 2,
            }}
            extra.write_text(json.dumps(value), "utf-8")
            with patch("phase1.results.reference_path", return_value=base), \
                    patch("experiments.static_layer_allocation.code.quality.load_plan",
                          return_value={"v_plan": {"references": {}}}):
                self.assertEqual(reference_result(root, "qwen", "gsm8k", "bf16", 2)["rows"],
                                 rows[:2])
                self.assertEqual(len(reference_result(root, "qwen", "gsm8k", "bf16", 4)["rows"]),
                                 4)
                value["rows"][0]["id"] = "a"
                extra.write_text(json.dumps(value), "utf-8")
                with self.assertRaisesRegex(RuntimeError, "重复身份"):
                    reference_result(root, "qwen", "gsm8k", "bf16", 4)
            self.assertEqual(len(json.loads(base.read_text())["rows"]), 2)

    def test_expansion_does_not_generate_k8v4(self) -> None:
        """首轮之后只有必要 BF16、KV8 和候选追加，K8V4 不被强制同步。"""
        plan = {"v_plan": {"tasks": {"gsm8k": {}}, "candidates": {"qwen": {"mixed": {}}}}}
        snapshots = [{"next_expansion": {"mixed/gsm8k": 128}},
                     {"next_expansion": {}, "stops": {}}]
        with patch("experiments.static_layer_allocation.code.quality.load_plan",
                   return_value=plan), \
                patch("experiments.static_layer_allocation.code.quality.totals",
                      return_value=[64, 128]), \
                patch("experiments.static_layer_allocation.code.quality.summarize",
                      side_effect=snapshots), \
                patch("experiments.static_layer_allocation.code.quality.measure") as measurement:
            run(Path("project"), "qwen")
        extra = [call.args[3] for call in measurement.call_args_list if call.args[4] == 128]
        self.assertEqual(extra, ["bf16", "kv8", "mixed"])

    def test_four_expansion_keeps_original_order(self) -> None:
        items = [{"id": str(index)} for index in range(192)]
        groups = expansion_groups(items)
        self.assertEqual(len(groups), 48)
        self.assertEqual([item for group in groups for item in group], items)
        with self.assertRaises(RuntimeError):
            expansion_groups(items[:191] + [items[0]])

    def test_long_subset_uses_baseline_only(self) -> None:
        old = [{"id": "a", "new_tokens": 300, "f1": 80.0,
                "input_sha256": "a", "input_tokens": 31000},
               {"id": "b", "new_tokens": 100, "f1": 100.0,
                "input_sha256": "b", "input_tokens": 31000}]
        new = [dict(old[0], new_tokens=1, f1=0.0, missing_answers=4, stop="eos"),
               dict(old[1], new_tokens=1024, f1=100.0, stop="limit")]
        result = judge({"n": 2, "rows": old}, {"n": 2, "rows": old},
                       {"n": 2, "rows": new}, "hotpot_four",
                       {"baseline_min_tokens": 256, "baseline_minimum": 40},
                       17, 10000, True)
        self.assertEqual(result["paired_ids"], ["a"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["state"], "淘汰")
        self.assertEqual(result["missing_answers"], 4)

    def test_all_zero_continuous_difference_is_insufficient(self) -> None:
        rows = [{"id": str(i), "new_tokens": 300, "f1": 80.0, "stop": "eos",
                 "input_sha256": str(i), "input_tokens": 31000} for i in range(64)]
        payload = {"n": 64, "rows": rows}
        result = judge(payload, payload, payload, "hotpot",
                       {"baseline_minimum": 40}, 17, 10000, True)
        self.assertIsNone(result["versus_bf16"]["upper"])
        self.assertEqual(result["state"], "证据不足")

    def test_elimination_stops_expansion_across_tasks(self) -> None:
        cells = {"mixed": {"math": {"state": "淘汰", "requested_n": 64},
                           "retrieval": {"state": "证据不足", "requested_n": 64}}}
        requests, stops = next_round(["mixed"], cells,
                                    {"math": [64, 128, 256], "retrieval": [64, 128, 256]})
        self.assertFalse(requests)
        self.assertIn("淘汰", stops["mixed"])

    def test_priority_switches_to_backup_at_primary_cap(self) -> None:
        cells = {"kv4": {"math": {"state": "证据不足", "requested_n": 256}},
                 "mixed": {"math": {"state": "证据不足", "requested_n": 64}}}
        requests, stops = next_round(["kv4", "mixed"], cells, {"math": [64, 128, 256]},
                                    ["kv4", "mixed"])
        self.assertEqual(requests, {"mixed/math": 128})
        self.assertIn("上限", stops["kv4"])

    def test_low_cost_retention_stops_backup_without_elimination(self) -> None:
        cells = {"kv4": {"math": {"state": "保留", "requested_n": 256}},
                 "mixed": {"math": {"state": "证据不足", "requested_n": 64}}}
        requests, stops = next_round(["kv4", "mixed"], cells, {"math": [64, 128, 256]},
                                    ["kv4", "mixed"])
        self.assertFalse(requests)
        self.assertIn("停止备选", stops["mixed"])
        self.assertNotIn("淘汰", stops["mixed"])


if __name__ == "__main__":
    unittest.main()
