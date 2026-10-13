"""保护格式整理中的原分数、未知值、输入冲突和扩样前缀读取。"""

import json
import tempfile
import unittest
from pathlib import Path

from phase1.result_format import normalize_quality_result
from experiments.static_layer_allocation.code.select_final import _combined, expanded_result_path


class ResultFormatTest(unittest.TestCase):
    """旧测量与补测使用同一格式，冲突不通过补字段掩盖。"""

    def test_historical_baseline_preserves_unknown_stop_and_scores(self) -> None:
        """重建输入只补身份，不推断历史停止原因或改写提交和分数。"""
        old = {"protocol": "old", "commit": "actual", "config": "bf16-h0",
               "task": "v-gsm8k", "n": 1, "em": 100.0,
               "rows": [{"id": "a", "correct": True, "new_tokens": 3}]}
        new = normalize_quality_result(old, {"a": {"input_sha256": "abc",
                                                  "input_tokens": 7}})
        for key in ("protocol", "commit", "em"):
            self.assertEqual(old[key], new[key])
        self.assertIsNone(new["rows"][0]["stop"])
        self.assertEqual(new["rows"][0]["input_identity_source"],
                         "reconstructed_from_frozen_inputs")
        self.assertEqual(new, normalize_quality_result(new))
        self.assertNotIn("input_sha256", old["rows"][0])

    def test_conflicting_recorded_input_is_rejected(self) -> None:
        """有题号也不能把不同全文的记录归并。"""
        old = {"config": "kv8", "task": "v-hotpot", "rows": [
            {"id": "a", "new_tokens": 1, "input_sha256": "wrong", "invalid_groups": 0}
        ]}
        with self.assertRaisesRegex(ValueError, "冻结输入不一致"):
            normalize_quality_result(old, {"a": {"input_sha256": "right",
                                                 "input_tokens": 7}})

    def test_structured_configuration_preserves_measurement(self) -> None:
        """独立生成的完整配置与 BF16 空配置均可补通用字段。"""
        for config, condition in ((None, "bf16"), ({"name": "kv4"}, "candidate")):
            old = {"config": config, "condition": condition, "task": "gsm8k",
                   "rows": [{"id": "a", "correct": True, "new_tokens": 3,
                             "input_sha256": "abc", "input_tokens": 7,
                             "invalid_groups": 0, "stop": "eos"}]}
            new = normalize_quality_result(old)
            self.assertEqual(new["config"], config)
            self.assertEqual(new["result_format_version"], 1)
            self.assertEqual(new["rows"][0]["input_identity_source"], "recorded")
            for key, value in old["rows"][0].items():
                self.assertEqual(new["rows"][0][key], value)
            self.assertEqual(new, normalize_quality_result(new))

    def test_larger_canonical_file_serves_earlier_round(self) -> None:
        """只保留一份扩样数据时，早期判定读取已核验的同条件前缀。"""
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "hotpot_uniform_quantization_n256.json"
            path.write_text(json.dumps({"extra_n": 192}), encoding="utf-8")
            self.assertEqual(expanded_result_path(folder, "hotpot", "e2", 128), path)
            with self.assertRaises(FileNotFoundError):
                expanded_result_path(folder, "hotpot", "kv8", 128)

    def test_final_selection_rejects_stale_expansion_config(self) -> None:
        """题数、题号和条件标签相同，配置摘要错误也不得用于最终判定。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = root / "research/phase1/experiments/generation_error_sources/results/qwen"
            expanded = root / (
                "research/phase1/experiments/static_layer_allocation/results/expanded_quality/qwen"
            )
            generation.mkdir(parents=True)
            expanded.mkdir(parents=True)
            (generation / "v_hotpot_bf16.json").write_text(json.dumps({
                "n": 64, "rows": [{"id": str(i)} for i in range(64)]
            }), encoding="utf-8")
            (expanded / "hotpot_bf16_n128.json").write_text(json.dumps({
                "model": "qwen", "task": "hotpot", "condition": "bf16",
                "extra_n": 64, "total_n_with_v": 128, "config_digest": "wrong",
                "rows": [{"id": str(i)} for i in range(64, 128)],
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "扩样配置"):
                _combined(root, "qwen", "hotpot", "bf16", 128)


if __name__ == "__main__":
    unittest.main()
