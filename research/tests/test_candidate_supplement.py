"""核验补测停止边界与预测持久化，避免恢复时丢失答案或改变评分。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from experiments.generation_error_sources.code import quantized_generation as generation
from experiments.generation_error_sources.code.early_stop import severe_failure


class CandidateSupplementTest(unittest.TestCase):
    """仅检查新增的记录与停止控制；量化数值复用已有回归。"""

    def test_invalid_groups_stop_at_two(self) -> None:
        self.assertIsNone(severe_failure([{"invalid_groups": 1}], [], "hotpot"))
        self.assertIsNotNone(severe_failure([{"invalid_groups": 1}] * 2, [], "hotpot"))

    def test_severe_exhaustion_requires_paired_evidence(self) -> None:
        rows = [{"id": str(i), "input_sha256": str(i), "input_tokens": 100,
                 "invalid_groups": 0, "stop": "limit", "correct": False} for i in range(9)]
        reference = {"rows": [{**row, "correct": True} for row in rows]}
        self.assertIsNotNone(severe_failure(rows, [reference, reference], "gsm8k"))
        self.assertIsNone(severe_failure(rows[:7], [reference, reference], "gsm8k"))
        self.assertIsNone(severe_failure(rows, [{"rows": rows}] * 2, "gsm8k"))
        rows[0]["stop"] = "eos"
        self.assertIsNone(severe_failure(rows, [reference, reference], "gsm8k"))

    def test_prediction_saved_without_changing_score_and_resume_checks_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "research/.runtime").mkdir(parents=True)
            (root / "research/phase1/splits").mkdir(parents=True)
            (root / "research/phase1/splits/templates.json").write_text(
                json.dumps({"gsm8k": "{question}"}), encoding="utf-8")
            item = {"id": "sample", "question_sha256": "question", "answer": "#### 42"}
            tokenizer = MagicMock()
            tokenizer.decode.return_value = "#### 42"
            with patch.object(generation, "project_root", return_value=root), patch.object(
                generation, "_inputs", return_value=[item]
            ), patch.object(generation.AutoTokenizer, "from_pretrained", return_value=tokenizer), \
                    patch.object(generation, "_calibration", return_value={}), \
                    patch.object(generation, "load_model"), \
                    patch.object(generation, "_eos_ids", return_value={9}), \
                    patch.object(generation.torch.cuda, "get_device_name", return_value="test"), \
                    patch.object(generation.torch.cuda, "get_device_properties",
                                 return_value=type("Device", (), {"total_memory": 24000})()), \
                    patch.object(generation.torch.cuda, "reset_peak_memory_stats"), \
                    patch.object(generation.torch.cuda, "max_memory_allocated", return_value=0), \
                    patch.object(generation.torch.cuda, "max_memory_reserved", return_value=0), \
                    patch.object(generation, "_ids_and_source", return_value=([1, 2], "input")), \
                    patch.object(generation, "_generate", return_value=([42, 9], "eos", 0)):
                generation.run("qwen", "gsm8k", "v", "k8v4-per_channel-pre", "base", 8,
                               save_predictions=True, execution_source={"actual": "digest"})
                path = root / ("research/phase1/experiments/generation_error_sources/results/"
                               "qwen/v_gsm8k_k8v4-per_channel-pre-residual8.json")
                result = json.loads(path.read_text("utf-8"))
                self.assertEqual(result["em"], 100)
                saved = json.loads((root / result["prediction_source"]["path"]).read_text("utf-8"))
                self.assertEqual(saved["predictions"][0]["text"], "#### 42")
                self.assertEqual(saved["predictions"][0]["token_ids"], [42, 9])
                with self.assertRaisesRegex(RuntimeError, "执行来源变化"):
                    generation.run("qwen", "gsm8k", "v", "k8v4-per_channel-pre", "base", 8,
                                   save_predictions=True, execution_source={"actual": "changed"})


if __name__ == "__main__":
    unittest.main()
