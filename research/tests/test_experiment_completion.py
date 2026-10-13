"""检查保留的候选生成辅助实现拒绝不一致的配对输入。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase1.complete_experiments import ExperimentCompletion

class ExperimentCompletionTest(unittest.TestCase):
    """防止题号相同但输入不同的旧结果被复用。"""

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



if __name__ == "__main__":
    unittest.main()
