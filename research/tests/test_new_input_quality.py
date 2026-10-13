"""新检索与四问联答的配对、复用边界和类通过／保留区分。"""

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from experiments.generation_error_sources.code.new_input_quality import (
    assert_allowed_reference,
    check_reuse,
    judge_condition,
    length_subset,
    offline_residual_category,
    planned_new_generations,
)
from phase1.inputs import _ids_and_source, _score_new_input


def _row(item_id: str, score: float, *, tokens: int = 400, stop: str = "eos",
         invalid: int = 0, missing: int = 0, binary: bool | None = None) -> dict:
    row = {
        "id": item_id, "input_sha256": f"hash-{item_id}", "input_tokens": 100,
        "new_tokens": tokens, "stop": stop, "invalid_groups": invalid,
        "missing_answers": missing, "f1": score,
    }
    if binary is not None:
        row["correct"] = binary
    return row


def _payload(rows: list[dict], *, usable: bool = True) -> dict:
    return {"n": len(rows), "rows": rows, "baseline_usable": usable, "em": 100}


class NewInputQualityTest(unittest.TestCase):
    """配对成员由 BF16 决定，类通过不写成保留。"""

    def test_long_input_paths_preserve_frozen_text_and_tokens(self) -> None:
        """四问相对字符串与检索绝对路径读取相同正文，校验和长度检查仍生效。"""
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = "Frozen four-question prompt.\n"
            path = root / "prompt.txt"
            path.write_text(source, encoding="utf-8")
            digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            tokenizer = Mock(return_value={"input_ids": [1, 2, 3]})
            for task, stored_path in (
                ("hotpot_four", "prompt.txt"), ("retrieval_ruler", path),
            ):
                with self.subTest(task=task):
                    item = {"id": "request", "path": stored_path, "sha256": digest,
                            "tokens": 3}
                    ids, actual_digest = _ids_and_source(root, tokenizer, "qwen", task, item)
                    self.assertEqual((ids, actual_digest), ([1, 2, 3], digest))
                    tokenizer.assert_called_with(source, add_special_tokens=False)
                    with self.assertRaisesRegex(RuntimeError, "校验失败"):
                        _ids_and_source(root, tokenizer, "qwen", task,
                                        {**item, "sha256": "changed"})
                    with self.assertRaisesRegex(RuntimeError, "长度不符"):
                        _ids_and_source(root, tokenizer, "qwen", task, {**item, "tokens": 4})

    def test_plan_counts_three_candidates_and_kv8(self) -> None:
        """两模型、64 道检索、16 个四问请求、五个条件共 800 次生成。"""
        self.assertEqual(planned_new_generations(), 800)

    def test_offline_pre_h0_is_paired_with_h8_only(self) -> None:
        """H=0 类通过即无残留可行；两者都失败时不外推未测的 H=16。"""
        self.assertEqual(offline_residual_category(True, True), "无残留可行")
        self.assertEqual(offline_residual_category(False, True), "依赖缓冲")
        self.assertEqual(offline_residual_category(False, False), "两个已测精度都未类通过")

    def test_old_references_are_rejected(self) -> None:
        """旧检索和单问长解码文件名不能作为本比较的参照。"""
        with self.assertRaises(RuntimeError):
            assert_allowed_reference("results/qwen/v_retrieval_bf16.json")
        with self.assertRaises(RuntimeError):
            assert_allowed_reference("v_hotpot_long_bf16.json")
        assert_allowed_reference("v_retrieval_ruler_bf16.json")

    def test_length_subset_ignores_candidate_length(self) -> None:
        """只有参照达到 256 token 的请求进入子集。"""
        rows = [
            _row("a", 100, tokens=300),
            _row("b", 100, tokens=20),
        ]
        self.assertEqual(length_subset(rows), ["a"])

    def test_short_limited_or_missing_candidate_stays_paired(self) -> None:
        """候选提前结束、缺答案或触及上限时仍留在 BF16 子集里。"""
        reference = _payload([
            _row("a", 100, tokens=300), _row("b", 80, tokens=500),
        ])
        kv8 = _payload([_row("a", 100, tokens=300), _row("b", 80, tokens=500)])
        candidate = _payload([
            _row("a", 0, tokens=4, stop="limit", missing=4),
            _row("b", 80, tokens=12, stop="eos", missing=1),
        ])
        # 直接使用长度规则构造的两请求参照；正式入口另要求子集恰好为 16。
        reference["n"] = 16
        reference["rows"] = reference["rows"] + [
            _row(f"p{i}", 50, tokens=256) for i in range(14)
        ]
        kv8["rows"] = kv8["rows"] + [_row(f"p{i}", 50, tokens=256) for i in range(14)]
        candidate["rows"] = candidate["rows"] + [
            _row(f"p{i}", 50, tokens=3, stop="limit", missing=2) for i in range(14)
        ]
        for payload in (reference, kv8, candidate):
            payload["n"] = 16
        cell = judge_condition(reference, kv8, candidate, "hotpot_four", seed=1, draws=20)
        self.assertEqual(cell["n"], 16)
        self.assertEqual(cell["below_256_tokens"], 16)
        self.assertGreater(cell["missing_answers"], 0)
        self.assertIn("limit", cell["stops"])
        self.assertFalse(cell["retention_awarded"])

    def test_class_pass_is_not_retention(self) -> None:
        """点估计过线只记类通过；区间上界过线也不在本轮授予保留。"""
        rows = [_row(f"q{i}", 100 if i else 99.9) for i in range(16)]
        close = [_row(f"q{i}", 100 if i else 99.8) for i in range(16)]
        reference = _payload(rows)
        cell = judge_condition(reference, _payload(rows), _payload(close), "hotpot_four", 1, 30)
        self.assertTrue(cell["class_unit_passes"])
        self.assertTrue(cell["interval_meets_retention_bounds"])
        self.assertEqual(cell["state"], "开发晋级")
        self.assertFalse(cell["retention_awarded"])

    def test_wide_binary_interval_can_class_pass_without_retention_bounds(self) -> None:
        """检索全对时下降点估计为零，但小样本区间上界仍超过 2。"""
        rows = [_row(f"q{i}", 0, binary=True) for i in range(64)]
        reference = {"n": 64, "rows": rows, "em": 100}
        cell = judge_condition(reference, reference, reference, "retrieval_ruler", 1, 10)
        self.assertTrue(cell["class_unit_passes"])
        self.assertFalse(cell["interval_meets_retention_bounds"])
        self.assertNotEqual(cell["state"], "保留")
        self.assertEqual(cell["n"], 64)

    def test_reuse_rejects_changed_numerics_and_accepts_matching_bytes(self) -> None:
        """结果字节、审核校验和数值文件都一致才复用。"""
        result = {
            "generation": "bf16_weights_sdpa_prefill_fp32_attention_decode_greedy",
            "config": "bf16-h0", "protocol": "phase1-ruler-number-retrieval",
            "rows": [{"id": "q", "input_sha256": "abc", "input_tokens": 10, "new_tokens": 2}],
        }
        runner = "research/phase1/src/phase1/runner.py"
        files = (runner,)
        common = dict(
            result=result, review_sha256="digest",
            publication_hashes_by_path={runner: "same"},
            current_hashes={runner: "same"}, numerical_files=files,
            manifest_rows=[{"id": "q", "sha256": "abc", "tokens": 10}],
            task="retrieval_ruler",
        )
        check_reuse(actual_sha256="digest", **common)
        with self.assertRaises(RuntimeError):
            check_reuse(actual_sha256="other", **common)
        changed = dict(common)
        changed["current_hashes"] = {"research/phase1/src/phase1/runner.py": "edited"}
        with self.assertRaises(RuntimeError):
            check_reuse(actual_sha256="digest", **changed)

    def test_invalid_group_zeroes_the_sample_score(self) -> None:
        """无效组把检索题和四问请求记为 0 分，评分字段仍保留。"""
        retrieval = _score_new_input(
            "retrieval_ruler", "1063198", {"answer": "1063198"}, 1,
        )
        request = _score_new_input(
            "hotpot_four",
            "Final answer 1: A\nFinal answer 2: B\nFinal answer 3: C\nFinal answer 4: D",
            {"answers": ["A", "B", "C", "D"]}, 2,
        )
        self.assertFalse(retrieval["correct"])
        self.assertEqual(request["f1"], 0)
        self.assertEqual(request["extracted_answers"], ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
