"""保障阶段审核拒绝重复、冲突与失效来源，同时保留实际不同的对照。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phase1.audit import check_hashes, check_quality


def _quality(config: str = "kv8", input_sha: str = "input") -> dict:
    """构造已记录输入身份的一题结果，未知停止原因保持原样。"""
    return {"model": "qwen", "config": config, "task": "v-gsm8k", "n": 1,
            "em": 100.0, "correct": 1, "rows": [
                {"id": "question", "correct": True, "new_tokens": 3,
                 "input_sha256": input_sha, "input_tokens": 7, "invalid_groups": 0,
                 "stop": None},
            ]}


class PackageAuditTest(unittest.TestCase):
    """清理依据来自实际输入和条件，不能按题号或文件名猜测。"""

    def test_duplicate_measurement_rejected(self) -> None:
        """重复文件即使更名、改变历史协议标签，也不能成为第二份正式结果。"""
        original, copy = _quality(), _quality()
        copy["protocol"] = "legacy"
        with self.assertRaisesRegex(RuntimeError, "同条件重复"):
            check_quality({"first.json": original, "copy.json": copy})

    def test_changed_input_or_condition_retained(self) -> None:
        """同题号的实际不同输入与不同格式均保留。"""
        result = check_quality({"first": _quality(), "condition": _quality("kv4"),
                                "input": _quality(input_sha="other")})
        self.assertEqual(result["files"], 3)

    def test_wrong_aggregate_rejected(self) -> None:
        """可信的逐题记录不能掩盖错误汇总。"""
        result = _quality()
        result["em"] = 50.0
        with self.assertRaisesRegex(RuntimeError, "复算数值"):
            check_quality({"result": result})

    def test_unknown_stop_and_original_bytes_preserved(self) -> None:
        """只读审核不补写未知历史停止原因。"""
        result = _quality()
        before = json.dumps(result)
        checks = check_quality({"result": result})
        self.assertEqual(checks["unknown_historical_stops"], 1)
        self.assertEqual(json.dumps(result), before)

    def test_missing_input_identity_rejected(self) -> None:
        """只有样本编号不能建立跨实验复用身份。"""
        result = _quality(input_sha="")
        with self.assertRaisesRegex(RuntimeError, "输入身份"):
            check_quality({"result": result})

    def test_hash_conflict_and_outside_path_rejected(self) -> None:
        """来源不能越出项目，实体变化必须由新审核明确处理。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "record.json"
            path.write_text("{}", "utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(check_hashes(root, {"record.json": digest}), 1)
            with self.assertRaisesRegex(RuntimeError, "校验失败"):
                check_hashes(root, {"record.json": "wrong"})
            with self.assertRaisesRegex(RuntimeError, "超出项目"):
                check_hashes(root, {"../other.json": digest})


if __name__ == "__main__":
    unittest.main()
