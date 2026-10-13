"""保障冻结七位数字评分的歧义、重复与标点边界。"""

import unittest

from phase1.score import retrieval_number_exact, retrieval_numbers, retrieval_exact


class RetrievalScoringTest(unittest.TestCase):
    """检索主评分只看唯一数字值，旧整段评分保持原义。"""

    def test_punctuation_and_repeated_target(self) -> None:
        for prediction in (": 2561432.", "2561432, 2561432"):
            self.assertTrue(retrieval_number_exact(prediction, "2561432"))
        self.assertFalse(retrieval_exact(": 2561432.", "2561432"))

    def test_missing_wrong_and_ambiguous(self) -> None:
        for prediction in ("no answer", "1234567", "2561432 and 1234567", "12561432"):
            self.assertFalse(retrieval_number_exact(prediction, "2561432"))
        self.assertEqual(retrieval_numbers("12561432, 02561432"), [])

    def test_invalid_target(self) -> None:
        with self.assertRaises(ValueError):
            retrieval_number_exact("123", "123")


if __name__ == "__main__":
    unittest.main()
