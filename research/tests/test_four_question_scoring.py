"""保障四问联答缺失标签、Markdown 和请求均值的评分边界。"""

import unittest

from phase1.score import score_request


class FourQuestionScoringTest(unittest.TestCase):
    """每请求包含四题，允许换行答案但不借用其他题的标签。"""

    def test_request_mean_and_numbered_order(self) -> None:
        """输出顺序变化不改变对应关系，四题平均而非只看一个答案。"""
        output = "Final answer 4: D\nFinal answer 1: A\nFinal answer 3: C\nFinal answer 2: wrong"
        row = score_request(output, ["A", "B", "C", "D"])
        self.assertEqual(row["extracted_answers"], ["A", "wrong", "C", "D"])
        self.assertEqual(row["f1"], 75)
        self.assertFalse(row["em"])
        self.assertEqual(row["answer_em_percent"], 75)

    def test_empty_label_does_not_borrow_next_line(self) -> None:
        """空标签与不存在标签计零，不能拿下一题答案代替。"""
        output = "Final answer 1:\nFinal answer 2: B\nFinal answer 4: D"
        row = score_request(output, ["A", "B", "C", "D"])
        self.assertEqual(row["extracted_answers"], ["", "B", "", "D"])
        self.assertEqual(row["missing_answers"], 2)
        self.assertEqual(row["f1"], 50)

    def test_markdown_and_last_answer(self) -> None:
        """允许常见强调标记，同一编号取末次标签，保持冻结规则。"""
        output = "Final answer 1: wrong\n**Final answer 1:** A\nFinal answer 2: **B**"
        row = score_request(output + "\nFinal answer 3: C\nFinal answer 4: D", ["A", "B", "C", "D"])
        self.assertTrue(row["em"])
        self.assertEqual(row["f1"], 100)

    def test_heading_and_following_line(self) -> None:
        """复现小样本抽取规则：标题后的下一非空行也是答案。"""
        output = (
            "### Final answer 1:\nA\n\n### Final answer 2:\nB\n"
            "**Final answer 3:**\nC\nFinal answer 4: D"
        )
        row = score_request(output, ["A", "B", "C", "D"])
        self.assertEqual(row["extracted_answers"], ["A", "B", "C", "D"])
        self.assertEqual(row["missing_answers"], 0)
        self.assertEqual(row["f1"], 100)

    def test_last_empty_label_scores_zero(self) -> None:
        """末次标签为空时计零，不能回退到前面的正确答案。"""
        row = score_request("Final answer 1: A\nFinal answer 1:", ["A", "B", "C", "D"])
        self.assertEqual(row["extracted_answers"], ["", "", "", ""])
        self.assertEqual(row["f1"], 0)


if __name__ == "__main__":
    unittest.main()
