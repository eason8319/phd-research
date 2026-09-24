"""E2 快速假量化与已核验的通用分组实现保持数值一致。"""

import unittest

import torch

from kvstudy.groups import fit_per_channel, quantize_per_channel, quantize_per_token
from phase1.e2_quant import _quantize


class E2QuantTest(unittest.TestCase):
    """检查 E2 快速路径的还原值和无效组计数。"""

    def test_per_token_matches_shared_implementation(self) -> None:
        """含普通组和非有限组的 INT4/INT8 结果逐位相同。"""
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            values = torch.linspace(-1, 1, 2 * 3 * 128, device=device).reshape(2, 3, 128)
            values[0, 1, 17] = float("nan")
            for bits in (4, 8):
                expected, stats = quantize_per_token(values, bits, 32)
                actual, invalid = _quantize(values, bits, "per_token", None)
                self.assertTrue(torch.allclose(actual, expected, atol=0, rtol=0, equal_nan=True))
                self.assertEqual(int(invalid), stats.invalid_groups)

    def test_per_channel_matches_shared_implementation(self) -> None:
        """外部 FP16 参数作用于留出 token 时与通用实现相同。"""
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            calibration = torch.linspace(-2, 2, 2 * 4 * 128, device=device).reshape(2, 4, 128)
            values = calibration[:, :3, :] * 1.1
            for bits in (4, 8):
                params = fit_per_channel(calibration, bits)
                expected, stats = quantize_per_channel(values, bits, *params)
                actual, invalid = _quantize(values, bits, "per_channel", params)
                self.assertTrue(torch.allclose(actual, expected, atol=0, rtol=0, equal_nan=True))
                self.assertEqual(int(invalid), stats.invalid_groups)


if __name__ == "__main__":
    unittest.main()
