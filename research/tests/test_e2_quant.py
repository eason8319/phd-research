"""E2 快速假量化与已核验的通用分组实现保持数值一致。"""

import unittest

import torch

from kvstudy.groups import fit_per_channel, quantize_per_channel, quantize_per_token
from phase1.e2_quant import RESIDUAL_TOKENS, _quantize, _write_residual_token


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

    def test_residual_ring_matches_full_history_copy(self) -> None:
        """短于及长于残留窗口的历史，在每次滑出后读到相同的 FP32 KV。"""
        for length in (4, RESIDUAL_TOKENS + 2):
            capacity = length + 5
            initial = torch.arange(2 * length * 3, dtype=torch.float32).reshape(
                1, 2, length, 3
            )
            low_key = torch.empty((1, 2, capacity, 3), dtype=torch.float32)
            low_value = torch.empty_like(low_key)
            low_key[:, :, :length, :] = initial
            low_value[:, :, :length, :] = initial * 2
            full_read_key = low_key.clone()
            full_read_value = low_value.clone()
            start = max(0, length - RESIDUAL_TOKENS)
            full_read_key[:, :, start:length, :] += 0.25
            full_read_value[:, :, start:length, :] += 0.5
            ring_read_key = full_read_key.clone()
            ring_read_value = full_read_value.clone()
            ring_key = torch.empty((1, 2, RESIDUAL_TOKENS, 3), dtype=torch.float32)
            ring_value = torch.empty_like(ring_key)
            indices = torch.arange(start, length) % RESIDUAL_TOKENS
            ring_key[:, :, indices, :] = initial[:, :, start:, :]
            ring_value[:, :, indices, :] = initial[:, :, start:, :] * 2
            for position in range(length, capacity):
                key = torch.full((2, 1, 3), float(position))
                value = key * 2
                high_key = key + 0.25
                high_value = value + 0.5
                low_key[:, :, position:position + 1, :] = key.unsqueeze(0)
                low_value[:, :, position:position + 1, :] = value.unsqueeze(0)
                full_read_key[:, :, position:position + 1, :] = high_key.unsqueeze(0)
                full_read_value[:, :, position:position + 1, :] = high_value.unsqueeze(0)
                if position >= RESIDUAL_TOKENS:
                    leaving = position - RESIDUAL_TOKENS
                    full_read_key[:, :, leaving:leaving + 1, :] = (
                        low_key[:, :, leaving:leaving + 1, :]
                    )
                    full_read_value[:, :, leaving:leaving + 1, :] = (
                        low_value[:, :, leaving:leaving + 1, :]
                    )
                _write_residual_token(
                    ring_key, ring_value, ring_read_key, ring_read_value,
                    key, value, high_key, high_value, position,
                )
                self.assertTrue(torch.equal(
                    full_read_key[:, :, :position + 1, :],
                    ring_read_key[:, :, :position + 1, :],
                ))
                self.assertTrue(torch.equal(
                    full_read_value[:, :, :position + 1, :],
                    ring_read_value[:, :, :position + 1, :],
                ))


if __name__ == "__main__":
    unittest.main()
