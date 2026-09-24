"""量化、打包和边界条件的回归。"""

import unittest

import torch

from kvstudy.quantize import (
    STATUS_CONSTANT,
    STATUS_INVALID,
    STATUS_NORMAL,
    dequantize,
    fit_group_params,
    pack_int4,
    quantize_dequantize_group,
    quantize_with_params,
    round_fp16,
    saturation_mask,
    unpack_int4,
)


class RoundFp16Test(unittest.TestCase):
    def test_directed_rounding_between_neighbors(self) -> None:
        low = torch.tensor(1.0, dtype=torch.float16)
        high = torch.nextafter(low, torch.tensor(float("inf"), dtype=torch.float16))
        gap = high.to(dtype=torch.float32) - low.to(dtype=torch.float32)
        above_mid = low.to(dtype=torch.float32) + gap * 0.75
        self.assertEqual(float(round_fp16(above_mid, toward="down")), float(low))
        self.assertEqual(float(round_fp16(above_mid, toward="up")), float(high))

    def test_exact_value_stays(self) -> None:
        value = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)
        rounded = round_fp16(value, toward="down")
        self.assertTrue(torch.equal(rounded, value.to(dtype=torch.float16)))

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_rounding_matches_cpu(self) -> None:
        """覆盖正负数、零、次正规数和溢出的定向舍入。"""
        values = torch.tensor([
            -1e8, -65504.0, -1.0005, -1e-8, -0.0,
            0.0, 1e-8, 1.0005, 65504.0, 1e8,
        ], dtype=torch.float32)
        for toward in ("down", "up"):
            reference = round_fp16(values, toward=toward)
            actual = round_fp16(values.cuda(), toward=toward).cpu()
            self.assertTrue(torch.equal(actual, reference))


class QuantizeTest(unittest.TestCase):
    def test_hand_calculated_ramp(self) -> None:
        values = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        restored, scale, offset, status = quantize_dequantize_group(values, 2)
        self.assertEqual(int(status), STATUS_NORMAL)
        self.assertEqual(float(scale), 1.0)
        self.assertEqual(float(offset), 0.0)
        self.assertTrue(torch.equal(restored, values))

    def test_constant_group(self) -> None:
        values = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        restored, scale, offset, status = quantize_dequantize_group(values, 4)
        self.assertEqual(int(status), STATUS_CONSTANT)
        self.assertEqual(float(offset), 1.0)
        self.assertTrue(torch.equal(restored, values))
        del scale

    def test_non_representable_constant_is_not_merged(self) -> None:
        values = torch.full((1, 4), 0.1)
        _, _, _, status = quantize_dequantize_group(values, 8)
        self.assertEqual(int(status), STATUS_NORMAL)

    def test_scale_underflow_uses_subnormal(self) -> None:
        values = torch.tensor([[0.0, 1e-12]])
        _, scale, _, status = quantize_dequantize_group(values, 8)
        smallest = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float16),
            torch.tensor(float("inf"), dtype=torch.float16),
        )
        self.assertEqual(int(status), STATUS_NORMAL)
        self.assertEqual(float(scale), float(smallest))

    def test_parameter_overflow_is_invalid(self) -> None:
        values = torch.tensor([[0.0, 1e8]])
        restored, _, _, status = quantize_dequantize_group(values, 4)
        self.assertEqual(int(status), STATUS_INVALID)
        self.assertTrue(torch.isnan(restored).all())

    def test_non_finite_is_invalid(self) -> None:
        for bad in (float("nan"), float("inf")):
            values = torch.tensor([[0.0, bad]])
            restored, _, _, status = quantize_dequantize_group(values, 8)
            self.assertEqual(int(status), STATUS_INVALID)
            self.assertTrue(torch.isnan(restored).all())

    def test_external_scale_saturates(self) -> None:
        calibration = torch.tensor([[0.0, 1.0]])
        scale, offset, status = fit_group_params(calibration, 4)
        outside = torch.tensor([[5.0]])
        codes = quantize_with_params(outside, scale, offset, status, 4)
        self.assertEqual(int(codes), 15)
        clipped = saturation_mask(outside, scale, offset, status, 4)
        self.assertTrue(bool(clipped))

    def test_int4_pack_roundtrip(self) -> None:
        values = torch.arange(32, dtype=torch.float32).view(1, 32)
        scale, offset, status = fit_group_params(values, 4)
        codes = quantize_with_params(values, scale, offset, status, 4)
        packed, pad = pack_int4(codes)
        self.assertEqual(pad, 0)
        self.assertEqual(int(packed.numel()), 16)
        unpacked = unpack_int4(packed, int(codes.numel())).view_as(codes)
        self.assertTrue(torch.equal(unpacked, codes))
        direct = dequantize(codes, scale, offset, status)
        via_pack = dequantize(unpacked, scale, offset, status)
        self.assertTrue(torch.equal(direct, via_pack))
        # 较小索引在低四位。
        self.assertEqual(int(packed[0] & 0x0F), int(codes.view(-1)[0]))
        self.assertEqual(int((packed[0] >> 4) & 0x0F), int(codes.view(-1)[1]))


if __name__ == "__main__":
    unittest.main()
