"""事务边界、缓冲守恒和流水依赖的长期回归。"""

import unittest

from experiments.storage_service_cost.code.service import (
    layer_service_time,
    pipeline_time,
    predict_time,
)
from experiments.storage_service_cost.code.transfer import boundary_flush, stream_step


class StorageServiceTest(unittest.TestCase):
    """用显式事务地址集合检查计数，覆盖跨块和部分刷新。"""

    def test_immediate_address_oracle(self) -> None:
        for block in (32, 64, 128):
            for stride in (16, 64, 128):
                for n in (0, 1, 2, 7, 8, block - 1, block, block + 1, 2 * block - 1):
                    read_addresses = {offset // 128 for offset in range(n * stride)}
                    append_addresses = {offset // 128 for offset in
                                        range(n * stride, (n + 1) * stride)}
                    preserved = read_addresses & append_addresses
                    row = stream_step(n, stride, block, 128, "immediate_rmw")
                    self.assertEqual(row["read"], 128 * len(read_addresses))
                    self.assertEqual(row["append_write"], 128 * len(append_addresses))
                    self.assertEqual(row["rmw_read"], 128 * len(preserved))

    def test_combined_prefix_conservation(self) -> None:
        for stride in (16, 64, 128):
            flushed = 0
            for n in range(130):
                row = stream_step(n, stride, 64, 128, "combined")
                self.assertEqual(row["read"] + row["buffer_read"], n * stride)
                flushed += row["append_write"]
                self.assertEqual(flushed + row["pending_after"], (n + 1) * stride)

    def test_partial_flush_and_resume(self) -> None:
        for stride in (16, 64, 128):
            for length in (1, 7, 8, 63, 64, 65):
                row = boundary_flush(length, stride, 128)
                complete = length * stride // 128 * 128
                self.assertEqual(complete + row["flush_write"] - row["flush_waste"],
                                 length * stride)
                self.assertEqual(row["resume_seed_read"], row["flush_write"])

    def test_window_256_writes(self) -> None:
        for stride in (16, 64, 128):
            rows = [stream_step(n, stride, 64, 128, "combined")
                    for n in range(32768 - 256, 32768)]
            self.assertEqual(sum(r["append_write"] for r in rows), stride * 256)
            self.assertEqual(rows[-1]["pending_after"], 0)

    def test_pipeline_fill_drain_and_dependency(self) -> None:
        self.assertEqual(pipeline_time([[2, 3], [2, 3]]), 8)
        self.assertEqual(pipeline_time([[2, 3, 4], [2, 3, 4]]), 13)
        self.assertEqual(layer_service_time([[2, 3]], [[7, 11]], 1, 13, 17), 54)

    def test_bounded_pipeline_backpressure(self) -> None:
        self.assertEqual(pipeline_time([[1, 1, 10]] * 3, capacity=1), 36)
        self.assertEqual(pipeline_time([[1, 1, 10]] * 3, capacity=2), 32)

    def test_invalid_layout_and_times(self) -> None:
        with self.assertRaises(ValueError):
            stream_step(1, 24, 64, 128, "combined")
        with self.assertRaises(ValueError):
            pipeline_time([[1, -1]])

    def test_full_model_requires_rates_and_serial_layers(self) -> None:
        layout = {"head_dim": 128, "kv_heads": 1, "query_heads": 1,
                  "actual_configurations": {"one": {"rope": "pre", "layer_bits": [[4, 4]]},
                                            "two": {"rope": "pre",
                                                    "layer_bits": [[4, 4], [4, 4]]}}}
        with self.assertRaises(ValueError):
            predict_time(layout, "one", 65, 1, 64, "combined", {})
        rates = {key: 1000 for key in ("memory", "decode4", "decode8", "quant4", "quant8",
                                      "rope", "dot", "softmax_arithmetic", "exp", "division",
                                      "sram_write", "sram_read", "parameter_sram",
                                      "score_sram", "coefficient_sram")}
        rates.update({"launch_s": 0.1, "coefficients_s": 0.2})
        one = predict_time(layout, "one", 65, 1, 64, "combined", rates)
        two = predict_time(layout, "two", 65, 1, 64, "combined", rates)
        self.assertAlmostEqual(two["total_s"] - 0.2, 2 * (one["total_s"] - 0.2))


if __name__ == "__main__":
    unittest.main()
