"""三本账与手算对照。"""

import unittest

from kvstudy.ledger import account_pair, nominal_bits, offchip_bytes_per_token


class LedgerTest(unittest.TestCase):
    def test_online_int8_hand_count(self) -> None:
        # 1 层、1 个 KV head、32 token、32 通道、G_c=32：每 token 一组。
        ledger = account_pair(
            layers=1,
            kv_heads=1,
            seq=32,
            dim=32,
            k_bits=8,
            v_bits=8,
            k_mode="per_token",
            v_mode="per_token",
            g_c=32,
            g_t=32,
            block_tokens=64,
        )
        self.assertEqual(ledger.elements, 2048)
        self.assertEqual(ledger.k_payload.raw_bytes, 1024)
        self.assertEqual(ledger.k_params.raw_bytes, 128)
        self.assertEqual(ledger.k_payload.padding_bytes, 0)
        self.assertEqual(ledger.resident_bytes, 2304)
        self.assertEqual(ledger.effective_bits, 9.0)
        self.assertEqual(ledger.block_count, 1)
        self.assertEqual(ledger.tail_valid_tokens, 32)

    def test_int4_is_five_bits_with_params(self) -> None:
        ledger = account_pair(
            layers=1,
            kv_heads=1,
            seq=32,
            dim=32,
            k_bits=4,
            v_bits=4,
            k_mode="per_token",
            v_mode="per_token",
            g_c=32,
            g_t=32,
            block_tokens=64,
        )
        self.assertEqual(ledger.k_payload.raw_bytes, 512)
        self.assertEqual(ledger.effective_bits, 5.0)
        self.assertLess(ledger.effective_bits, 9.0)

    def test_tail_block_keeps_valid_length(self) -> None:
        ledger = account_pair(
            layers=1,
            kv_heads=1,
            seq=80,
            dim=32,
            k_bits=8,
            v_bits=8,
            k_mode="per_token",
            v_mode="per_token",
            g_c=32,
            g_t=32,
            block_tokens=64,
        )
        self.assertEqual(ledger.block_count, 2)
        self.assertEqual(ledger.tail_valid_tokens, 16)
        self.assertEqual(ledger.k_payload.raw_bytes, 80 * 32)

    def test_just_closed_cross_token_group_has_no_open_copy(self) -> None:
        ledger = account_pair(
            layers=1,
            kv_heads=1,
            seq=64,
            dim=4,
            k_bits=4,
            v_bits=4,
            k_mode="cross_token",
            v_mode="per_token",
            g_c=4,
            g_t=32,
            block_tokens=64,
        )
        self.assertEqual(ledger.open_group_tokens, 0)
        self.assertEqual(ledger.residual.raw_bytes, 0)

    def test_open_cross_token_group_is_high_precision(self) -> None:
        ledger = account_pair(
            layers=1,
            kv_heads=1,
            seq=40,
            dim=4,
            k_bits=4,
            v_bits=4,
            k_mode="cross_token",
            v_mode="per_token",
            g_c=4,
            g_t=32,
            block_tokens=64,
        )
        self.assertEqual(ledger.open_group_tokens, 8)
        # 8 个未闭合 token、4 个通道、2 字节，只计 Key。
        self.assertEqual(ledger.residual.raw_bytes, 8 * 4 * 2)
        # 闭合的 32 token 才进入 4 bit 载荷。
        self.assertEqual(ledger.k_payload.raw_bytes, 32 * 4 // 2)

    def test_full_block(self) -> None:
        ledger = account_pair(
            layers=1,
            kv_heads=1,
            seq=64,
            dim=32,
            k_bits=8,
            v_bits=8,
            k_mode="per_token",
            v_mode="per_token",
            g_c=32,
            g_t=32,
            block_tokens=64,
        )
        self.assertEqual(ledger.block_count, 1)
        self.assertEqual(ledger.tail_valid_tokens, 64)

    def test_nominal_mix(self) -> None:
        self.assertEqual(nominal_bits(8, 4), 6.0)

    def test_write_once_per_new_token(self) -> None:
        movement = offchip_bytes_per_token(
            layers=1,
            kv_heads=1,
            dim=32,
            history=32,
            k_bits=8,
            v_bits=8,
            k_mode="per_token",
            v_mode="per_token",
            g_c=32,
            g_t=32,
        )
        # 新 token：K/V 各 32 字节载荷 + 各 4 字节参数。对齐没有跳变。
        self.assertEqual(movement["write_bytes"], 72.0)


if __name__ == "__main__":
    unittest.main()
