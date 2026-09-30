"""E1 选择规则。"""

import unittest

from experiments.representation_interactions.code.select import E1Cell, select_e1


def cell(
    name: str,
    tier: str,
    error: float,
    bits: float = 6.0,
    full_bytes: int = 100,
    invalid: bool = False,
) -> E1Cell:
    return E1Cell(name, tier, error, bits, full_bytes, invalid)


class SelectE1Test(unittest.TestCase):
    def test_keeps_best_of_each_tier(self) -> None:
        cells = [
            cell("k8v4-a", "6", 0.2, bits=7.0),
            cell("k4v8-b", "6", 0.1, bits=7.0),
            cell("kv4-a", "4", 0.4, bits=5.0),
            cell("kv4-b", "4", 0.3, bits=5.0),
        ]
        chosen = select_e1(cells, kv8_bits=9.0, kv4_error=0.5)
        self.assertEqual(chosen.kept, ("k4v8-b", "kv4-b"))

    def test_four_bit_must_beat_uniform_kv4(self) -> None:
        cells = [
            cell("k8v4-a", "6", 0.2, bits=7.0, full_bytes=80),
            cell("k4v8-b", "6", 0.3, bits=7.0, full_bytes=90),
            cell("kv4-a", "4", 0.5, bits=5.0),
        ]
        chosen = select_e1(cells, kv8_bits=9.0, kv4_error=0.5)
        self.assertEqual(chosen.kept, ("k8v4-a", "k4v8-b"))

    def test_invalid_ranks_after_clean(self) -> None:
        cells = [
            cell("clean", "6", 0.4),
            cell("bad", "6", 0.1, invalid=True),
            cell("four", "4", 0.2, bits=5.0),
        ]
        chosen = select_e1(cells, kv8_bits=9.0, kv4_error=0.5)
        self.assertEqual(chosen.kept, ("clean", "four"))

    def test_equal_error_prefers_fewer_bytes(self) -> None:
        cells = [
            cell("wide", "6", 0.2, full_bytes=200),
            cell("narrow", "6", 0.2, full_bytes=100),
        ]
        chosen = select_e1(cells, kv8_bits=9.0, kv4_error=0.0)
        self.assertEqual(chosen.kept[0], "narrow")

    def test_bits_not_below_kv8_are_dropped(self) -> None:
        cells = [cell("too-wide", "6", 0.01, bits=9.0)]
        chosen = select_e1(cells, kv8_bits=9.0, kv4_error=1.0)
        self.assertEqual(chosen.kept, ())


if __name__ == "__main__":
    unittest.main()
