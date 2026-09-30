"""E1 的 6 bit / 4 bit 档选择。追加格不进入这里。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E1Cell:
    """一格的两折平均误差和容量。invalid 表示留出折上出现过无效组。"""

    name: str
    tier: str
    error: float
    effective_bits: float
    full_bytes: int
    invalid: bool


@dataclass(frozen=True)
class E1Selection:
    """进入 E2 的表示。kept 为空表示只记录失败。"""

    kept: tuple[str, ...]
    reason: str


def select_e1(cells: list[E1Cell], kv8_bits: float, kv4_error: float) -> E1Selection:
    """按误差、无效组和完整字节选择。容量有效位宽必须严格低于均匀 KV8。"""
    eligible = [cell for cell in cells if cell.effective_bits < kv8_bits]
    band6 = _ranked([cell for cell in eligible if cell.tier == "6"])
    band4 = _ranked(
        [
            cell
            for cell in eligible
            if cell.tier == "4" and cell.error < kv4_error
        ]
    )
    if band4:
        return E1Selection((band6[0].name, band4[0].name) if band6 else (band4[0].name,), "6bit+4bit")
    if len(band6) >= 2:
        return E1Selection((band6[0].name, band6[1].name), "4bit 档为空，6bit 档留下两种")
    if len(band6) == 1:
        return E1Selection((band6[0].name,), "4bit 档为空，6bit 档只有一种")
    return E1Selection((), "没有可留的非基线表示")


def _ranked(cells: list[E1Cell]) -> list[E1Cell]:
    return sorted(cells, key=lambda cell: (cell.invalid, cell.error, cell.full_bytes, cell.name))
