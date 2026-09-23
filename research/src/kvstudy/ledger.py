"""容量、片外传输和片上访问三本账。128 字节对齐是接口假设。"""

from __future__ import annotations

from dataclasses import dataclass

ALIGNMENT_BYTES = 128
WEIGHT_FLOAT_BYTES = 2


@dataclass(frozen=True)
class RegionBytes:
    """一块连续存储。padding 是补到对齐边界的字节。"""

    raw_bytes: int
    padding_bytes: int

    @property
    def stored_bytes(self) -> int:
        return self.raw_bytes + self.padding_bytes


@dataclass(frozen=True)
class Ledger:
    """一次布局核算。字节数都是整数。"""

    k_payload: RegionBytes
    v_payload: RegionBytes
    k_params: RegionBytes
    v_params: RegionBytes
    residual: RegionBytes
    labels: RegionBytes
    elements: int
    open_group_tokens: int
    tail_valid_tokens: int
    block_count: int

    @property
    def resident_bytes(self) -> int:
        return (
            self.k_payload.stored_bytes
            + self.v_payload.stored_bytes
            + self.k_params.stored_bytes
            + self.v_params.stored_bytes
            + self.residual.stored_bytes
            + self.labels.stored_bytes
        )

    @property
    def effective_bits(self) -> float:
        if self.elements == 0:
            return 0.0
        return self.resident_bytes * 8 / self.elements


def align_up(raw_bytes: int, alignment: int = ALIGNMENT_BYTES) -> RegionBytes:
    """把原始字节补齐到 alignment 的整数倍。"""
    if raw_bytes < 0:
        raise ValueError("字节数不能为负")
    if alignment <= 0:
        raise ValueError("对齐粒度必须为正")
    stored = (raw_bytes + alignment - 1) // alignment * alignment
    return RegionBytes(raw_bytes=raw_bytes, padding_bytes=stored - raw_bytes)


def _payload_bytes(elements: int, bits: int) -> int:
    if elements < 0 or bits <= 0:
        raise ValueError("元素数或位宽无效")
    if (elements * bits) % 8 != 0:
        raise ValueError("载荷位数不能凑成整字节")
    return elements * bits // 8


def _group_count(tokens: int, channels: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError("组大小必须为正")
    full = channels // group_size
    tail = 1 if channels % group_size else 0
    return tokens * (full + tail)


def account_pair(
    *,
    layers: int,
    kv_heads: int,
    seq: int,
    dim: int,
    k_bits: int,
    v_bits: int,
    k_mode: str,
    v_mode: str,
    g_c: int,
    g_t: int,
    block_tokens: int,
    residual_h: int = 0,
    residual_r: int = 0,
    label_bytes_per_token: int = 0,
    weight_float_bytes: int = WEIGHT_FLOAT_BYTES,
) -> Ledger:
    """核算 K/V 分存时的驻留容量。

    per_token 沿通道分组；per_channel 每个通道一组，参数在整段序列上共享；
    cross_token 沿 token 分组，未闭合组以权重浮点保留。
    """
    if seq < 0 or block_tokens <= 0:
        raise ValueError("序列或物理块长度无效")
    units = layers * kv_heads
    elements_one = units * seq * dim
    elements = elements_one * 2
    k_payload, k_params, k_open = _side(
        units, seq, dim, k_bits, k_mode, g_c, g_t, weight_float_bytes
    )
    v_payload, v_params, v_open = _side(
        units, seq, dim, v_bits, v_mode, g_c, g_t, weight_float_bytes
    )
    open_tokens = max(k_open, v_open)
    residual_raw = _residual_bytes(
        units,
        seq,
        dim,
        residual_h,
        residual_r,
        open_tokens,
        g_c,
        weight_float_bytes,
    )
    labels = align_up(units * seq * label_bytes_per_token)
    if seq == 0:
        tail_valid = 0
        block_count = 0
    else:
        block_count = (seq + block_tokens - 1) // block_tokens
        tail_valid = seq - (block_count - 1) * block_tokens
    return Ledger(
        k_payload=k_payload,
        v_payload=v_payload,
        k_params=k_params,
        v_params=v_params,
        residual=align_up(residual_raw),
        labels=labels,
        elements=elements,
        open_group_tokens=open_tokens,
        tail_valid_tokens=tail_valid,
        block_count=block_count,
    )


def _side(
    units: int,
    seq: int,
    dim: int,
    bits: int,
    mode: str,
    g_c: int,
    g_t: int,
    weight_float_bytes: int,
) -> tuple[RegionBytes, RegionBytes, int]:
    if mode == "per_token":
        stored_tokens = seq
        groups = units * _group_count(seq, dim, g_c)
        open_tokens = 0
    elif mode == "per_channel":
        stored_tokens = seq
        groups = units * dim
        open_tokens = 0
    elif mode == "cross_token":
        if g_t <= 0:
            raise ValueError("跨 token 组长度必须为正")
        closed = (seq // g_t) * g_t
        stored_tokens = closed
        groups = units * closed // g_t * dim
        open_tokens = seq - closed
    else:
        raise ValueError(f"未知表示: {mode}")
    payload = align_up(_payload_bytes(units * stored_tokens * dim, bits))
    params = align_up(groups * 4)
    # 未闭合组的高精度副本由调用方计入残留，这里只返回 token 数。
    del weight_float_bytes
    return payload, params, open_tokens


def _residual_bytes(
    units: int,
    seq: int,
    dim: int,
    residual_h: int,
    residual_r: int,
    open_tokens: int,
    g_c: int,
    weight_float_bytes: int,
) -> int:
    """近期残留与未闭合组的并集。未闭合组始终按权重浮点计，避免和 H 副本重复。"""
    if residual_h not in (0, 8, 16):
        raise ValueError(f"残留精度只能是 0、8 或 16: {residual_h}")
    window = min(seq, residual_r) if residual_h else 0
    # 未闭合组位于序列末尾，和残留窗口从末尾重叠。
    overlap = min(open_tokens, window)
    high_only = open_tokens
    h_only = window - overlap
    per_token = units * dim
    raw = high_only * per_token * weight_float_bytes
    if residual_h == 16:
        raw += h_only * per_token * weight_float_bytes
    elif residual_h == 8:
        raw += _payload_bytes(h_only * per_token, 8)
        raw += h_only * units * _group_count(1, dim, g_c) * 4
    return raw


def nominal_bits(k_bits: int, v_bits: int) -> float:
    """K 与 V 元素数相同时的名义位宽。"""
    return (k_bits + v_bits) / 2


def offchip_bytes_per_token(
    *,
    layers: int,
    kv_heads: int,
    dim: int,
    history: int,
    k_bits: int,
    v_bits: int,
    k_mode: str,
    v_mode: str,
    g_c: int,
    g_t: int,
    residual_h: int = 0,
    residual_r: int = 0,
    pre_rope: bool = False,
) -> dict[str, float]:
    """一个新 token 的片外读写。同一字节只计一次。

    pre_rope 为真时，Key 的读出后再做旋转；旋转本身不额外增加片外读取，
    因为读入的就是未旋转的 Key。
    """
    del pre_rope
    read_tokens = history
    ledger = account_pair(
        layers=layers,
        kv_heads=kv_heads,
        seq=max(history, 1),
        dim=dim,
        k_bits=k_bits,
        v_bits=v_bits,
        k_mode=k_mode,
        v_mode=v_mode,
        g_c=g_c,
        g_t=g_t,
        block_tokens=64,
        residual_h=residual_h,
        residual_r=residual_r,
    )
    # 用 history 与 history+1 的驻留差作为写入，避免把对齐跳变重复算进每个 token。
    before = account_pair(
        layers=layers,
        kv_heads=kv_heads,
        seq=history,
        dim=dim,
        k_bits=k_bits,
        v_bits=v_bits,
        k_mode=k_mode,
        v_mode=v_mode,
        g_c=g_c,
        g_t=g_t,
        block_tokens=64,
        residual_h=residual_h,
        residual_r=residual_r,
    )
    after = account_pair(
        layers=layers,
        kv_heads=kv_heads,
        seq=history + 1,
        dim=dim,
        k_bits=k_bits,
        v_bits=v_bits,
        k_mode=k_mode,
        v_mode=v_mode,
        g_c=g_c,
        g_t=g_t,
        block_tokens=64,
        residual_h=residual_h,
        residual_r=residual_r,
    )
    # 写入按原始载荷计。对齐补齐属于驻留容量，不把扩容时的 padding 跳变算进每个 token。
    write = max(0, _raw_bytes(after) - _raw_bytes(before))
    read = _read_bytes(ledger, read_tokens, history)
    return {"read_bytes": float(read), "write_bytes": float(write)}


def _raw_bytes(ledger: Ledger) -> int:
    return (
        ledger.k_payload.raw_bytes
        + ledger.v_payload.raw_bytes
        + ledger.k_params.raw_bytes
        + ledger.v_params.raw_bytes
        + ledger.residual.raw_bytes
        + ledger.labels.raw_bytes
    )


def _read_bytes(ledger: Ledger, read_tokens: int, history: int) -> int:
    if history == 0:
        return 0
    payload = ledger.k_payload.raw_bytes + ledger.v_payload.raw_bytes
    params = ledger.k_params.raw_bytes + ledger.v_params.raw_bytes
    # 参数在整个历史上共享或随 token 增长，读取当前已驻留的参数。
    scale = read_tokens / history
    return int(round(payload * scale)) + params + ledger.residual.raw_bytes


def onchip_bytes_per_token(
    *,
    layers: int,
    kv_heads: int,
    dim: int,
    history: int,
    k_bits: int,
    v_bits: int,
    g_c: int,
) -> dict[str, int]:
    """解包、反量化中间量和参数缓存。不加入片外 bytes/token。"""
    elements_one = layers * kv_heads * history * dim
    unpack = _payload_bytes(elements_one, k_bits) + _payload_bytes(elements_one, v_bits)
    dequant = elements_one * 2 * 4
    groups = layers * kv_heads * _group_count(history, dim, g_c)
    params = groups * 4 * 2
    return {"unpack_bytes": unpack, "dequant_bytes": dequant, "param_cache_bytes": params}
