"""用在线布局的独立字节公式复核静态候选成本与质量来源。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from kvstudy.paths import project_root


def _read(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _same(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(f"完整成本与独立计数不同：{actual}，{expected}")


def _row(value: dict[str, Any], row: dict[str, Any]) -> None:
    """在线 H=0 布局以逐 token 载荷和参数重算；服务窗口独立取算术均值。"""
    configuration = value["actual_configurations"][row["format"]]
    bits = configuration["layer_bits"]
    heads, dim = value["kv_heads"] * row["batch"], value["head_dim"]
    length, layers = row["length"], value["n_layers"]
    if (len(bits) != layers or configuration["key_mode"] != "per_token"
            or configuration["value_mode"] != "per_token" or row["residual_bits"] != 0
            or row["rope"] != configuration["rope"]):
        raise RuntimeError("本轮成本出现未冻结的格式或残留")
    k_payload = sum(k for k, _ in bits) * heads * dim // 8
    v_payload = sum(v for _, v in bits) * heads * dim // 8
    params = layers * heads * math.ceil(dim / 32) * 4
    k_token, v_token = k_payload + params, v_payload + params
    token = k_token + v_token
    for name, expected in (("k_payload", k_payload * length),
                           ("v_payload", v_payload * length),
                           ("k_params", params * length), ("v_params", params * length),
                           ("padding", 0), ("residual", 0), ("labels", 0),
                           ("stored", token * length)):
        _same(row["components"][name], expected)
    _same(row["resident_bytes"], token * length)
    _same(row["effective_bits"], token * 8 / (2 * layers * heads * dim))
    converted = row["prefill_conversion_once"]
    _same(converted["converted_write_raw_bytes"], token * length)
    bf16 = 4 * layers * heads * length * dim
    _same(converted["bf16_input_read_bytes"], bf16)
    _same(converted["simultaneous_bf16_and_converted_upper_bytes"], bf16 + token * length)
    if length == 131072:
        if "decode_256" in row:
            raise RuntimeError("128K 只能核算布局")
        return
    decode = row["decode_256"]
    if (decode["steps"] != 256 or decode["initial_history"] != length - 256
            or decode["final_history"] != length):
        raise RuntimeError("服务窗口没有保持连续 256 步")
    mean_history = length - 128.5
    averages = decode["average_bytes"]
    for name, expected in (("read_k_lower_per_step", k_token * mean_history),
                           ("read_v_lower_per_step", v_token * mean_history),
                           ("read_k_full_per_step", k_token * mean_history),
                           ("read_v_full_per_step", v_token * mean_history),
                           ("write_per_step", token),
                           ("io_lower_per_step", token * (mean_history + 1))):
        _same(averages[name], expected)
    rotation = (3 * layers * heads * dim * mean_history
                if configuration["rope"] == "pre" else 0)
    _same(decode["onchip_and_arithmetic"]["pre_rope_mul_adds_per_step"], rotation)
    for bound in decode["bandwidth_lower_bounds"]:
        width = bound["bandwidth_TBps"] * 1e12
        _same(bound["shared_read_lower_s"], token * mean_history / width)
        _same(bound["shared_io_lower_s"], token * (mean_history + 1) / width)
        _same(bound["split_equal_read_lower_s"],
              2 * max(k_token, v_token) * mean_history / width)


def audit(root: Path) -> dict[str, Any]:
    """复核全部行、固定容量核对值、B=4 代表条件和冻结结果依赖。"""
    folder = root / "research/phase1/experiments/storage_service_cost/results"
    quality = root / (
        "research/phase1/experiments/static_layer_allocation/results/final_quality_review.json"
    )
    if _read(quality)["status"] != "passed":
        raise RuntimeError("质量结果尚未完成审核")
    publication_path = root / "research/.runtime/static-final/cost_publication.json"
    publication = _read(publication_path)
    for relative, expected in publication["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("成本执行来源与发布清单不同")
    sources, counts = {}, {}
    for model in ("qwen", "llama"):
        path = folder / model / "layout.json"
        value = _read(path)
        if value["protocol"] != "static-allocation-independent-generation":
            raise RuntimeError("成本结果不是本轮冻结协议")
        layout_code = root / "research/phase1/experiments/storage_service_cost/code/layout.py"
        if value["layout_code_sha256"] != hashlib.sha256(layout_code.read_bytes()).hexdigest():
            raise RuntimeError("成本结果记录的源码与实际执行副本不同")
        selection = _read(root / (
            "research/phase1/experiments/static_layer_allocation/results/expanded_quality"
        ) / model / "selection.json")
        if (value["quality"] != selection["cells"]
                or value["candidate_stops"] != selection["stops"]):
            raise RuntimeError("成本结果的质量状态与最终候选不同")
        for relative, expected in value["source_sha256"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("成本引用的质量或层表来源发生变化")
        if len(value["rows"]) != (36 if model == "qwen" else 37):
            raise RuntimeError("成本主矩阵不完整")
        identities = {(row["format"], row["length"], row["batch"], row["block_tokens"])
                      for row in value["rows"]}
        if len(identities) != len(value["rows"]):
            raise RuntimeError("成本主矩阵包含重复条件")
        names = {row["format"] for row in value["rows"] if row["batch"] == 1}
        expected = {(name, length, 1, block) for name in names
                    for length in (4096, 32768, 131072) for block in (32, 64, 128)}
        if {item for item in identities if item[2] == 1} != expected:
            raise RuntimeError("冻结长度与物理块没有完整交叉")
        for row in value["rows"]:
            _row(value, row)
        main = [row for row in value["rows"] if row["length"] == 32768
                and row["batch"] == 1 and row["block_tokens"] == 64]
        expected_mib = [560, 672, 784, 1008] if model == "qwen" else [1280, 1536, 1792, 2304]
        if sorted(row["resident_bytes"] / 2 ** 20 for row in main) != expected_mib:
            raise RuntimeError("32K 完整容量与冻结核对值不同")
        _same(value["mixed_32k"]["relative_k8v4_capacity_saved"], 1 / 7)
        _same(value["mixed_32k"]["relative_kv4_capacity_extra"], 1 / 5)
        if model == "llama":
            representative = [row for row in value["rows"] if row["batch"] == 4]
            if (len(representative) != 1
                    or representative[0]["format"] != "k8v4-per_token-post"
                    or representative[0]["block_tokens"] != 64):
                raise RuntimeError("B=4 既定代表条件发生改变")
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[model] = len(value["rows"])
    payload = {
        "status": "passed", "counts": counts, "results": sources,
        "quality_review_sha256": hashlib.sha256(quality.read_bytes()).hexdigest(),
        "execution_source": {
            "path": publication_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(publication_path.read_bytes()).hexdigest(),
        },
        "review_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": ["independent_payload_and_parameter_formula", "full_32k_capacity",
                   "decode_256_reads_writes_rotation", "assumed_bandwidth_bounds",
                   "quality_source_and_fixed_batch_representative"],
    }
    (folder / "review.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(audit(project_root()), ensure_ascii=False), flush=True)
