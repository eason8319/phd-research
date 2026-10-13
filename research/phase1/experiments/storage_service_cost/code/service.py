"""登记读出操作、逐层依赖与实现吞吐要求，保留未知实际吞吐。"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

from kvstudy.artifacts import file_hash as digest
from experiments.storage_service_cost.code.transfer import stream_step, transfer_row


def pipeline_time(jobs: list[list[float]], capacity: int = 2) -> float:
    """有限队列流水：每阶段独占资源，最多 capacity 个在途 tile；返回秒。"""
    if capacity < 1:
        raise ValueError("在途 tile 容量必须为正")
    if not jobs:
        return 0.0
    clocks = [0.0] * len(jobs[0])
    completed = []
    for index, job in enumerate(jobs):
        if len(job) != len(clocks) or any(not math.isfinite(value) or value < 0 for value in job):
            raise ValueError("流水阶段必须等长且时间非负")
        for stage, duration in enumerate(job):
            prior = (clocks[stage - 1] if stage else
                     completed[index - capacity] if index >= capacity else 0.0)
            clocks[stage] = max(clocks[stage], prior) + duration
        completed.append(clocks[-1])
    return clocks[-1]


def layer_service_time(k_jobs: list[list[float]], v_jobs: list[list[float]],
                       quant_s: float, softmax_s: float, append_s: float) -> float:
    """串行层的服务时间；K 完成后才做 softmax 和 V，追加写入不隐藏。"""
    if min(quant_s, softmax_s, append_s) < 0:
        raise ValueError("服务时间不可为负")
    return quant_s + pipeline_time(k_jobs) + softmax_s + pipeline_time(v_jobs) + append_s


def predict_time(layout: dict[str, Any], name: str, history: int, batch: int,
                 block: int, mode: str, rates: dict[str, float]) -> dict[str, Any]:
    """代入经测量的并发吞吐后逐层递推；所有吞吐、启动及系数时间必须显式提供。

    带宽单位 B/s，值吞吐为 value/s，dot/rope 为普通 FLOP/s；不使用设备峰值默认值。
    解码和旋转采用流式 FIFO；KV tile、参数及 score 使用独立 SRAM bank。
    """
    rate_names = ("memory", "decode4", "decode8", "quant4", "quant8", "rope",
                  "dot", "softmax_arithmetic", "exp", "division", "sram_write", "sram_read",
                  "parameter_sram", "score_sram", "coefficient_sram")
    if any(key not in rates or not math.isfinite(rates[key]) or rates[key] <= 0
           for key in rate_names):
        raise ValueError("必须显式提供全部正的并发实测吞吐")
    if any(key not in rates or not math.isfinite(rates[key]) or rates[key] < 0
           for key in ("launch_s", "coefficients_s")):
        raise ValueError("必须显式提供启动和三角参数时间")
    if history < 0 or batch < 1:
        raise ValueError("历史长度或批量非法")
    shape = layout["actual_configurations"][name]
    dim, heads = layout["head_dim"], layout["kv_heads"] * batch
    queries = layout["query_heads"] * batch
    layer_times = []
    for kbits, vbits in shape["layer_bits"]:
        phases = []
        quant_s = heads * dim * (1 / rates[f"quant{kbits}"] + 1 / rates[f"quant{vbits}"])
        if shape["rope"] == "post":
            quant_s += max(3 * heads * dim / rates["rope"],
                           dim * 4 / rates["coefficient_sram"])
        append_bytes = 0.0
        for side, bits in (("k", kbits), ("v", vbits)):
            jobs = []
            stride = dim * bits // 8
            for start in range(0, history, 64):
                tokens = min(64, history - start)
                data = stream_step(tokens, stride, block, 128, mode)
                params = stream_step(tokens, dim // 32 * 4, block, 128, mode)
                read = heads * (data["read"] + params["read"])
                values = tokens * heads * dim
                decode = (values / rates[f"decode{bits}"]
                          + values / 32 * 16 / rates["parameter_sram"])
                rotate = (max(3 * values / rates["rope"],
                              tokens * dim * 4 / rates["coefficient_sram"])
                          if side == "k" and shape["rope"] == "pre" else 0)
                # SRAM 端口与本阶段取最大；跨阶段必须确有独立端口。
                produce = 4 * values / rates["sram_write"]
                decode = max(decode, produce) if not rotate else decode
                rotate = max(rotate, produce) if rotate else 0
                dot = max(2 * queries * tokens * dim / rates["dot"],
                          4 * values / rates["sram_read"])
                dot += 4 * queries * tokens / rates["score_sram"]
                jobs.append([read / rates["memory"], decode, rotate, dot])
            current = heads * dim
            jobs.append([0, current / rates[f"decode{bits}"],
                         max(3 * current / rates["rope"],
                             dim * 4 / rates["coefficient_sram"])
                         if side == "k" and shape["rope"] == "pre" else 0,
                         2 * queries * dim / rates["dot"]
                         + 4 * queries / rates["score_sram"]])
            phases.append(jobs)
            for size in (stride, dim // 32 * 4):
                tail = stream_step(history, size, block, 128, mode)
                append_bytes += heads * (tail["append_write"] + tail["rmw_read"])
        positions = queries * (history + 1)
        softmax = ((4 * positions - 2 * queries) / rates["softmax_arithmetic"]
                   + positions / rates["exp"] + positions / rates["division"]
                   + 24 * positions / rates["score_sram"])
        # QK 写和 AV 读已逐 tile 计入 dot 阶段，其余六次 score 访问在 softmax。
        query_rotation = 3 * queries * dim / rates["rope"]
        layer_times.append(layer_service_time(
            phases[0], phases[1], quant_s + query_rotation + rates["launch_s"],
            softmax, append_bytes / rates["memory"],
        ))
    return {"total_s": rates["coefficients_s"] + sum(layer_times),
            "layer_s": layer_times, "rates": rates,
            "source_type": "conditional_execution_model_not_measured_latency"}


def work_row(layout: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """按值、组、比较、FMA 和普通浮点运算分别计数；历史与当前分开。"""
    bits = layout["actual_configurations"][row["format"]]["layer_bits"]
    layers, heads = len(bits), layout["kv_heads"] * row["batch"]
    qheads, dim = layout["query_heads"] * row["batch"], layout["head_dim"]
    history = row["length"] - 128.5
    one = layers * heads * dim
    n4 = sum((k == 4) + (v == 4) for k, v in bits) * heads * dim
    elements = 2 * one * history
    groups_new = 2 * one // 32
    score_count = layers * qheads * (history + 1)
    rotation_values = one * history if row["rope"] == "pre" else 0
    # x*c + rotate(x)*s 为两次乘法和一次加法，或一次乘法加一次 FMA。
    operations = {
        "historical_values_int4": n4 * history,
        "historical_values_int8": (2 * one - n4) * history,
        "unpack_int4_bitwise_ops": n4 * history * 1.5,
        "integer_to_fp32_conversions": elements,
        "parameter_fp16_to_fp32_conversions": elements / 16,
        "dequant_mul": elements, "dequant_add": elements,
        "dequant_fma_alternative": elements,
        "rope_historical_mul": 2 * rotation_values,
        "rope_historical_add": rotation_values,
        "rope_mul_plus_fma_alternative_each": rotation_values,
        "qk_historical_fma": layers * qheads * dim * history,
        "av_historical_fma": layers * qheads * dim * history,
        "qk_current_fma": layers * qheads * dim,
        "av_current_fma": layers * qheads * dim,
        "softmax_scale_mul": score_count,
        "softmax_max_comparisons": layers * qheads * history,
        "softmax_sub_exp_div_each": score_count,
        "softmax_sum_add": layers * qheads * history,
        "append_values": 2 * one, "append_groups": groups_new,
        "append_min_max_comparisons": groups_new * 62,
        "append_range_sub_and_div_each": groups_new,
        "append_parameter_directed_roundings": groups_new * 2,
        "append_sub_div_round_each": 2 * one,
        "append_clip_comparisons": 4 * one,
        "append_int4_pack_bitwise_ops": 2 * n4,
        "current_integer_to_fp32_conversions": 2 * one,
        "current_parameter_fp16_to_fp32_conversions": groups_new * 2,
        "current_reconstruction_mul_and_add_each": 2 * one,
        "current_key_rope_mul": 2 * one,
        "current_key_rope_add": one,
        "current_query_rope_mul": 2 * layers * qheads * dim,
        "current_query_rope_add": layers * qheads * dim,
    }
    # 每个位置的频率相位只计算一次，并在全部层和 head 复用。
    coeff = {"phase_mul": (history + 1) * dim / 2,
             "sin_and_cos_evaluations_each": (history + 1) * dim / 2,
             "frequency_vector_bytes": dim // 2 * 4,
             "table_reserved_bytes": row["length"] * dim * 4,
             "offchip_table_read_bytes": 0,
             "onchip_table_write_bytes": (history + 1) * dim * 4,
             "onchip_table_read_bytes": layers * (history + 1) * dim * 4,
             "model_scaling_setup_time_s": None}
    if row["rope"] != "pre":
        coeff.update({"phase_mul": dim / 2, "sin_and_cos_evaluations_each": dim / 2,
                      "table_reserved_bytes": dim * 4, "onchip_table_write_bytes": dim * 4,
                      "onchip_table_read_bytes": layers * dim * 4})
    onchip = {
        "kv_fp32_tile_write_plus_read_bytes": 8 * elements,
        "rope_extra_tile_write_plus_read_if_unfused_bytes": 8 * rotation_values,
        "softmax_array_access_bytes": 32 * score_count,
        "softmax_array_access_explanation": (
            "QK write, scale read/write, max read, exp read/write, sum read, AV read; "
            "normalization division fuses into AV read"
        ),
        "score_buffer_reserved_bytes": 4 * qheads * row["length"],
        "double_kv_tile_reserved_bytes": 2 * heads * dim * 64 * 4,
        "fp32_parameter_tile_reserved_bytes": 2 * heads * (dim // 32) * 2 * 64 * 4,
        "packed_tile_fifo_reserved_bytes": 2 * heads * (dim + dim // 32 * 4) * 64,
        "parameter_tile_write_plus_broadcast_read_bytes": elements / 32 * 16,
        "unpacked_codes_materialization": "register stream; no uint8 history array",
        "full_history_fp32_kv_materialized": False,
        "post_rope_query_and_output_register_bytes": 2 * qheads * dim * 4,
        "current_kv_fp32_reserved_bytes": 2 * heads * dim * 4,
        "current_packed_kv_and_params_upper_bytes": 2 * heads * (dim + dim // 32 * 4),
    }
    old_ops = row["decode_256"]["onchip_and_arithmetic"]
    assert operations["qk_historical_fma"] * 2 == old_ops["qk_flops_per_step"]
    assert rotation_values * 3 == old_ops["pre_rope_mul_adds_per_step"]
    return {"format": row["format"], "length": row["length"], "batch": row["batch"],
            "block_tokens": row["block_tokens"], "operations_per_batch_step": operations,
            "rope_coefficients": coeff, "onchip": onchip,
            "per_layer_bits": bits, "mean_history": history,
            "append_exception_checks": "finite/constant/subnormal/overflow checks per group; "
            "data dependent counts require implementation measurement"}


def compare_rows(layout: dict[str, Any], transfers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用两配置额外开销之差建立临界值，保留实际吞吐为未知。"""
    comparisons = []
    for mode in ("immediate_rmw", "combined"):
        for length in (4096, 32768):
            rows = {r["format"]: r for r in transfers if r["mode"] == mode
                    and r["length"] == length and r["batch"] == 1 and r["block_tokens"] == 64}
            for name, candidate in rows.items():
                for reference in ("k8v4", "kv8"):
                    if reference == name:
                        continue
                    base = rows[reference]
                    configs = layout["actual_configurations"]
                    cbits, rbits = configs[name]["layer_bits"], configs[reference]["layer_bits"]
                    changed = sum((r == 8 and c == 4) for rs, cs in zip(rbits, cbits)
                                  for r, c in zip(rs, cs))
                    up = sum((r == 4 and c == 8) for rs, cs in zip(rbits, cbits)
                             for r, c in zip(rs, cs))
                    if up:
                        continue
                    n = changed * layout["kv_heads"] * layout["head_dim"] * (length - 128.5)
                    saving = base["interface_io_bytes_per_step"] - candidate[
                        "interface_io_bytes_per_step"]
                    effective_saving = base["effective_io_bytes_per_step"] - candidate[
                        "effective_io_bytes_per_step"]
                    rotation_delta = ((configs[name]["rope"] == "pre")
                                      - (configs[reference]["rope"] == "pre"))
                    rotation_delta *= (3 * layout["n_layers"] * layout["kv_heads"]
                                       * layout["head_dim"] * (length - 128.5))
                    comparisons.append({
                        "mode": mode, "length": length, "candidate": name,
                        "reference": reference, "effective_bytes_saved": effective_saving,
                        "interface_bytes_saved": saving, "changed_decode_values": n,
                        "extra_rope_scalar_flops": rotation_delta,
                        "condition": "N*(1/R4-1/R8) + delta_append + delta_rope "
                        "+ delta_coefficients + delta_onchip + delta_launch < delta_bytes/W",
                        "streaming_condition": "sum(layer_service_time(candidate)) + "
                        "candidate_coefficient_setup < corresponding reference total",
                        "budgets": [{"bandwidth_TBps": w,
                                     "extra_time_budget_s": saving / (w * 1e12),
                                     "decode_only_inverse_rate_gap_max_s_per_value": (
                                         saving / (w * 1e12 * n) if n else None),
                                     "decode_only_R4_floor_if_R8_infinite_values_s": (
                                         n * w * 1e12 / saving if saving > 0 else None)}
                                    for w in layout["bandwidth_TBps"]],
                    })
    return comparisons


def requirements(layout: dict[str, Any]) -> list[dict[str, Any]]:
    """满事务流水稳态资源门槛；不是含填充和 softmax 的完整服务时间。"""
    group_share = layout["query_heads"] / layout["kv_heads"]
    rows = []
    for bits in (4, 8):
        byte_per_value = bits / 8 + 4 / 32
        for w in layout["bandwidth_TBps"]:
            values = w * 1e12 / byte_per_value
            rows.append({"bits": bits, "bandwidth_TBps": w,
                         "decode_values_s_min": values,
                         "parameters_groups_s_min": values / 32,
                         "rope_scalar_flops_s_min_if_pre": values * 3,
                         "qk_or_av_flops_s_min": values * group_share * 2,
                         "kv_tile_sram_bytes_s_min": values * 8,
                         "parameter_sram_bytes_s_min": values / 32 * 16})
    return rows


def run(root: Path) -> dict[str, Any]:
    """复用 73 行布局，为含 decode 的 49 行建立传输与服务账本。"""
    folder = root / "research/phase1/experiments/storage_service_cost"
    cfg_path = root / "research/phase1/configs/storage_service.json"
    config = json.loads(cfg_path.read_text("utf-8"))
    if [config[key] for key in ("transaction_bytes", "group_channels",
                               "parameter_bytes_per_group", "tile_tokens")] != [128, 32, 4, 64]:
        raise RuntimeError("接口或操作假设改变，须先调整计数与审核")
    base_review = json.loads((folder / "results/review.json").read_text("utf-8"))
    if base_review["status"] != "passed":
        raise RuntimeError("复用布局尚未审核")
    sources = [cfg_path, folder / "results/review.json", Path(__file__),
               folder / "code/transfer.py", root / "research/src/kvstudy/quantize.py",
               root / "research/phase1/src/phase1/generation.py"]
    quality_path = root / (
        "research/phase1/experiments/static_layer_allocation/results/final_quality_review.json"
    )
    quality = json.loads(quality_path.read_text("utf-8"))
    if quality["status"] != "passed":
        raise RuntimeError("最新质量审核尚未通过")
    for relative, sha in quality["results"].items():
        if digest(root / relative) != sha:
            raise RuntimeError(f"最新质量审核引用实体改变：{relative}")
    payload: dict[str, Any] = {
        "source_type": config["source_type"], "config": config,
        "models": {}, "latency_measurement": None,
        "execution": {"platform": platform.system(), "python": platform.python_version(),
                      "device": "CPU", "authorization": "user-approved local CPU audit",
                      "gpu_used": False},
        "refreshed_audit_dependencies": [],
    }
    for model in ("qwen", "llama"):
        path = folder / "results" / model / "layout.json"
        if digest(path) != base_review["results"][path.relative_to(root).as_posix()]:
            raise RuntimeError("复用布局与既有审核校验不符")
        layout = json.loads(path.read_text("utf-8"))
        for relative, sha in layout["source_sha256"].items():
            if digest(root / relative) != sha:
                if (root / relative).resolve() != quality_path.resolve():
                    raise RuntimeError(f"层表或质量来源改变：{relative}")
                selection = json.loads((root / (
                    "research/phase1/experiments/static_layer_allocation/results/expanded_quality"
                ) / model / "selection.json").read_text("utf-8"))
                if (selection["cells"] != layout["quality"]
                        or selection["stops"] != layout["candidate_stops"]):
                    raise RuntimeError("更新后的审核对应不同质量选择")
                payload["refreshed_audit_dependencies"].append({
                    "model": model, "path": relative, "historical_sha256": sha,
                    "current_sha256": digest(root / relative),
                    "reason": "Current passed audit result hashes verified; frozen selection, "
                    "stops, quantization plan and layer tables unchanged; old layouts retained.",
                })
            sources.append(root / relative)
        sources.append(path)
        transfers = [transfer_row(layout, row, config, mode) for row in layout["rows"]
                     if "decode_256" in row for mode in config["modes"]]
        work = [work_row(layout, row) for row in layout["rows"] if "decode_256" in row]
        payload["models"][model] = {
            "reused_layout_rows": len(layout["rows"]), "transfer_rows": transfers,
            "work_rows": work, "comparisons": compare_rows(layout, transfers),
            "streaming_steady_state_requirements": requirements(layout),
            "quality": {name: {task: {"state": cell["state"], "n": cell["n"]}
                               for task, cell in tasks.items()}
                        for name, tasks in layout["quality"].items()},
            "reused_quality_scores": {
                name: {task: {"n": entries[name]["generated_n"],
                              "score_0_100": entries[name]["all_request_score"]}
                       for task, entries in quality["diagnostics"][model].items()
                       if name in entries}
                for name in ("kv8", "k8v4")
            },
        }
    payload["source_sha256"] = {path.relative_to(root).as_posix(): digest(path)
                                for path in sources}
    path = folder / "results/service.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().root.resolve()), ensure_ascii=False))
