"""独立核对复用来源、传输差额、操作口径和同格式共同成本。"""

from __future__ import annotations

import argparse
import io
import json
import math
import unittest
from pathlib import Path
from typing import Any

from experiments.storage_service_cost.code.review import _row
from experiments.storage_service_cost.code.service import digest


def audit(root: Path) -> dict[str, Any]:
    """审核新服务结果；不改写旧布局与旧审核的实际来源。"""
    folder = root / "research/phase1/experiments/storage_service_cost"
    path = folder / "results/service.json"
    data = json.loads(path.read_text("utf-8"))
    for relative, expected in data["source_sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError(f"服务结果来源变动：{relative}")
    suite = unittest.TestLoader().discover(str(root / "research/tests"),
                                          pattern="test_storage_service.py")
    output = io.StringIO()
    tests = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.testsRun != 8:
        raise RuntimeError(output.getvalue())
    counts = {}
    for model, result in data["models"].items():
        layout = json.loads((folder / "results" / model / "layout.json").read_text("utf-8"))
        for row in layout["rows"]:
            _row(layout, row)
        expected = 48 if model == "qwen" else 50
        assert len(result["transfer_rows"]) == expected
        for item in result["transfer_rows"]:
            streams = item["average_per_batch_step"]
            heads = layout["kv_heads"] * item["batch"] * layout["n_layers"]
            configs = layout["actual_configurations"][item["format"]]
            n4 = sum((k == 4) + (v == 4) for k, v in configs["layer_bits"])
            four_streams = n4 * layout["kv_heads"] * item["batch"]
            param_streams = 2 * heads
            if item["mode"] == "immediate_rmw":
                # 完整周期里 4-bit 载荷均匀落在两个半事务；参数落在八个位置。
                tail = 32 * four_streams + 56 * param_streams
                rmw = 64 * four_streams + 112 * param_streams
                extra_write = 64 * four_streams + 112 * param_streams
                expected_diff = tail + rmw + extra_write
                assert item["difference"]["read_tail_waste"] == tail
            else:
                expected_diff = -(32 * four_streams + 56 * param_streams)
                assert item["difference"]["rmw_read"] == 0
                assert item["difference"]["append_padding_or_rewrites"] == 0
            assert item["interface_io_bytes_per_step"] - item[
                "effective_io_bytes_per_step"] == expected_diff
            assert item["interface_io_bytes_per_generated_token"] * item["batch"] == item[
                "interface_io_bytes_per_step"]
            assert sum(v["read"] + v["append_write"] + v["rmw_read"]
                       for v in streams.values()) == item["interface_io_bytes_per_step"]
        for item in result["work_rows"]:
            ops = item["operations_per_batch_step"]
            assert ops["dequant_mul"] == ops["dequant_add"]
            assert ops["dequant_mul"] == ops["historical_values_int4"] + ops[
                "historical_values_int8"]
        for item in result["comparisons"]:
            if item["reference"] == "k8v4":
                assert item["extra_rope_scalar_flops"] == 0
            for budget in item["budgets"]:
                assert math.isclose(budget["extra_time_budget_s"]
                                    * budget["bandwidth_TBps"] * 1e12,
                                    item["interface_bytes_saved"], abs_tol=1e-6)
        selection_path = root / (
            "research/phase1/experiments/static_layer_allocation/results/expanded_quality"
        ) / model / "selection.json"
        selection = json.loads(selection_path.read_text("utf-8"))
        for name, tasks in result["quality"].items():
            for task, cell in tasks.items():
                actual = selection["cells"][name][task]
                assert cell == {"state": actual["state"], "n": actual["n"]}
        counts[model] = {"layout_rows_reused": len(layout["rows"]),
                         "transfer_rows": len(result["transfer_rows"]),
                         "work_rows": len(result["work_rows"])}
    payload = {"status": "passed", "counts": counts,
               "service_sha256": digest(path), "review_code_sha256": digest(Path(__file__)),
               "regression": {"tests": tests.testsRun, "failures": len(tests.failures),
                              "errors": len(tests.errors),
                              "source": "research/tests/test_storage_service.py",
                              "sha256": digest(root / "research/tests/test_storage_service.py")},
               "legacy_formula_auditor_sha256": digest(folder / "code/review.py"),
               "checks": ["unchanged_73_layout_rows_and_quality_dependencies",
                          "independent_transaction_cycle_formula", "per_generated_token",
                          "same_format_rotation_cancels", "component_conservation",
                          "separate_fma_and_scalar_ops", "frozen_quality_states"]}
    (folder / "results/service_review.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().root.resolve()), ensure_ascii=False))
