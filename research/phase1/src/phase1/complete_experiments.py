"""补齐共同生成候选与残留，再完成三格式的静态 Key 分配 C 比较。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kvstudy.paths import project_root
from experiments.generation_error_sources.code.candidates import (
    KV8, TASKS, generation_candidates,
)

GENERATION = "experiments.generation_error_sources.code"
ALLOCATION = "experiments.static_layer_allocation.code"
COST = "experiments.storage_service_cost.code"


def _read(path: Path) -> dict[str, Any]:
    """调度读取结果不加载模型或区间计算依赖。"""
    return json.loads(path.read_text("utf-8"))


def _sha256(path: Path) -> str:
    """计算实体文件校验值，用于执行来源和复用核对。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExperimentCompletion:
    """单 GPU 顺序执行；每个步骤完成后保存恢复点，异常保留失败原因。"""

    def __init__(self, commit: str) -> None:
        """以实际发布提交识别执行，恢复时不混用其他提交的进度。"""
        self.root = project_root()
        self.commit = commit
        self.runtime = self.root / "research/.runtime"
        self.path = self.runtime / f"phase1-completion-{commit}.json"
        self.state = _read(self.path) if self.path.exists() else {
            "commit": commit, "status": "running", "completed_steps": [],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.state["commit"] != commit:
            raise RuntimeError("补测恢复点与执行提交不一致")

    def _save(self) -> None:
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        temporary.replace(self.path)

    def _run(self, module: str, *args: str) -> None:
        key = " ".join((module, *args))
        if key in self.state["completed_steps"]:
            return
        self.state.update(status="running", current_step=key)
        self._save()
        print("执行", key, flush=True)
        subprocess.run([sys.executable, "-u", "-m", module, *args], check=True,
                       cwd=self.root)
        self.state["completed_steps"].append(key)
        self._save()

    def _folder(self, experiment: str, model: str) -> Path:
        return self.root / "research/phase1/experiments" / experiment / "results" / model

    def _fresh_progress(self, *names: str) -> None:
        """保留旧提交的运行缓存原字节，新提交从独立的进度重新执行。"""
        for name in names:
            path = self.runtime / name
            if not path.exists():
                continue
            state = _read(path)
            if state.get("commit") == self.commit:
                continue
            folder = self.runtime / "hold" / "superseded_progress"
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / f"{path.stem}-{_sha256(path)[:16]}{path.suffix}"
            if destination.exists():
                if _sha256(destination) != _sha256(path):
                    raise RuntimeError("旧进度保存路径存在不相同的实体")
                path.unlink()
            else:
                path.rename(destination)

    def _generation(self, model: str, task: str, name: str, h: int = 0) -> None:
        folder = self._folder("generation_error_sources", model)
        suffix = f"-residual{h}" if h else ""
        path = folder / f"v_{task}_{name}{suffix}.json"
        if path.exists():
            result = _read(path)
            if (result["model"] != model or result["config"] != name
                    or result.get("residual_bits", 0) != h
                    or result["n"] != len(result["rows"])):
                raise RuntimeError(f"既有生成配置不一致：{path}")
            if task in TASKS:
                from experiments.generation_error_sources.code.select import _paired

                seeds = _read(self.root / "research/phase1/splits/seeds.json")
                _paired(_read(folder / f"v_{task}_{KV8}.json"), result, task,
                        seeds["bootstrap"], seeds["bootstrap_draws"])
            return
        legacy = f"-h{h}" if h else ""
        self._fresh_progress(
            f"generation-{model}-{task}-v-{name}{suffix}.json",
            f"e2-{model}-{task}-v-{name}{legacy}.json",
        )
        self._run(f"{GENERATION}.quantized_generation", model, task, "v", name,
                  self.commit, str(h))

    def _diagnostics(self, model: str, selection: dict[str, Any]) -> None:
        folder = self._folder("generation_error_sources", model)
        name = selection["selected_low_bit"]
        for module, filename, current, legacy in (
            ("trajectories", "trajectories.json", "generation-propagation",
             "generation-trajectories"),
        ):
            path = folder / filename
            if path.exists():
                record = _read(path)
                plan = self.root / "research/phase1/configs/trajectory_replay.json"
                if (record.get("schema") == "kv-propagation-v1"
                        and record.get("plan_sha256") == _sha256(plan)):
                    continue
            if module == "trajectories" and name is None:
                continue
            self._fresh_progress(f"{current}-{model}.json", f"{legacy}-{model}.json")
            self._run(f"{GENERATION}.{module}", model, self.commit)
        if name is None:
            return
        baseline = folder / "v_hotpot_long_bf16.json"
        if not baseline.exists():
            self._fresh_progress(f"generation-{model}-hotpot_long-v.json",
                                 f"e2-{model}-hotpot_long-v.json")
            self._run(f"{GENERATION}.long_baseline", model, "hotpot_long", "v",
                      self.commit)
        rows = [row for row in _read(baseline)["rows"] if row["new_tokens"] >= 256]
        if rows and sum(row["f1"] for row in rows) / len(rows) >= 40:
            self._generation(model, "hotpot_long", KV8)
            self._generation(model, "hotpot_long", name)
            self._generation(model, "hotpot_long", name,
                             int(selection["e3_residual_bits"]) or 16)
        self._run(f"{GENERATION}.select_long_decode", model, self.commit)

    def _layers(self, model: str, selection: dict[str, Any]) -> None:
        """三格式独立测量；恢复时逐步核对来源，保留兼容的全层 KL 进度。"""
        from experiments.static_layer_allocation.code.protocol import (
            SCHEMA, calibration_source, code_sources, digest, frozen_sources, load_plan,
        )

        folder = self._folder("static_layer_allocation", model)
        plan_digest = digest(load_plan(self.root))
        probe_path = folder / "probe.json"
        allocations_path = folder / "allocations.json"
        gate_path = folder / "gate.json"

        def run_step(module: str) -> None:
            key = f"{ALLOCATION}.{module} {model} --commit {self.commit}"
            self.state["completed_steps"] = [
                old for old in self.state["completed_steps"] if old != key
            ]
            self._run(f"{ALLOCATION}.{module}", model, "--commit", self.commit)

        probe = _read(probe_path) if probe_path.exists() else {}
        probe_valid = (
            probe.get("schema") == SCHEMA and probe.get("plan_sha256") == plan_digest
            and probe.get("code_sha256") == code_sources(self.root)
            and probe.get("frozen_sources") == frozen_sources(self.root, model)
            and probe.get("calibration_source") == calibration_source(self.root, model)
        )
        if not probe_valid:
            run_step("probe")
            probe = _read(probe_path)
        allocation = _read(allocations_path) if allocations_path.exists() else {}
        allocation_valid = (
            allocation.get("schema") == SCHEMA and allocation.get("plan_sha256") == plan_digest
            and allocation.get("probe_sha256") == digest(probe)
            and allocation.get("allocation_digest") == digest({
                key: value for key, value in allocation.items() if key != "allocation_digest"
            })
        )
        if not allocation_valid:
            run_step("allocations")
            allocation = _read(allocations_path)
        gate = _read(gate_path) if gate_path.exists() else {}
        gate_valid = (
            gate.get("schema") == SCHEMA and gate.get("plan_sha256") == plan_digest
            and gate.get("allocation_digest") == allocation.get("allocation_digest")
            and gate.get("code_sha256") == code_sources(self.root)
        )
        if not gate_valid:
            self._fresh_progress(f"key-allocation-kl-{model}.json")
            run_step("gate")
        self.state["downstream_status"] = "await_frozen_key_allocation_v_plan"

    def _expansion(self, model: str) -> None:
        from experiments.static_layer_allocation.code.expand import _condition
        from experiments.static_layer_allocation.code.select_final import result_file_name

        folder = (self.root / "research/phase1/experiments/static_layer_allocation/results"
                  / "expanded_quality" / model)
        self._run(f"{ALLOCATION}.select_final", model, "64", self.commit)
        while True:
            plan = _read(folder / "selection.json")["next_expansion"]
            if not plan:
                break
            totals = {entry["next_total_n"] for entry in plan.values()}
            if len(totals) != 1:
                raise RuntimeError("扩样计划存在不相同的轮次")
            total = totals.pop()
            for task, entry in plan.items():
                for condition in entry["conditions"]:
                    config, residual_h, source = _condition(self.root, model, condition)
                    digest = hashlib.sha256(json.dumps({
                        "name": condition, "source": source,
                        "config": config, "residual_h": residual_h,
                    }, sort_keys=True).encode("utf-8")).hexdigest()
                    filename = result_file_name(task, condition, total)
                    path = folder / filename
                    if path.exists() and _read(path)["config_digest"] == digest:
                        continue
                    label = filename[len(task) + 1:].rsplit("_n", 1)[0]
                    self._fresh_progress(f"expanded-quality-{model}-{task}-{label}.json",
                                         f"f-{model}-{task}-{condition}.json")
                    self._run(f"{ALLOCATION}.expand", model, task, condition,
                              str(total - 64), self.commit)
            self._run(f"{ALLOCATION}.select_final", model, str(total), self.commit)

    def run_model(self, model: str) -> None:
        """补齐生成候选后保留其选择，静态 Key 分配按独立三格式计划运行。"""
        folder = self._folder("generation_error_sources", model)
        matrix = _read(self._folder("representation_interactions", model) / "summary.json")
        for name in generation_candidates(matrix):
            if (folder / f"v_gsm8k_{name}_early_stop.json").exists():
                continue
            for task in TASKS:
                self._generation(model, task, name)
        self._run(f"{GENERATION}.select", model, self.commit)
        selected = _read(folder / "no_residual_selection.json")["selected_low_bit"]
        residual_path = folder / "residual_selection.json"
        joint = residual_path.exists() and _read(residual_path).get("selection_policy") == (
            "joint_measured_format_residual"
        )
        if selected and not joint:
            for h in (8, 16):
                for task in TASKS:
                    self._generation(model, task, selected, h)
        self._run(f"{GENERATION}.select_residual", model, self.commit)
        selection = _read(folder / "residual_selection.json")
        self._diagnostics(model, selection)
        self._layers(model, selection)
        from experiments.static_layer_allocation.code.protocol import load_plan

        if load_plan(self.root)["v_plan"]["frozen"]:
            raise RuntimeError("新 Key 分配的 V 入口需按另行冻结的方案实现后再衔接")
        self.state["downstream_status"] = "await_frozen_key_allocation_v_plan"

    def run(self, models: list[str]) -> None:
        """核对离线校准实体后运行，保存可回传的小型执行索引。"""
        calibration = _read(
            self.root / "research/phase1/experiments/generation_error_sources/results"
            / "calibration.json"
        )
        for model in models:
            entry = calibration["files"][model]
            if _sha256(self.root / entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"全部 C 的冻结校准校验失败：{model}")
        try:
            for model in models:
                self.run_model(model)
        except Exception as error:
            self.state.update(status="failed", error=str(error))
            self._save()
            raise
        if self.state.get("error"):
            self.state.setdefault("recovered_errors", []).append(self.state.pop("error"))
        self.state.update(status="completed", current_step=None, models=models)
        results = {}
        for experiment in ("generation_error_sources", "static_layer_allocation",
                           "storage_service_cost"):
            for path in (self.root / "research/phase1/experiments" / experiment
                         / "results").rglob("*.json"):
                results[path.relative_to(self.root).as_posix()] = _sha256(path)
        self.state["result_sha256"] = results
        self.state["reports_pending_review"] = True
        self._save()
        output = self.root / "research/phase1/experiments/generation_error_sources/results"
        (output / "completion_execution.json").write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )


def main() -> None:
    """默认按 Qwen、Llama 顺序补齐；单模型参数用于中断恢复。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--models", nargs="+", choices=("qwen", "llama"),
                        default=["qwen", "llama"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--residual-diagnostics", action="store_true",
                      help="独立补测三种表示的残留，不重新选择或执行下游实验")
    mode.add_argument("--joint-selection", action="store_true",
                      help="联合排名已测格式加 H，只更新选择，之后审核下游复用范围")
    args = parser.parse_args()
    if args.residual_diagnostics:
        from experiments.generation_error_sources.code.residual_diagnostics import run

        run(args.commit, args.models)
    elif args.joint_selection:
        from experiments.generation_error_sources.code.select_residual import run

        for model in args.models:
            run(model, args.commit, joint=True)
    else:
        ExperimentCompletion(args.commit).run(args.models)


if __name__ == "__main__":
    main()
