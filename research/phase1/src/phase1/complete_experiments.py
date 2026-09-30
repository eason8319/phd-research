"""补齐共同候选，并按冻结协议依次复核残留、层分配、扩样和成本。"""

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
    TASKS, generation_candidates,
)
from experiments.generation_error_sources.code.select import KV8, _paired, _read
from experiments.static_layer_allocation.code.expand import _condition
from experiments.static_layer_allocation.code.select_final import result_file_name

GENERATION = "experiments.generation_error_sources.code"
ALLOCATION = "experiments.static_layer_allocation.code"
COST = "experiments.storage_service_cost.code"


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
            ("cross_token_groups", "cross_token_gt32.json", "generation-cross-token",
             "e2-cross-token"),
            ("trajectories", "trajectories.json", "generation-trajectories",
             "e2-trajectories"),
        ):
            chosen = name or "kv4-per_token-post"
            path = folder / filename
            if path.exists() and _read(path)["representation"] == chosen:
                continue
            if module == "trajectories" and name is None:
                continue
            self._fresh_progress(f"{current}-{model}.json", f"{legacy}-{model}.json")
            self._run(f"{GENERATION}.{module}", model, self.commit)
        if name is None:
            return
        if "per_channel" in name:
            for module in ("scale_drift", "range_refit"):
                if not (folder / f"{module}_{name}.json").exists():
                    prefix = module.replace("_", "-")
                    self._fresh_progress(f"generation-{prefix}-{model}-{name}.json",
                                         f"e2-{prefix}-{model}-{name}.json")
                    self._run(f"{GENERATION}.{module}", model, name, self.commit)
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
        folder = self._folder("static_layer_allocation", model)
        name = selection["e3_representation"]
        h = selection["e3_residual_bits"]
        allocations = folder / "allocations.json"
        reuse = allocations.exists() and (
            _read(allocations)["representation"] == name
            and _read(allocations)["residual_bits"] == h
        )
        if not reuse:
            self._run(f"{ALLOCATION}.probe", model, name, self.commit)
            self._run(f"{ALLOCATION}.allocations", model, self.commit)
            self._fresh_progress(f"layer_allocation-gate-{model}.json")
            self._run(f"{ALLOCATION}.gate", model, self.commit)
        gate = _read(folder / "gate.json")
        allocation = _read(allocations)
        for task in TASKS:
            for allocation_name in gate["selected_nonrandom_for_v"] + gate["random_for_v"]:
                path = folder / f"v_{task}_{allocation_name}.json"
                old = _read(path) if path.exists() else {}
                if (reuse and old.get("allocation_commit") == allocation["commit"]
                        and old.get("gate_commit") == gate["commit"]):
                    continue
                self._fresh_progress(f"layer_allocation-{model}-{task}-v-"
                                     f"{allocation_name}.json")
                self._run(f"{ALLOCATION}.quality", model, task, allocation_name, self.commit)
        self._run(f"{ALLOCATION}.select", model, self.commit)

    def _expansion(self, model: str) -> None:
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
        """四候选全部补齐后才重选；改变的表示继续重测全部依赖步骤。"""
        folder = self._folder("generation_error_sources", model)
        matrix = _read(self._folder("representation_interactions", model) / "summary.json")
        for name in generation_candidates(matrix):
            if (folder / f"v_gsm8k_{name}_early_stop.json").exists():
                continue
            for task in TASKS:
                self._generation(model, task, name)
        self._run(f"{GENERATION}.select", model, self.commit)
        selected = _read(folder / "no_residual_selection.json")["selected_low_bit"]
        if selected:
            for h in (8, 16):
                for task in TASKS:
                    self._generation(model, task, selected, h)
        self._run(f"{GENERATION}.select_residual", model, self.commit)
        selection = _read(folder / "residual_selection.json")
        self._diagnostics(model, selection)
        self._layers(model, selection)
        self._expansion(model)
        self._run(f"{COST}.layout", model, self.commit)

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
    args = parser.parse_args()
    ExperimentCompletion(args.commit).run(args.models)


if __name__ == "__main__":
    main()
