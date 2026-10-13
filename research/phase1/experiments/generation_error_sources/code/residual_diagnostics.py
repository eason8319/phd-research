"""独立补齐近期残留诊断，复用逐题测量并保护当前下游选择。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from experiments.generation_error_sources.code.select import KV8
from experiments.generation_error_sources.code.select_residual import TASKS, _cell
from phase1.complete_experiments import ExperimentCompletion, _sha256
from phase1.quality_stats import _paired, _read
from phase1.review_results import _quality_plan


FORMATS = ("k8v4-per_channel-pre", "k8v4-per_token-pre", "k8v4-per_token-post")
SELECTED = {"qwen": "k8v4-per_token-pre", "llama": "k8v4-per_token-post"}
RESULTS = Path("research/phase1/experiments/generation_error_sources/results")


def paired_gain(
    baseline: dict[str, Any], candidate: dict[str, Any], task: str,
    seed: int, draws: int,
) -> dict[str, Any]:
    """把共同下降区间改写为 H−H=0 的增益，正值表示改善。"""
    drop = _paired(baseline, candidate, task, seed, draws)
    return {
        **drop, "estimate": -drop["estimate"],
        "lower": -drop["upper"] if drop["upper"] is not None else None,
        "upper": -drop["lower"] if drop["lower"] is not None else None,
        "direction": "candidate_minus_h0",
    }


def h8_decision(cells: dict[str, Any], existing: bool) -> dict[str, Any]:
    """只按预登记的类挽回或配对正向区间决定 H=8，已有结果始终保留。"""
    rescued = [task for task in TASKS if not cells[task]["0"]["class_unit_passes"]
               and cells[task]["16"]["class_unit_passes"]]
    supported = [task for task in TASKS
                 if cells[task]["16"]["gain_vs_h0"]["lower"] is not None
                 and cells[task]["16"]["gain_vs_h0"]["lower"] > 0]
    positive = [task for task in TASKS if cells[task]["16"]["gain_vs_h0"]["estimate"] > 0]
    trigger = bool(rescued or supported)
    return {
        "run": trigger and not existing, "existing_retained": existing,
        "rescued_by_h16": rescued, "positive_interval_tasks": supported,
        "positive_point_tasks": positive,
        "status": "复用" if existing else "触发补测" if trigger else "未补测",
        "reason": ("已有 H=8 测量照常保留" if existing else
                   "H=16 挽回任务类" if rescued else
                   "H=16 相对 H=0 的配对增益区间下界大于零" if supported else
                   "仅点估计改善，区间不足以支持；待确认" if positive else
                   "未挽回任务类且没有配对区间支持的正向改善"),
    }


class ResidualDiagnostics(ExperimentCompletion):
    """独立恢复状态只允许调用生成入口，不执行原补测的下游链。"""

    def __init__(self, commit: str) -> None:
        super().__init__(commit)
        self.path = self.runtime / f"residual-diagnostics-{commit}.json"
        self.state = _read(self.path) if self.path.exists() else {
            "commit": commit, "status": "pending_preflight", "completed_steps": [],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.folder = self.root / RESULTS
        self.output = self.folder / "residual_diagnostics.json"
        self.seeds = _read(self.root / "research/phase1/splits/seeds.json")

    def _measurement(self, model: str, task: str, name: str, h: int) -> Path:
        suffix = f"-residual{h}" if h else ""
        return self.folder / model / f"v_{task}_{name}{suffix}.json"

    def _sources(self) -> dict[str, str]:
        paths = []
        for relative in ("research/src", "research/phase1/src", "research/phase1/experiments"):
            paths.extend((self.root / relative).rglob("*.py"))
        for relative in ("research/phase1/configs", "research/phase1/splits"):
            paths.extend(path for path in (self.root / relative).rglob("*") if path.is_file())
        paths.append(self.root / "research/phase1/guide.md")
        return {path.relative_to(self.root).as_posix(): _sha256(path) for path in paths}

    def _check_protected(self) -> None:
        for relative, digest in self.state["protected_sha256"].items():
            if _sha256(self.root / relative) != digest:
                raise RuntimeError(f"既有结果或下游选择发生变化：{relative}")
        if self._sources() != self.state["code_config_sha256"]:
            raise RuntimeError("恢复期间执行代码、冻结配置或输入索引发生变化")

    def _check_calibration(self) -> None:
        calibration = _read(self.folder / "calibration.json")
        for model, entry in calibration["files"].items():
            if _sha256(self.root / entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"全部 C 校准校验失败：{model}")

    def _validate(self, model: str, task: str, name: str, h: int) -> dict[str, Any]:
        path = self._measurement(model, task, name, h)
        value = _read(path)
        if (value["model"] != model or value["config"] != name or value["task"] != f"v-{task}"
                or value["n"] != 64 or len(value["rows"]) != 64
                or value["residual_bits"] != h or value["residual_tokens"] != (128 if h else 0)):
            raise RuntimeError(f"配置或冻结样本数不一致：{path}")
        h0 = _read(self._measurement(model, task, name, 0))
        if value["calibration_commit"] != h0["calibration_commit"]:
            raise RuntimeError(f"残留与 H=0 校准来源不符：{path}")
        _paired(h0, value, task, self.seeds["bootstrap"], self.seeds["bootstrap_draws"])
        return value

    def preflight(self, models: list[str]) -> None:
        """在任何生成前核对完整复用集、冻结输入、校准和当前生效选择。"""
        self._check_calibration()
        if "protected_sha256" in self.state:
            if models != self.state["models"]:
                raise RuntimeError("恢复时模型范围不能改变")
            self._check_protected()
            return
        records = {}
        for model in models:
            selection = _read(self.folder / model / "residual_selection.json")
            if (selection["e3_representation"] != SELECTED[model]
                    or selection["e3_residual_bits"] != 0):
                raise RuntimeError("当前下游选择与本轮登记边界不符")
            baseline = _read(self.folder / model / "no_residual_selection.json")
            if any(baseline["baseline_usable"].get(task) is not True for task in TASKS):
                raise RuntimeError("有效任务边界发生变化")
            for task in TASKS:
                for name in ("bf16", KV8):
                    path = self.folder / model / f"v_{task}_{name}.json"
                    records[path] = _read(path)
                for name in FORMATS:
                    if baseline["cells"][task][name]["state"] == "淘汰":
                        raise RuntimeError("本轮表示已被淘汰")
                    for h in (0, 8, 16):
                        path = self._measurement(model, task, name, h)
                        if path.exists():
                            records[path] = self._validate(model, task, name, h)
                        elif h == 0 or name == SELECTED[model]:
                            raise RuntimeError(f"应复用的既有结果缺失：{path}")
        _quality_plan(self.root, records)
        protected = {
            path.relative_to(self.root).as_posix(): _sha256(path)
            for path in (self.root / "research/phase1/experiments").rglob("results/**/*.json")
            if path != self.output
        }
        self.state.update(
            models=models, protected_sha256=protected, code_config_sha256=self._sources(),
            reused_measurements=[path.relative_to(self.root).as_posix() for path in records],
            preregistration={
                "formats": FORMATS, "tasks": TASKS, "n": 64, "R": 128,
                "order": "all_missing_h16_then_conditional_h8",
                "h8_trigger": "rescued_class_or_paired_gain_lower_strictly_above_zero",
                "point_only": "pending_confirmation_no_expansion",
                "existing_h8": "retain_including_negative_results",
                "downstream": "unchanged_no_selection_or_rerun",
                "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            preflight={"frozen_inputs_verified": True, "calibration_sha256_verified": True},
            status="preflight_passed",
        )
        self._save()

    def summarize(self, model: str, name: str) -> dict[str, Any]:
        """复用统一类通过与区间算法；缺测 H=8 明示缺测，不归入不可行。"""
        cells = {}
        for task in TASKS:
            cells[task] = {}
            h0 = self._validate(model, task, name, 0)
            for h in (0, 16, 8):
                path = self._measurement(model, task, name, h)
                if not path.exists():
                    cells[task][str(h)] = {"measurement_status": "未测"}
                    continue
                value = self._validate(model, task, name, h)
                cell = _cell(self.folder / model, model, name, task, h,
                             self.seeds["bootstrap"], self.seeds["bootstrap_draws"], True)
                relative = path.relative_to(self.root).as_posix()
                cell.update(
                    measurement_status="复用" if relative in self.state["protected_sha256"]
                    else "新测", path=relative, sha256=_sha256(path),
                    execution_commit=value["commit"], n=value["n"],
                )
                if h:
                    cell["gain_vs_h0"] = paired_gain(
                        h0, value, task, self.seeds["bootstrap"], self.seeds["bootstrap_draws"],
                    )
                    cell["rescued_class"] = (
                        not cells[task]["0"]["class_unit_passes"] and cell["class_unit_passes"]
                    )
                    cell["lost_class_pass"] = (
                        cells[task]["0"]["class_unit_passes"] and not cell["class_unit_passes"]
                    )
                cells[task][str(h)] = cell
        return cells

    def run_diagnostics(self, models: list[str]) -> None:
        """所有 H=16 完成后登记 H=8 决策，审核后输出独立诊断汇总。"""
        self.preflight(models)
        try:
            for model in models:
                for name in FORMATS:
                    for task in TASKS:
                        self._generation(model, task, name, 16)
                        self._validate(model, task, name, 16)
            if "h8_decisions" not in self.state:
                decisions = {}
                for model in models:
                    decisions[model] = {}
                    for name in FORMATS:
                        existing = [self._measurement(model, task, name, 8).exists()
                                    for task in TASKS]
                        if any(existing) and not all(existing):
                            raise RuntimeError("H=8 既有覆盖不完整，需核查来源")
                        cells = self.summarize(model, name)
                        decisions[model][name] = h8_decision(cells, all(existing))
                        decisions[model][name]["h16_evidence"] = {
                            task: cells[task]["16"] for task in TASKS
                        }
                self.state["h8_decisions"] = decisions
                self._save()
            for model in models:
                for name in FORMATS:
                    if self.state["h8_decisions"][model][name]["run"]:
                        for task in TASKS:
                            self._generation(model, task, name, 8)
            summary = {model: {name: self.summarize(model, name) for name in FORMATS}
                       for model in models}
            records = {
                self.root / cell["path"]: _read(self.root / cell["path"])
                for names in summary.values() for tasks in names.values()
                for hs in tasks.values() for cell in hs.values() if "path" in cell
            }
            _quality_plan(self.root, records)
            self._check_calibration()
            self._check_protected()
            self.state.update(status="completed", current_step=None,
                              audit={"status": "passed", "frozen_inputs_verified": True,
                                     "calibration_sha256_verified": True,
                                     "protected_results_unchanged": True,
                                     "quality_files_verified": len(records)})
            self._save()
            payload = {
                **self.state, "cells": summary, "scope": "diagnostic_only",
                "active_selection": {model: {"representation": SELECTED[model], "H": 0}
                                     for model in models},
            }
            self.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                              allow_nan=False) + "\n", encoding="utf-8")
        except Exception as error:
            self.state.update(status="failed", error=str(error))
            self._save()
            raise


def run(commit: str, models: list[str]) -> None:
    """供阶段独立模式调用，保持原共同候选补测流程不变。"""
    ResidualDiagnostics(commit).run_diagnostics(models)


def run_candidate_supplement(commit: str, device_family: str = "ada48") -> None:
    """按明确授权仅补 Qwen 离线 pre H=8；不触发选择、回放或下游入口。"""
    from experiments.generation_error_sources.code.quantized_generation import run as generate

    diagnostic = ResidualDiagnostics(commit)
    diagnostic.path = diagnostic.runtime / "candidate-supplement.json"
    diagnostic.state = _read(diagnostic.path) if diagnostic.path.exists() else {
        "commit": commit, "status": "pending_preflight", "completed_steps": [],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    diagnostic.preflight(["qwen", "llama"])
    diagnostic.state["preregistration"] = {
        "models": ["qwen"], "formats": ["k8v4-per_channel-pre"], "H": 8, "R": 128,
        "tasks": TASKS, "n": 64, "order": "gsm8k_then_hotpot",
        "authorization": "用户明确要求补测，不依赖 H=16 是否触发",
        "downstream": "unchanged_no_selection_or_rerun", "fixed_replay": "禁止新增实验",
    }
    publication = _read(diagnostic.runtime / "candidate-publication.json")
    if publication["base_commit"] != commit:
        raise RuntimeError("发布来源与补测基准提交不符")
    for relative, digest in publication["files"].items():
        if _sha256(diagnostic.root / relative) != digest:
            raise RuntimeError(f"发布后的代码或冻结配置变化：{relative}")
    name = "k8v4-per_channel-pre"
    device_name, required_mib = {
        "ada48": ("RTX 6000 Ada", 24576), "rtx4090": ("RTX 4090", 23552),
    }[device_family]
    diagnostic.state["device_admission"] = {
        "family": device_family, "required_free_mib": required_mib,
        "reference": "research/phase1/configs/runtime.md",
    }
    missing = [task for task in TASKS
               if not diagnostic._measurement("qwen", task, name, 8).exists()]
    try:
        if missing:
            while True:
                diagnostic._check_protected()
                raw = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,name,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ], text=True)
                devices = [line.split(", ") for line in raw.strip().splitlines()]
                ready = [row for row in devices if device_name in row[1]
                         and int(row[2]) >= required_mib]
                diagnostic.state.update(status="waiting_for_gpu", resource_observation=raw)
                diagnostic._save()
                print(datetime.now(timezone.utc).isoformat(), raw.strip(), flush=True)
                if ready:
                    os.environ["CUDA_VISIBLE_DEVICES"] = max(ready, key=lambda row: int(row[2]))[0]
                    break
                time.sleep(60)
        for task in TASKS:
            diagnostic._check_protected()
            path = diagnostic._measurement("qwen", task, name, 8)
            if not path.exists():
                diagnostic.state.update(status="running", current_step=task)
                diagnostic._save()
                generate("qwen", task, "v", name, commit, 8, save_predictions=True,
                         guard_severe_failure=True, execution_source=publication)
            diagnostic._validate("qwen", task, name, 8)
        cells = diagnostic.summarize("qwen", name)
        records = {diagnostic._measurement("qwen", task, name, 8):
                   _read(diagnostic._measurement("qwen", task, name, 8)) for task in TASKS}
        _quality_plan(diagnostic.root, records)
        diagnostic._check_calibration()
        diagnostic._check_protected()
        diagnostic.state.pop("error", None)
        diagnostic.state.update(
            status="completed", current_step=None,
            authorization="2026-10-06 用户明确补测，不依赖 H=16 筛选；禁止固定回放实验",
            audit={"status": "passed", "frozen_inputs_verified": True,
                   "calibration_sha256_verified": True, "protected_results_unchanged": True},
        )
        diagnostic._save()
        summary = _read(diagnostic.output)
        for task in TASKS:
            summary["cells"]["qwen"][name][task]["8"] = cells[task]["8"]
        summary["candidate_supplement"] = diagnostic.state
        summary["candidate_supplement"]["execution_source"] = publication
        diagnostic.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                allow_nan=False) + "\n", encoding="utf-8")
    except Exception as error:
        diagnostic.state.update(status="failed", error=str(error))
        diagnostic._save()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-supplement", action="store_true", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--device-family", choices=("ada48", "rtx4090"), default="ada48")
    arguments = parser.parse_args()
    run_candidate_supplement(arguments.commit, arguments.device_family)
