"""按当前协议恢复阶段实验，默认完成冻结静态配置的独立生成。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from kvstudy.artifacts import read_json as _read, file_hash as _sha256
from experiments.generation_error_sources.code.select import KV8, TASKS
from kvstudy.paths import project_root

GENERATION = "experiments.generation_error_sources.code"


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
                from phase1.quality_stats import _paired

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

    def run(self, models: list[str]) -> None:
        """阻止已退出的旧全阶段补测流程重新产生未报告结果。"""
        raise RuntimeError("旧全阶段补测流程已退出，请使用 phase1.entry")


def main() -> None:
    """保留静态独立生成、残留诊断和联合选择三个正式入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--models", nargs="+", choices=("qwen", "llama"),
                        default=["qwen", "llama"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--static-final-quality", action="store_true",
                      help="只恢复已冻结静态三候选的独立生成与必要扩样，不重跑 C")
    mode.add_argument("--residual-diagnostics", action="store_true",
                      help="独立补测三种表示的残留，不重新选择或执行下游实验")
    mode.add_argument("--joint-selection", action="store_true",
                      help="联合排名已测格式加 H，只更新选择，之后审核下游复用范围")
    args = parser.parse_args()
    from experiments.static_layer_allocation.code.protocol import load_plan

    static_quality = args.static_final_quality
    if not args.residual_diagnostics and not args.joint_selection:
        static_quality |= load_plan(project_root())["v_plan"].get("protocol") == (
            "static-allocation-independent-generation"
        )
    if static_quality:
        from phase1.execution import CURRENT

        root = project_root()
        if (root / CURRENT).exists() and not (root / ".execution.json").exists():
            subprocess.run([
                sys.executable, "-m", "phase1.entry", "run", "--experiment",
                "static_layer_allocation", "--commit", args.commit, "--models", *args.models,
            ], cwd=root, check=True)
            return
        import torch
        from experiments.static_layer_allocation.code.protocol import require_v_plan
        from experiments.static_layer_allocation.code.quality import run

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        root = project_root()
        require_v_plan(root)
        for model in args.models:
            run(root, model)
    elif args.residual_diagnostics:
        from experiments.generation_error_sources.code.residual_diagnostics import run

        run(args.commit, args.models)
    elif args.joint_selection:
        from experiments.generation_error_sources.code.select_residual import run

        for model in args.models:
            run(model, args.commit)
    else:
        raise RuntimeError("旧全阶段补测流程已退出，请使用 phase1.entry 指定实验和缺测条件")

if __name__ == "__main__":
    main()
