"""第一阶段统一的结果盘点、复用计划、执行和审核入口。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from kvstudy.paths import project_root
from phase1.reuse import GENERATION_RESULTS, POLICY, file_hash
from phase1.results import EXPERIMENTS, ResultCatalog


def inventory(root: Path) -> dict[str, Any]:
    """只读盘点唯一结果与公共参考，不加载模型或安排重跑。"""
    catalog = ResultCatalog(root)
    records = {name: catalog.inventory(name) for name in EXPERIMENTS}
    return {"experiments": records, "reference_owner": GENERATION_RESULTS,
            "policy": POLICY,
            "reuse": {name: catalog.decision(name)["action"] for name in EXPERIMENTS}}


def plan(
    root: Path, experiment: str, models: list[str], task: str | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    """已有结果先复用或核验；计划本身不改变文件和运行状态。"""
    if bool(task) != bool(condition):
        raise ValueError("单条件计划须同时指定任务和格式")
    if task and experiment != "generation_error_sources":
        raise ValueError("任务与格式参数只用于真实生成的单条件入口")
    if experiment == "static_layer_allocation":
        from experiments.static_layer_allocation.code.quality import execution_plan

        return {model: execution_plan(root, model) for model in models}
    if experiment == "generation_error_sources" and task and condition:
        catalog = ResultCatalog(root)
        records = {}
        for model in models:
            path = root / GENERATION_RESULTS / model / f"v_{task}_{condition}.json"
            relative = path.relative_to(root).as_posix()
            records[model] = {**catalog.decision(experiment, relative), "source": relative}
        return {"experiment": experiment, "task": task, "condition": condition,
                "cells": records}
    return {"experiment": experiment, **ResultCatalog(root).decision(experiment)}


def _published_run(args: argparse.Namespace, root: Path) -> None:
    """从当前已核对快照启动，维护树的后续编辑不改变本次运行。"""
    from phase1.execution import CURRENT, verify_publication

    current = root / CURRENT
    if not (root / ".execution.json").exists():
        if not current.exists():
            raise RuntimeError("先发布当前代码再执行；不能使用未核对的服务器维护树")
        publication = json.loads(current.read_text("utf-8"))
        source = verify_publication(root, publication)
        if any(file_hash(root / name) != expected
               for name, expected in publication["files"].items()):
            raise RuntimeError("维护树已变化，先重新发布，保留原执行快照")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(str(source / part) for part in (
            "research/src", "research/phase1/src", "research/phase1",
        ))
        subprocess.run([sys.executable, "-m", "phase1.entry", *sys.argv[1:]],
                       cwd=source, env=environment, check=True)
        return
    verify_publication(root, json.loads((root / ".execution.json").read_text("utf-8")))
    if not args.task and not args.condition:
        decision = ResultCatalog(root).decision(args.experiment)
        if decision["action"] == "reuse":
            print(json.dumps({"experiment": args.experiment, **decision},
                             ensure_ascii=False, indent=2), flush=True)
            return
        if decision["action"] == "review":
            raise RuntimeError(f"已有结果先审核，不自动重跑：{decision['reason']}")
    elif args.experiment == "generation_error_sources":
        proposal = plan(root, args.experiment, args.models, args.task, args.condition)
        if all(item["action"] == "reuse" for item in proposal["cells"].values()):
            print(json.dumps(proposal, ensure_ascii=False, indent=2), flush=True)
            return
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.experiment == "static_layer_allocation":
        from experiments.static_layer_allocation.code.quality import run

        for model in args.models:
            run(root, model)
        return
    if args.experiment == "generation_error_sources" and args.task and args.condition:
        proposal = plan(root, args.experiment, args.models, args.task, args.condition)
        for model, decision in proposal["cells"].items():
            if decision["action"] == "reuse":
                continue
            if decision["action"] == "review":
                raise RuntimeError(f"已有结果缺少当前复用核验，不重新生成：{decision['source']}")
            if args.condition == "bf16":
                if args.task == "gsm8k":
                    from experiments.generation_error_sources.code.short_baseline import run

                    run(model, args.commit)
                else:
                    raise RuntimeError("长任务 BF16 须先按指导冻结输入并审核对应基线")
            else:
                from experiments.generation_error_sources.code.quantized_generation import run

                run(model, args.task, "v", args.condition, args.commit, save_predictions=True,
                    execution_source=json.loads((root / ".execution.json").read_text("utf-8")))
        print(json.dumps(proposal, indent=2, ensure_ascii=False), flush=True)
        return
    proposal = plan(root, args.experiment, args.models)
    if proposal["action"] == "review":
        raise RuntimeError("已有结果先核验或显式指定缺测条件，不自动重跑整项实验")
    if args.experiment == "storage_service_cost":
        from experiments.storage_service_cost.code.layout import run

        for model in args.models:
            run(model, args.commit)
        return
    if args.experiment == "representation_interactions":
        from experiments.representation_interactions.code.evaluate import run_model

        for model in args.models:
            run_model(model, args.commit)
        return
    raise RuntimeError("先在指导中确定本次缺测条件，再调用该实验的条件入口")


def main() -> None:
    """统一阶段入口；默认盘点，生成只由显式 run 动作启动。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="status",
                        choices=("status", "plan", "publish", "run", "review"))
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"))
    parser.add_argument("--models", nargs="+", choices=("qwen", "llama"),
                        default=["qwen", "llama"])
    parser.add_argument("--commit")
    parser.add_argument("--task", choices=("gsm8k", "hotpot", "retrieval_ruler", "hotpot_four"))
    parser.add_argument("--condition")
    args = parser.parse_args()
    if bool(args.task) != bool(args.condition):
        parser.error("须同时指定 --task 和 --condition")
    root = project_root()
    if args.action == "status":
        result = inventory(root)
    elif args.action == "publish":
        if not args.commit:
            parser.error("publish 需要实际基准 --commit，文件校验另行登记")
        from phase1.execution import publish

        result = publish(root, args.commit)
    else:
        if not args.experiment:
            parser.error("该动作需要 --experiment")
        if args.experiment == "all" and args.action != "review":
            parser.error("all 仅用于只读审核；实验运行仍按各实验与缺测条件执行")
        if args.action == "plan":
            result = plan(root, args.experiment, args.models, args.task, args.condition)
        elif args.action == "run":
            if not args.commit:
                parser.error("run 需要实际基准 --commit")
            _published_run(args, root)
            return
        else:
            from phase1.audit import audit

            result = audit(root, None if args.experiment == "all" else args.experiment)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
