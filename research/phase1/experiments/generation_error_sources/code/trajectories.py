"""三个开发候选在两模型上的跨层、跨步传播配对实验。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.generation_error_sources.code.trajectory_replay import candidate_run, reference_run
from kvstudy.paths import project_root
from phase1.generation import MAX_CONTEXT, _calibration, _config, _generate
from phase1.inputs import _inputs
from phase1.models import _eos_ids
from phase1.representation import MODELS, _render_gsm
from phase1.runner import load_model
from phase1.score import gsm8k_exact, token_f1


def _input_ids(
    tokenizer: Any, model_key: str, item: dict[str, Any]
) -> torch.Tensor:
    """把预登记题目恢复成 CUDA 上的输入 token，并核对冻结全文与长度。"""
    if item["kind"] == "math":
        return _render_gsm(tokenizer, item["question"], model_key == "llama")
    if "text" in item:
        source = item["text"]
    else:
        source = item["path"].read_text("utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != item["sha256"]:
            raise RuntimeError(f"快照全文校验失败：{item['id']}")
    ids = tokenizer(source, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    if ids.shape[1] != item["tokens"]:
        raise RuntimeError(f"快照 token 长度不符：{item['id']}")
    return ids


def _input_hash(
    root: Path, tokenizer: Any, model_key: str, item: dict[str, Any]
) -> str:
    """计算实际送入 tokenizer 的完整题面 SHA256。"""
    if item["kind"] == "math":
        template = json.loads(
            (root / "research/phase1/splits/templates.json").read_text("utf-8")
        )["gsm8k"]
        user = template.format(question=item["question"])
        kwargs = {"date_string": "26 Jul 2024"} if model_key == "llama" else {}
        source = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, **kwargs,
        )
    else:
        source = item.get("text") or item["path"].read_text("utf-8")
        expected = item.get("sha256")
        if expected and hashlib.sha256(source.encode("utf-8")).hexdigest() != expected:
            raise RuntimeError(f"轨迹全文校验失败：{item['id']}")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text("utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _compact(payload: dict[str, Any], progress: Path, root: Path) -> dict[str, Any]:
    """完整预测与层步矩阵留在服务器，回传逐步标量及层汇总。"""
    result = copy.deepcopy(payload)
    result["raw_source"] = {"path": progress.relative_to(root).as_posix(),
                            "sha256": _sha(progress)}
    for row in result["rows"]:
        row["bf16_new_tokens"] = len(row.pop("bf16_tokens"))
        row.pop("bf16_text")
        for candidate in row["candidates"]:
            free = candidate["paired_free_generation"]
            free["new_tokens"] = len(free.pop("token_ids"))
            free.pop("text")
            steps = candidate["steps"]
            if not steps:
                raise RuntimeError("参考首 token 已终止，缺少可诊断 decode 步")
            candidate["layer_mean_rse"] = [
                {name: sum(step["layers"][layer][name] for step in steps) / len(steps)
                 for name in ("hidden",)}
                for layer in range(len(steps[0]["layers"]))
            ]
            for step in steps:
                step["final_hidden_rse"] = step["layers"][-1]["hidden"]
                step.pop("layers")
    return result


def run(model_key: str, commit: str) -> None:
    """按冻结病例运行，逐题持久保存；正式文件只在全部病例完成后替换。"""
    root = project_root()
    plan_path = root / "research/phase1/configs/trajectory_replay.json"
    plan = _read(plan_path)
    for source in plan["sources"]:
        if _sha(root / source["path"]) != source["sha256"]:
            raise RuntimeError(f"冻结质量来源变化：{source['path']}")
    publication = root / "research/.runtime/trajectory-publication.json"
    provenance = _read(publication)
    for path, digest in provenance["files"].items():
        if _sha(root / path) != digest:
            raise RuntimeError(f"执行源码与发布清单不一致：{path}")
    folder = root / "research/phase1/experiments/generation_error_sources/results" / model_key
    progress = root / f"research/.runtime/generation-propagation-{model_key}.json"
    identity = {"model": model_key, "commit": commit, "plan_sha256": _sha(plan_path),
                "publication_sha256": _sha(publication)}
    payload = _read(progress) if progress.exists() else {
        **identity, "schema": "kv-propagation-v1", "rows": [], "reviewed": False,
    }
    if any(payload[k] != value for k, value in identity.items()):
        raise RuntimeError("断点来源与当前执行不一致")
    if [row["id"] for row in payload["rows"]] != [
            case["id"] for case in plan["cases"][:len(payload["rows"])]]:
        raise RuntimeError("断点题目顺序不一致")
    tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model_key])
    torch.set_num_threads(4)
    model = load_model(str(root / MODELS[model_key]))
    eos = _eos_ids(tokenizer, model_key == "llama")
    items = {task: {row["id"]: row for row in _inputs(root, model_key, task, "v")}
             for task in ("gsm8k", "hotpot")}
    with torch.inference_mode():
        for case in plan["cases"][len(payload["rows"]):]:
            started = time.monotonic()
            task = case["task"]
            item = {**items[task][case["id"]], "kind": "math" if task == "gsm8k" else task}
            ids = _input_ids(tokenizer, model_key, item)
            digest = _input_hash(root, tokenizer, model_key, item)
            if ids.shape[1] + plan["max_new_tokens"] > MAX_CONTEXT:
                raise RuntimeError("诊断超出冻结上下文长度")
            quality = {}
            for condition in [{"name": "bf16", "h": 0}, *plan["conditions"]]:
                suffix = f"-residual{condition['h']}" if condition["h"] else ""
                name = condition["name"] + suffix
                source = folder / f"v_{task}_{name}.json"
                row = next(row for row in _read(source)["rows"] if row["id"] == item["id"])
                if row["input_sha256"] != digest or row["input_tokens"] != ids.shape[1]:
                    raise RuntimeError("已有自由质量与回放实际输入不一致")
                quality[name] = {"score": 100 * int(row["correct"]) if task == "gsm8k"
                                else row["f1"], "new_tokens": row["new_tokens"],
                                "source": source.relative_to(root).as_posix(),
                                "sha256": _sha(source)}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            offload = 8 if model_key == "llama" else 0
            tokens, stop, references = reference_run(
                model, ids, eos, plan["max_new_tokens"], consume_cache=bool(offload),
            )
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            score = (100 * gsm8k_exact(text, item["answer"]) if task == "gsm8k"
                     else token_f1(text, item["answer"]))
            historical_match = (abs(score - quality["bf16"]["score"]) <= 1e-8
                                and len(tokens) == quality["bf16"]["new_tokens"])
            candidates = []
            for condition in plan["conditions"]:
                config = _config(condition["name"])
                calibrated = _calibration(root, model_key, config)
                measured = candidate_run(
                    model, ids, tokens, references, config, calibrated, condition["h"],
                    plan["max_new_tokens"],
                    offload_layers=offload,
                )
                free_tokens, free_stop, invalid = _generate(
                    model, ids[0].tolist(), eos, plan["max_new_tokens"], config,
                    calibrated, condition["h"],
                    offload_layers=offload,
                )
                if invalid:
                    raise RuntimeError("同次自由生成出现无效组")
                free_text = tokenizer.decode(free_tokens, skip_special_tokens=True)
                free_score = (100 * gsm8k_exact(free_text, item["answer"])
                              if task == "gsm8k" else token_f1(free_text, item["answer"]))
                divergence = next((i for i, (a, b) in enumerate(zip(tokens, free_tokens))
                                   if a != b), None)
                if divergence is None and len(tokens) != len(free_tokens):
                    divergence = min(len(tokens), len(free_tokens))
                if divergence != measured["first_divergence_index"]:
                    raise RuntimeError("自由生成与强制回放的首次分歧不一致")
                measured["paired_free_generation"] = {
                    "token_ids": free_tokens, "text": free_text, "stop": free_stop,
                    "score": free_score, "bf16_score": score, "delta": free_score - score,
                }
                candidates.append(measured)
                print(model_key, task, item["id"], condition["name"],
                      measured["first_divergence_index"], flush=True)
                del calibrated
                torch.cuda.empty_cache()
            payload["rows"].append({
                "id": item["id"], "task": task, "role": case["role"],
                "input_tokens": int(ids.shape[1]), "input_sha256": digest,
                "bf16_tokens": tokens, "bf16_text": text, "bf16_stop": stop,
                "bf16_score": score, "historical_bf16_score_length_match": historical_match,
                "free_quality": quality, "candidates": candidates,
                "device_name": torch.cuda.get_device_name(),
                "cache_storage": f"cpu_first_{offload}_layers" if offload else "gpu",
                "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "elapsed_s": time.monotonic() - started,
            })
            _write(progress, payload)
            print("COMPLETED", model_key, len(payload["rows"]), "peak_GiB",
                  torch.cuda.max_memory_reserved() / 2 ** 30, flush=True)
            del references, ids
            torch.cuda.empty_cache()
    payload["n"] = len(payload["rows"])
    _write(progress, payload)
    _write(folder / "trajectories.json", _compact(payload, progress, root))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
