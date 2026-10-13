"""只从已审核的两模型传播结果生成报告图。"""

from __future__ import annotations

import hashlib
import json
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from kvstudy.paths import project_root


matplotlib.use("Agg")


# 与冻结回放病例顺序一致。显示用的生成 token 从 1 计数，等于 first_divergence_index + 1。
QUESTIONS = (
    ("5ae7fd085542993210984007", "美国总统"),
    ("5ac1ff2255429964131be2b3", "郊区人口"),
    ("5a7ef49a5542994959419a91", "续订日期"),
    ("5a738b855542992d56e7e363", "哪家医院更早"),
    ("gsm8k-train-2131", "直角三角形"),
    ("gsm8k-train-2684", "州集市收入"),
    ("gsm8k-train-2887", "Cara 的年龄"),
    ("gsm8k-train-1", "保姆报酬"),
)


def _first_divergence(experiment, destination, labels: tuple[str, ...],
                     colors: tuple[str, ...]) -> None:
    """绘制八题上首次 argmax 分歧的生成步；空心点表示参考轨迹结束时仍一致。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), constrained_layout=True)
    offsets = (-0.22, 0.0, 0.22)
    for axis, model in zip(axes, ("qwen", "llama")):
        path = experiment / f"results/{model}/trajectories.json"
        payload = json.loads(path.read_text("utf-8"))
        rows = {row["id"]: row for row in payload["rows"]}
        if [row["id"] for row in payload["rows"]] != [item[0] for item in QUESTIONS]:
            raise RuntimeError("病例顺序与报告图不一致")
        for index, (question_id, _name) in enumerate(QUESTIONS):
            row = rows[question_id]
            end = row["bf16_new_tokens"]
            for candidate, color, offset in zip(row["candidates"], colors, offsets):
                y = index + offset
                first = candidate["first_divergence_index"]
                if first is None:
                    axis.plot([1, end], [y, y], color=color, linewidth=1.3, alpha=0.35)
                    axis.scatter([end], [y], s=22, facecolors="white", edgecolors=color,
                                 linewidths=1.2, zorder=3)
                else:
                    token = first + 1
                    axis.plot([1, token], [y, y], color=color, linewidth=1.5)
                    axis.scatter([token], [y], s=26, color=color, zorder=3)
        axis.set_xscale("log")
        axis.set_xlim(0.85, 700)
        axis.set_yticks(range(len(QUESTIONS)))
        axis.set_yticklabels([name for _question_id, name in QUESTIONS])
        axis.set_ylim(len(QUESTIONS) - 0.45, -0.55)
        axis.set(title=model.capitalize(), xlabel="生成 token（从 1 计）")
        axis.grid(axis="x", which="major", alpha=0.25)
    handles = [
        Line2D([0], [0], color=color, marker="o", markerfacecolor=color,
               markersize=5, label=label)
        for label, color in zip(labels, colors)
    ]
    handles.append(Line2D([0], [0], color="#666666", marker="o", markerfacecolor="white",
                          markersize=5, label="结束时仍一致"))
    fig.legend(handles, [handle.get_label() for handle in handles], loc="outside upper center",
               ncol=4, frameon=False, fontsize=9)
    fig.savefig(destination / "propagation_steps.png")
    plt.close(fig)


def run() -> None:
    """验证审核实体仍对应当前结果，再绘制层误差与首次分歧位置。"""
    root = project_root()
    experiment = root / "research/phase1/experiments/generation_error_sources"
    review = json.loads((experiment / "results/trajectory_review.json").read_text("utf-8"))
    if not review["reviewed"] or not review["complete"]:
        raise RuntimeError("两模型尚未完成审核，不能生成报告图")
    for source in review["sources"]:
        if hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("结果在审核后发生变化")
    labels = ("Online pre, H=0", "Online post, H=0", "Offline pre, H=8")
    colors = ("#2563a6", "#da7c24", "#29845a")
    plt.rcParams.update({
        "font.size": 10,
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 180,
    })
    destination = experiment / "report"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True, sharey=True)
    for axis, model in zip(axes, ("qwen", "llama")):
        for condition, label, color in zip(review["models"][model]["conditions"], labels, colors):
            values = condition["mean_layer_hidden_rse"]
            axis.plot(range(1, len(values) + 1), values, label=label, color=color, linewidth=1.8)
        axis.set(title=model.capitalize(), xlabel="Transformer layer (1-based)", yscale="log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Layer-output relative squared error")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(destination / "propagation_layers.png")
    plt.close(fig)
    _first_divergence(
        experiment, destination, ("在线 pre，H=0", "在线 post，H=0", "离线 pre，H=8"), colors,
    )


if __name__ == "__main__":
    run()
