"""只从两模型已审核的 Key 分配结果生成层分布与预算趋势图。"""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kvstudy.paths import project_root

FORMATS = {
    "online_pre": "Online pre, H=0",
    "online_post": "Online post, H=0",
    "offline_pre": "Offline pre, H=8, R=128",
}
COLORS = ("#c13b3a", "#2878a0", "#606870", "#368864", "#9a67b0", "#c69735")


def _read(path):
    return json.loads(path.read_text("utf-8"))


def _verify(experiment) -> None:
    review = _read(experiment / "results/review.json")
    if review["status"] != "passed":
        raise RuntimeError("结果未审核，不能作图")
    root = project_root()
    for path, checksum in review["source_files"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != checksum:
            raise RuntimeError("审核后结果变化，不能复用审核作图")


def _layer_cost(experiment) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7))
    for row, model in enumerate(("qwen", "llama")):
        probe = _read(experiment / f"results/{model}/probe.json")
        for column, (name, label) in enumerate(FORMATS.items()):
            ax = axes[row, column]
            spec = probe["formats"][name]
            for kind, color, marker in (("math", "#2878a0", "o"),
                                        ("retrieval", "#d77b24", "s")):
                values = [item["sensitivity"] for item in spec["layers"][kind]]
                ax.plot(range(len(values)), values, color=color, marker=marker,
                        markersize=2.8, linewidth=1.3,
                        label="Math (64)" if kind == "math" else "Retrieval (32)")
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_title(f"{model.title()} | {label}", fontsize=10)
            ax.set_xlabel("Layer index (0-based)")
            ax.set_ylabel("Mean relative squared change")
            ax.set_xticks(range(0, len(values), 5))
            ax.grid(axis="y", alpha=0.22)
            for i in spec["rankings_descending"]["retrieval"]["sensitivity"][:2]:
                value = spec["layers"]["retrieval"][i]["sensitivity"]
                ax.annotate(str(i), (i, value), xytext=(4, 5), textcoords="offset points",
                            fontsize=8, color="#a45817")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Layer cost of K8V4 to K4V4", fontsize=14, y=0.997)
    fig.text(0.5, 0.008,
             "Source: probe.json, reviewed full C | Common high-precision output denominator"
             " | Equal weight per question",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.9))
    fig.savefig(experiment / "report/key_downgrade_layers.png", dpi=200)
    plt.close(fig)


def _budget_kl(experiment) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7))
    for row, model in enumerate(("qwen", "llama")):
        allocation = _read(experiment / f"results/{model}/allocations.json")
        gate = _read(experiment / f"results/{model}/gate.json")
        means = gate["mean_next_token_kl"]["retrieval"]
        seeds = allocation["random_seeds"]
        strategies = ("sensitivity", "reconstruction", "inverse",
                      *(f"random_{seed}" for seed in seeds))
        labels = ("Sensitivity", "Reconstruction", "Reverse", "Random 1", "Random 2", "Random 3")
        for column, (name, label) in enumerate(FORMATS.items()):
            ax = axes[row, column]
            baseline = means[name + "_k8v4"]
            for strategy, legend, color in zip(strategies, labels, COLORS):
                points = gate["budget_trends"][name][strategy]
                x = [0] + [point["r"] for point in points]
                y = [baseline] + [point["mean_kl"]["retrieval"] for point in points]
                ax.plot(x, y, label=legend, color=color,
                        marker="o" if strategy == "sensitivity" else ".",
                        markersize=4 if strategy == "sensitivity" else 3,
                        linewidth=2 if strategy == "sensitivity" else 1.1,
                        linestyle="-" if strategy == "sensitivity" else "--", alpha=0.95)
            ax.axhline(means["kv8"], color="#26313c", linestyle=":", linewidth=1,
                       label="KV8 reference")
            if all(value > 0 for value in y) and means["kv8"] > 0:
                ax.set_yscale("log")
            else:
                ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_title(f"{model.title()} | {label}", fontsize=10)
            ax.set_xlabel("Number of K4 layers")
            ax.set_ylabel("Mean KL(BF16 || candidate), nat")
            ax.set_xticks(x)
            ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.96), fontsize=9)
    fig.suptitle("Full-layer distribution change across frozen budgets", fontsize=14, y=0.997)
    fig.text(0.5, 0.008,
             "Source: gate.json, reviewed full C | Retrieval: 32 questions"
             " | Common BF16 first token, one decode | r=0: same-format K8V4",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.87))
    fig.savefig(experiment / "report/key_budget_kl.png", dpi=200)
    plt.close(fig)


def run() -> None:
    """复核审核文件校验后，在现有报告目录保存两张正式图。"""
    experiment = project_root() / "research/phase1/experiments/static_layer_allocation"
    _verify(experiment)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    _layer_cost(experiment)
    _budget_kl(experiment)


if __name__ == "__main__":
    run()

