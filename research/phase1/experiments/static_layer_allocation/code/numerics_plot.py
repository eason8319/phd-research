"""审核后生成当前报告使用的层敏感度与完整分配内容损失图。"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from kvstudy.artifacts import read_json as read
from experiments.static_layer_allocation.code.numerics_logits import file_sha
from kvstudy.paths import project_root


matplotlib.use('Agg')


FORMATS = ['online_pre', 'online_post', 'offline_pre']
FORMAT_LABELS = ['Online pre, H=0', 'Online post, H=0', 'Offline pre, H=8, R=128']
COLORS = ['#007f7c', '#dc7b32', '#9154a1', '#4379b0', '#bf9225', '#708d45',
          '#4379b0', '#bf9225', '#708d45']


def matrix_figures(result: Path, output: Path, summaries: dict) -> None:
    """九种同预算策略同时保留；新矩阵采用本轮当前计算条件。"""
    measures = [('nll', 'key_budget_nll.png', 'Content NLL (nat/token)')]
    for metric, filename, ylabel in measures:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True)
        for row, model in enumerate(('qwen', 'llama')):
            summary = summaries[model]
            seeds = read(result / model / 'allocations.json')['random_seeds']
            strategies = ['sensitivity', 'reconstruction', 'inverse']
            strategies += [f'random_{seed}' for seed in seeds]
            strategies += [f'protected_random_{seed}' for seed in seeds]
            labels = ['Sensitivity', 'Reconstruction', 'Reverse', 'Random 1', 'Random 2',
                      'Random 3', 'Protect + random 1', 'Protect + random 2', 'Protect + random 3']
            for col, fmt in enumerate(FORMATS):
                ax = axes[row, col]
                budgets = sorted(summary['budget_comparisons'][fmt], key=int)

                def value(name: str) -> float:
                    means = summary['path_means']['current'][name]
                    return means['content_nll']

                for index, strategy in enumerate(strategies):
                    names = [f'{fmt}_k4_layers_{r}_{strategy}' for r in budgets]
                    ax.plot([int(r) for r in budgets], [value(name) for name in names],
                            color=COLORS[index], ls='--' if index >= 6 else '-',
                            marker='s' if index >= 6 else 'o', ms=4, lw=1.4,
                            label=labels[index], alpha=.9)
                for name, label, color, style in ((fmt + '_k8v4', 'K8V4', '#222222', '-.'),
                                                  ('bf16', 'BF16', '#222222', ':'),
                                                  ('kv8', 'KV8', '#aaaaaa', '--')):
                    ax.axhline(value(name), color=color, ls=style, lw=1, label=label)
                ax.set_title(model.capitalize() + ' | ' + FORMAT_LABELS[col])
                ax.set_xlabel('Number of K4 layers')
                ax.set_ylabel(ylabel)
                ax.set_xticks([int(r) for r in budgets])
                ax.grid(alpha=.2)
                ax.set_yscale('log')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=6, frameon=False,
                   fontsize=9)
        fig.supxlabel('Source: reviewed numerics_summary.json | Validation n=16 per model | '
                      'Equal-weight questions | Current condition', fontsize=9)
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def run() -> None:
    """逐项核对审核索引后才生成报告图片。"""
    root = project_root()
    result = root / 'research/phase1/experiments/static_layer_allocation/results'
    review = read(result / 'numerics_review.json')
    if review['status'] != 'passed':
        raise RuntimeError('配对数值结果尚未审核通过')
    for name, checksum in review['source_files'].items():
        if file_sha(result / name) != checksum:
            raise RuntimeError('审核后的结果字节改变：' + name)
    output = root / 'research/phase1/experiments/static_layer_allocation/report'
    summaries = {model: read(result / model / 'numerics_summary.json')
                 for model in ('qwen', 'llama')}
    matrix_figures(result, output, summaries)
    from experiments.static_layer_allocation.code import content_plot

    content_plot.plot(root)


if __name__ == '__main__':
    run()
