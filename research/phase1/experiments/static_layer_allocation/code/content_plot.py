"""只从审核通过的内容测量作图，显示全部模型、格式与预算。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from kvstudy.paths import project_root

COLORS = ['#1761a0', '#d18120', '#a72f37', '#447f48', '#855397', '#6d6d6d']
LABELS = ['Sensitivity', 'Reconstruction', 'Inverse', 'Random 1', 'Random 2', 'Random 3']
FORMATS = ['online_pre', 'online_post', 'offline_pre']
FORMAT_NAMES = ['Online pre, H=0', 'Online post, H=0', 'Offline pre, H=8']


def load(path: Path) -> dict:
    """读取审核过的小型正式结果。"""
    return json.loads(path.read_text('utf-8'))


def plot() -> None:
    """生成科学图，题目为统计单位，图表数据不从报告文字反推。"""
    root = project_root()
    folder = root / 'research/phase1/experiments/static_layer_allocation'
    result, output = folder / 'results', folder / 'report'
    review = load(result / 'review.json')
    if review['status'] != 'passed':
        raise RuntimeError('结果审核未通过')
    probes = {m: load(result / m / 'probe.json') for m in ('qwen', 'llama')}
    gates = {m: load(result / m / 'gate.json') for m in probes}
    allocations = {m: load(result / m / 'allocations.json') for m in probes}
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, model in enumerate(probes):
        for col, name in enumerate(FORMATS):
            data = probes[model]['formats'][name]['layers']
            ax = axes[row, col]
            ax.plot([x['layer'] for x in data], [x['sensitivity'] for x in data],
                    color=COLORS[0], marker='o', ms=3, label='All content tokens')
            first = np.mean([[frame['formats'][name]['first_content_layers'][i]['sensitivity']
                              for i in range(len(data))]
                             for frame in probes[model]['samples']], axis=0)
            ax.plot(range(len(data)), first, color=COLORS[1], ls='--', label='First content token')
            ax.set_yscale('symlog', linthresh=1e-5)
            ax.set_title(model.capitalize() + ' | ' + FORMAT_NAMES[col])
            ax.set_xlabel('Layer (zero based)')
            ax.set_ylabel('Local output relative squared change')
            ax.grid(alpha=.2)
            if row == col == 0:
                ax.legend(frameon=False)
    fig.savefig(output / 'key_downgrade_layers.png', dpi=160)
    plt.close(fig)
    for metric, filename, ylabel in (
        ('nll', 'key_budget_nll.png', 'Question mean content NLL (nat/token)'),
        ('free_exact', 'key_budget_accuracy.png', 'Complete answer exact accuracy'),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for row, model in enumerate(probes):
            alloc, gate = allocations[model], gates[model]
            strategies = ['sensitivity', 'reconstruction', 'inverse']
            strategies += [f'random_{seed}' for seed in alloc['random_seeds']]
            for col, name in enumerate(FORMATS):
                ax = axes[row, col]
                for i, strategy in enumerate(strategies):
                    names = [name + '_' + budget + '_' + strategy
                             for budget in alloc['budget_groups'][name]]
                    x = [alloc['allocations'][scheme]['r'] for scheme in names]
                    y = [gate['means'][scheme][metric] for scheme in names]
                    ax.plot(x, y, color=COLORS[i], marker='o', ms=4, label=LABELS[i])
                for baseline, color, style in ((name + '_k8v4', '#000000', '-.'),
                                                ('bf16', '#000000', ':'),
                                                ('kv8', '#aaaaaa', '--')):
                    ax.axhline(gate['means'][baseline][metric], color=color, ls=style,
                               lw=1, label=baseline.split('_')[-1].upper())
                ax.set_title(model.capitalize() + ' | ' + FORMAT_NAMES[col])
                ax.set_xlabel('Number of K4 layers')
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                if metric == 'nll':
                    ax.set_yscale('log')
                if metric == 'free_exact':
                    ax.set_ylim(-.03, 1.03)
                if row == col == 0:
                    ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    diagnosis = load(result / 'llama/single_layer.json')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key, label in zip(axes, ['delta_nll', 'kl_baseline'],
                              ['Single-layer delta NLL (nat/token)', 'KL(B || single) (nat)']):
        rows = diagnosis['layers']
        ax.scatter([row['matched_local_sensitivity'] for row in rows],
                   [row[key] for row in rows], color=COLORS[0], s=26)
        for row in rows:
            ax.annotate(str(row['layer']), (row['matched_local_sensitivity'], row[key]),
                        xytext=(3, 3), textcoords='offset points', fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.set_xlabel('Local S in K8V4 background')
        ax.set_ylabel(label)
        ax.set_title('Llama | Online post, H=0 | n=16')
        ax.grid(alpha=.2)
    fig.savefig(output / 'single_layer_prediction.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    means = diagnosis['additive_means']
    for ax, predicted, label in zip(axes, ['additive', 'sum'],
                                    ['KL after adding logits', 'Sum of single-layer KL']):
        xs, ys = [], []
        for name in means:
            actual = np.mean([row['additive'][name]['actual']['kl_baseline']
                              for row in diagnosis['rows']])
            value = np.mean([row['additive'][name]['additive']['kl_baseline']
                             for row in diagnosis['rows']]) if predicted == 'additive' \
                else means[name]['sum_single_layer_kl']
            xs.append(actual)
            ys.append(value)
            strategy = allocations['llama']['allocations'][name]['strategy']
            strategies = ['sensitivity', 'reconstruction', 'inverse']
            strategies += [f'random_{seed}' for seed in allocations['llama']['random_seeds']]
            index = strategies.index(strategy)
            r = allocations['llama']['allocations'][name]['r']
            ax.scatter(actual, value, color=COLORS[index], s=22 + r * 3,
                       label=LABELS[index] if r == 2 else None)
        low, high = min(xs + ys) * .8, max(xs + ys) * 1.2
        ax.plot([low, high], [low, high], color='#777777', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Actual combination KL(B || M), nat')
        ax.set_ylabel(label + ', nat')
        ax.set_title('Llama | Online post | 18 budget strategies')
        ax.grid(alpha=.2)
        if predicted == 'additive':
            ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / 'logit_superposition.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    plot()
