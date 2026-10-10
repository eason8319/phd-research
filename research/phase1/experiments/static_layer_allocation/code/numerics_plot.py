"""审核后绘制当前完整矩阵、三精度单层影响和绝对叠加闭合图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.numerics_logits import file_sha

FORMATS = ['online_pre', 'online_post', 'offline_pre']
FORMAT_LABELS = ['Online pre, H=0', 'Online post, H=0', 'Offline pre, H=8, R=128']
MODES = ['current', 'head_fp32', 'residual_fp32']
MODE_LABELS = ['Current', 'FP32 head', 'FP32 residual + head']
COLORS = ['#007f7c', '#dc7b32', '#9154a1', '#4379b0', '#bf9225', '#708d45',
          '#4379b0', '#bf9225', '#708d45']


def read(path: Path) -> dict:
    """只读已经审核的小型结果。"""
    return json.loads(path.read_text('utf-8'))


def matrix_figures(result: Path, output: Path, summaries: dict, allocations: dict) -> None:
    """九种同预算策略同时保留；新矩阵采用本轮当前计算条件。"""
    measures = [('nll', 'key_budget_nll.png', 'Content NLL (nat/token)'),
                ('kl', 'key_budget_kl.png', 'Content KL(BF16 || A) (nat/token)'),
                ('accuracy', 'key_budget_accuracy.png', 'Normalized content accuracy')]
    for metric, filename, ylabel in measures:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.2), constrained_layout=True)
        for row, model in enumerate(('qwen', 'llama')):
            allocation, summary = allocations[model], summaries[model]
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
                    if metric == 'kl':
                        return summary['raw_diagnostics']['precisions']['current'][
                            'distributions'][name]['content_kl_bf16']
                    means = summary['path_means']['current'][name]
                    return means['content_nll'] if metric == 'nll' else \
                        means['content_correct'] / means['n']

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
                if metric == 'nll':
                    ax.set_yscale('log')
                elif metric == 'kl':
                    ax.set_yscale('symlog', linthresh=1e-9 if model == 'qwen' else 1e-7)
                    ax.set_ylim(bottom=0)
                else:
                    ax.set_ylim(-.03, 1.03)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=6, frameon=False,
                   fontsize=9)
        fig.supxlabel('Source: reviewed numerics_summary.json | Validation n=16 per model | '
                      'Equal-weight questions | Current condition', fontsize=9)
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def single_layers(result: Path, output: Path, summary: dict) -> None:
    """三个条件分别展示校准 S 与独立验证单层效应，避免把层作为独立样本。"""
    probe = read(result / 'llama/probe.json')['formats']['online_post']['layers']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, mode in enumerate(MODES):
        diagnostic = summary['single_layer']['precisions'][mode]
        rows = diagnostic['layers']
        for row, key, label in ((0, 'mean_delta_content_nll', 'Delta content NLL (nat/token)'),
                                (1, 'mean_content_kl_baseline', 'KL(B || single) (nat/token)')):
            ax = axes[row, col]
            x = [probe[item['layer']]['sensitivity'] for item in rows]
            y = [item[key] for item in rows]
            reliable = [summary['single_layer']['reliability'][item['name']]['metrics'][
                'content_nll']['status'] != 'unresolved' for item in rows]
            ax.scatter(x, y, c=['#007f7c' if valid else '#a5aab0' for valid in reliable], s=28)
            for item, left, right in zip(rows, x, y):
                if item['layer'] not in (0, 1, 5, 14, 15, 20, 25, 27):
                    continue
                ax.annotate(str(item['layer']), (left, right), xytext=(3, 3),
                            textcoords='offset points', fontsize=7)
            ax.set_xscale('log')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            ax.set_title('Llama | ' + MODE_LABELS[col])
            ax.set_xlabel('Frozen calibration local S')
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    fig.supxlabel('Source: reviewed numerics_summary.json and probe.json | '
                  'Online post, H=0 | Validation n=16 | Labels: selected layer indices; '
                  'teal: resolved delta NLL', fontsize=9)
    fig.savefig(output / 'single_layer_prediction.png', dpi=160)
    plt.close(fig)


def closure(output: Path, summary: dict, allocation: dict) -> None:
    """绝对 RMS 与分布叠加各自成图，三个预算和全部策略均保留。"""
    names = [name for name, spec in allocation['allocations'].items()
             if spec['format'] == 'online_post' and spec['r'] in (2, 8, 16)]
    budget_colors = {2: '#4379b0', 8: '#dc7b32', 16: '#007f7c'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, scope in zip(axes, ('all_path', 'content')):
        for name in names:
            spec = allocation['allocations'][name]
            sensitive = spec['strategy'] == 'sensitivity'
            y = [summary['raw_diagnostics']['precisions'][mode]['additive'][name][scope][
                'root_mean_absolute_centered_residual_mse'] for mode in MODES]
            ax.plot(range(3), y, color=budget_colors[spec['r']],
                    alpha=1 if sensitive else .2, lw=2.3 if sensitive else .8,
                    marker='o' if sensitive else None, ms=5,
                    label=f"Sensitivity, r={spec['r']}" if sensitive else None)
        ax.set_xticks(range(3), MODE_LABELS)
        ax.set_yscale('log')
        ax.set_ylim(.025, .85)
        ax.set_yticks([.03, .06, .1, .2, .4, .8], ['0.03', '0.06', '0.1', '0.2', '0.4', '0.8'])
        ax.set_ylabel('Absolute centered additive residual RMS (logit)')
        label = 'All answer + termination' if scope == 'all_path' else 'Content'
        ax.set_title('Llama | ' + label)
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
    fig.supxlabel('Source: reviewed numerics_summary.json | Online post, H=0 | Validation n=16 | '
                  'Root of equal-weight question mean MSE; thin lines: all other strategies',
                  fontsize=9)
    fig.savefig(output / 'numerical_closure.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, mode, label in zip(axes, MODES, MODE_LABELS):
        values = summary['raw_diagnostics']['precisions'][mode]['additive']
        for name in names:
            item = values[name]['content']
            color = budget_colors[allocation['allocations'][name]['r']]
            ax.scatter(item['actual_kl_baseline'], item['additive_kl_baseline'],
                       color=color, marker='o', s=28, alpha=.75)
            ax.scatter(item['actual_kl_baseline'], item['sum_single_layer_kl'],
                       color=color, marker='x', s=28, alpha=.75)
        keys = ('actual_kl_baseline', 'additive_kl_baseline', 'sum_single_layer_kl')
        points = [value['content'][key] for name, value in values.items()
                  if name in names for key in keys]
        lower, upper = min(points) * .5, max(points) * 1.5
        ax.plot([lower, upper], [lower, upper], color='#999999', ls='--', lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel('Actual combination KL(B || A) (nat/token)')
        ax.set_ylabel('Predicted combination KL (nat/token)')
        ax.set_title('Llama | ' + label)
        ax.grid(alpha=.2)
    handles = [Line2D([], [], color='#555555', marker='o', ls='', label='Additive logits'),
               Line2D([], [], color='#555555', marker='x', ls='', label='Sum of single KL')]
    fig.legend(handles=handles, loc='outside upper center', ncol=2, frameon=False)
    fig.supxlabel('Source: reviewed numerics_summary.json | Online post, H=0 | Validation n=16 | '
                  'Equal-weight content token / question means | '
                  'Colors: r=2 (blue), 8 (orange), 16 (teal)', fontsize=9)
    fig.savefig(output / 'logit_superposition.png', dpi=160)
    plt.close(fig)


def paired_budgets(output: Path, summaries: dict) -> None:
    """区间包含题目重采样与三精度实测包络；灰色表示未分辨差异。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, model in enumerate(('qwen', 'llama')):
        budgets = summaries[model]['budget_comparisons']['online_post']
        for col, r in enumerate(sorted(budgets, key=int)):
            ax = axes[row, col]
            items = budgets[r]['same_budget_comparisons']
            labels = []
            for index, (name, item) in enumerate(items.items()):
                label = name.split(f'_k4_layers_{r}_')[1]
                label = label.replace('protected_random_', 'Protect + random ')
                label = label.replace('random_', 'Random ')
                for seed, digit in (('202609231', '1'), ('202609232', '2'), ('202609233', '3')):
                    label = label.replace(seed, digit)
                labels.append(label.capitalize())
                metric = item['metrics']['content_nll']
                lower, upper = metric['question_bootstrap_numeric_envelope_95']
                color = {'reliably_lower': '#007f7c', 'reliably_higher': '#dc7b32',
                         'unresolved': '#899198'}[metric['status']]
                ax.plot([lower, upper], [index, index], color=color, lw=2)
                ax.scatter(metric['measured_variant_means'][-1], index, color=color, s=24)
            ax.axvline(0, color='#222222', ls='--', lw=1)
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xscale('symlog', linthresh=1e-8 if model == 'qwen' else 1e-6)
            if model == 'qwen':
                ax.set_xlim(-12, 1.1e-6)
                ax.set_xticks([-10, -1e-4, -1e-6, 0, 1e-6],
                              ['-10', '-1e-4', '-1e-6', '0', '1e-6'])
            ax.tick_params(axis='x', labelsize=8)
            ax.minorticks_off()
            ax.set_xlabel('Sensitivity minus comparison NLL (nat/token)')
            ax.set_title(f'{model.capitalize()} | Online post | r={r}')
            ax.grid(axis='x', alpha=.2)
    fig.supxlabel('Source: reviewed numerics_summary.json | Validation n=16 | '
                  '95% paired question bootstrap + observed precision/repeat envelope | '
                  'Dot: FP32 residual mean; gray: unresolved', fontsize=9)
    fig.savefig(output / 'paired_budget_comparison.png', dpi=160)
    plt.close(fig)


def full_path_case(result: Path, output: Path) -> None:
    """同一实际题目展示全部 token 领先量，前缀分支与内容支持度放在一起。"""
    rows = [read(path) for path in (result / 'qwen/numerical_questions').glob('*.json')]
    row = next(item for item in rows if item['key'] == 'key-909c1bd1')
    choices = [('current', 'online_pre_k4_layers_2_sensitivity', 'Online pre, r=2'),
               ('current', 'online_post_k4_layers_14_sensitivity', 'Online post, r=14'),
               ('current', 'offline_pre_k4_layers_7_sensitivity', 'Offline pre, r=7'),
               ('head_fp32', 'online_post_k4_layers_14_sensitivity', 'Post, FP32 head'),
               ('residual_fp32', 'online_post_k4_layers_14_sensitivity', 'Post, FP32 residual')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    first = row['precisions']['current'][0]['schemes'][choices[0][1]]['frames']
    labels = ['EOS' if frame['role'] == 'termination' else frame['target_text'] for frame in first]
    for index, (mode, name, label) in enumerate(choices):
        measured = row['precisions'][mode][0]['schemes'][name]
        frames = measured['frames']
        axes[0].plot(range(len(frames)), [frame['margin'] for frame in frames],
                     color=COLORS[index], marker='o', ms=4, label=label)
        axes[1].plot([0, 1], [measured['content_nll'], measured['all_path_nll']],
                     color=COLORS[index], marker='o', ms=5, label=label)
    axes[0].axhline(0, color='#222222', ls='--', lw=1)
    axes[0].set_xticks(range(len(first)), labels)
    axes[0].set_xlabel('Actual canonical answer tokens, then termination')
    axes[0].set_ylabel('Correct token minus strongest wrong logit')
    axes[1].set_xticks([0, 1], ['Content mean', 'All path mean'])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Target NLL (nat/token)')
    for ax in axes:
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False, fontsize=9)
    axes[0].set_title(row['key'] + ' -> ' + row['answer'], fontsize=12)
    axes[1].set_title('Content support and full answer path', fontsize=12)
    fig.supxlabel('Source: reviewed numerical_questions | One fixed Qwen validation question | '
                  'Shared correct prefix; each scheme maintains its own cache', fontsize=9)
    fig.savefig(output / 'full_path_margins.png', dpi=160)
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
    allocations = {model: read(result / model / 'numerical_allocations.json')
                   for model in ('qwen', 'llama')}
    matrix_figures(result, output, summaries, allocations)
    single_layers(result, output, summaries['llama'])
    closure(output, summaries['llama'], allocations['llama'])
    paired_budgets(output, summaries)
    full_path_case(result, output)


if __name__ == '__main__':
    run()
