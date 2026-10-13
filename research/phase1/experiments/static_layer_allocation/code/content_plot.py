"""从审核后的内容探针生成报告仍使用的逐层降位宽图。"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from kvstudy.artifacts import read_json as load
from kvstudy.paths import project_root


matplotlib.use('Agg')


COLORS = ['#1761a0', '#d18120', '#a72f37', '#447f48', '#855397', '#6d6d6d']
FORMATS = ['online_pre', 'online_post', 'offline_pre']
FORMAT_NAMES = ['Online pre, H=0', 'Online post, H=0', 'Offline pre, H=8']


def plot(root: Path | None = None) -> None:
    """生成科学图，题目为统计单位，图表数据不从报告文字反推。"""
    root = root or project_root()
    folder = root / 'research/phase1/experiments/static_layer_allocation'
    result, output = folder / 'results', folder / 'report'
    review = load(result / 'review.json')
    if review['status'] != 'passed':
        raise RuntimeError('结果审核未通过')
    probes = {m: load(result / m / 'probe.json') for m in ('qwen', 'llama')}
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
    fig.supxlabel('Source: reviewed probe.json | Calibration n=16 | '
                  'Valid groups, content tokens, then equal-weight questions', fontsize=9)
    fig.savefig(output / 'key_downgrade_layers.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    plot()
