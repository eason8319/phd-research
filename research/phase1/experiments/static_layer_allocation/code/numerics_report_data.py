"""从审核后的正式结果整理报告表格、精度诊断与真实题目案例。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.numerics_logits import file_sha


def read(path: Path) -> dict:
    """读取审核后的 JSON，不读取完整预测或模型实体。"""
    return json.loads(path.read_text('utf-8'))


def model_data(folder: Path) -> dict:
    """按题目等权汇总首内容损失，其他指标直接引用正式汇总。"""
    summary = read(folder / 'numerics_summary.json')
    allocation = read(folder / 'numerical_allocations.json')
    rows = [read(path) for path in sorted((folder / 'numerical_questions').glob('*.json'))]
    result = {'peak_reserved_gib': summary['peak_reserved_bytes'] / 2**30,
              'repeats': summary['repeats'], 'protected': allocation['protected'],
              'aliases': allocation['configuration_aliases'],
              'rescoring': read(folder / 'rescoring.json'), 'precisions': {}}
    for mode, means in summary['path_means'].items():
        diagnostic = summary['single_layer']['precisions'][mode]
        diagnostic = {key: value for key, value in diagnostic.items() if key != 'layers'}
        layers = summary['single_layer']['reliability']
        reliable = {name: value['metrics']['content_nll'] for name, value in layers.items()
                    if value['metrics']['content_nll']['status'] != 'unresolved'}
        reliable = {name: {key: value for key, value in item.items()
                          if key != 'paired_question_effects'} for name, item in reliable.items()}
        endpoints = {}
        for name, item in means.items():
            if 'k4_layers_' in name or 'single_layer_' in name:
                continue
            first = [row['precisions'][mode][0]['schemes'][name]['frames'][
                row['content_indices'][0]]['nll'] for row in rows]
            endpoints[name] = {**item, 'first_content_nll': float(np.mean(first)),
                               **summary['raw_diagnostics']['precisions'][mode][
                                   'distributions'][name]}
        result['precisions'][mode] = {
            'single_layer': diagnostic, 'reliable_layers': reliable, 'endpoints': endpoints,
            'budget_means': {name: item for name, item in means.items() if 'k4_layers_' in name},
            'single_values': {name: item for name, item in means.items()
                              if 'single_layer_' in name},
            'additive_sensitivity': {name: item for name, item in
                summary['raw_diagnostics']['precisions'][mode]['additive'].items()
                if name.endswith('_sensitivity')},
        }
    comparisons = {}
    for fmt, budgets in summary['budget_comparisons'].items():
        comparisons[fmt] = {}
        for r, item in budgets.items():
            comparisons[fmt][r] = {
                'full_precision_envelope': item['full_precision_envelope'],
                'below_all_random': item['sensitivity_reliably_below_all_random'],
                'below_all_protected': item['sensitivity_reliably_below_all_protected_random'],
                'comparisons': {name: {
                    metric: {key: value for key, value in measured.items()
                             if key != 'paired_question_effects'}
                    for metric, measured in measured_item['metrics'].items()}
                    for name, measured_item in item['same_budget_comparisons'].items()},
            }
    result['comparisons'] = comparisons
    result['current_distributions'] = summary['raw_diagnostics']['precisions'][
        'current']['distributions']
    result['cases'] = {}
    for row in rows:
        if row['key'] != 'key-909c1bd1':
            continue
        result['cases'][row['key']] = {
            'answer': row['answer'], 'content_indices': row['content_indices'],
            'precisions': {mode: runs[0]['schemes'] for mode, runs in
                           row['precisions'].items()},
        }
    return result


def run() -> None:
    """审核索引字节一致时才整理服务器报告数据，原正式结果不变。"""
    root = project_root()
    formal = root / 'research/phase1/experiments/static_layer_allocation/results'
    review = read(formal / 'numerics_review.json')
    if review['status'] != 'passed':
        raise RuntimeError('原始结果尚未审核通过')
    for name, checksum in review['source_files'].items():
        if file_sha(formal / name) != checksum:
            raise RuntimeError('审核结果字节变化：' + name)
    data = {'review_sha256': file_sha(formal / 'numerics_review.json'),
            'code_sha256': file_sha(Path(__file__)),
            'models': {model: model_data(formal / model) for model in ('qwen', 'llama')}}
    destination = root / 'research/.runtime/static-key-numerics/report_data.json'
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print('REVIEWED_REPORT_DATA_SAVED', destination.relative_to(root), flush=True)


if __name__ == '__main__':
    run()
