"""汇总完整答案路径、同预算保护对照和题目配对的实测数值包络。"""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.static_layer_allocation.code.numerics_metrics import paired_reliability, spearman


def variants(rows: list[dict[str, Any]], modes: list[str]) -> list[tuple[str, int]]:
    """重复较少的题目使用最后一次测量填充，保留原始重复数。"""
    return [(mode, repeat) for mode in modes
            for repeat in range(max(len(row['precisions'][mode]) for row in rows))]


def measurements(row: dict[str, Any], mode: str, repeat: int) -> dict[str, Any]:
    """按指定实测条件取方案，填充不会扩大逐题最小／最大包络。"""
    records = row['precisions'][mode]
    return records[min(repeat, len(records) - 1)]['schemes']


def effects(
    rows: list[dict[str, Any]], candidate: str, reference: str, metric: str,
    selected: list[tuple[str, int]],
) -> np.ndarray:
    """方案差值始终使用同题、同精度、同重复的共同参照。"""
    return np.array([[measurements(row, mode, repeat)[candidate][metric]
                      - measurements(row, mode, repeat)[reference][metric]
                      for mode, repeat in selected] for row in rows], dtype=np.float64)


def paired_comparison(
    rows: list[dict[str, Any]], candidate: str, reference: str,
    selected: list[tuple[str, int]], design: dict[str, Any],
) -> dict[str, Any]:
    """内容损失与完整路径损失分别重采样，题内 token 已等权平均。"""
    spec = design['reliability']
    metrics = {}
    for metric in ('content_nll', 'all_path_nll'):
        values = effects(rows, candidate, reference, metric, selected)
        measured = paired_reliability(values, spec['bootstrap_seed'], spec['bootstrap_samples'])
        contribution = values[:, -1]
        order = np.argsort(-np.abs(contribution), kind='stable')
        total = float(np.abs(contribution).sum())
        metrics[metric] = {
            **measured, 'paired_question_effects': values.tolist(),
            'largest_absolute_contribution_ids': [rows[i]['id'] for i in order[:3]],
            'largest_question_absolute_share': float(abs(contribution[order[0]]) / total)
            if total else None,
        }
    return {'candidate': candidate, 'reference': reference,
            'variants': [{'precision': mode, 'repeat': repeat} for mode, repeat in selected],
            'metrics': metrics}


def generation_divergence(row: dict[str, Any], scheme: dict[str, Any]) -> dict[str, Any] | None:
    """首次自由生成偏离单独保存；并列时指定路径未必立即偏离。"""
    generated = scheme['generation']['token_ids']
    for index, target in enumerate(row['answer_ids']):
        if index >= len(generated) or generated[index] != target:
            frame = scheme['frames'][index]
            return {'answer_index': index, 'role': frame['role'], 'target_id': target,
                    'generated_id': generated[index] if index < len(generated) else None,
                    'correct_path_margin': frame['margin'],
                    'canonical_path_first_nonpositive_index':
                    scheme['first_nonpositive']['answer_index']
                    if scheme['first_nonpositive'] else None}
    return None


def path_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总两种损失、三种文本评分、首次指定路径分歧及严格正路径关系。"""
    result = {}
    for mode in rows[0]['precisions']:
        selected = [row['precisions'][mode][0]['schemes'] for row in rows]
        means = {}
        for name in selected[0]:
            values = [row[name] for row in selected]
            means[name] = {
                key: float(np.mean([row[key] for row in values]))
                for key in ('content_nll', 'all_path_nll')
            }
            means[name].update({key: sum(row['generation'][key] for row in values)
                               for key in ('content_correct', 'format_compliant',
                                           'formatted_correct')})
            means[name].update({
                'n': len(rows), 'positive_canonical_paths': sum(
                    row['all_strictly_positive'] for row in values),
                'nonpositive_paths_with_correct_content': sum(
                    not row['all_strictly_positive'] and row['generation']['content_correct']
                    for row in values),
                'zero_margin_questions': sum(any(frame['margin'] == 0
                                                for frame in row['frames']) for row in values),
                'mean_minimum_margin': float(np.mean([row['minimum']['margin'] for row in values])),
            })
        result[mode] = means
    return result


def budget_comparisons(
    rows: list[dict[str, Any]], allocation: dict[str, Any], plan: dict[str, Any],
    design: dict[str, Any],
) -> dict[str, Any]:
    """在线 post 使用三精度包络；其他格式只报告当前题目配对区间。"""
    result = {}
    for fmt in plan['formats']:
        result[fmt] = {}
        selected = variants(rows, design['precisions']) if fmt in design['precision_formats'] \
            else [('current', 0)]
        for r in plan['models'][allocation['model']]['k4_layers']:
            prefix = f'{fmt}_k4_layers_{r}_'
            sensitive = prefix + 'sensitivity'
            names = [name for name in allocation['allocations']
                     if name.startswith(prefix) and name != sensitive]
            comparisons = {name: paired_comparison(rows, sensitive, name, selected, design)
                           for name in names}
            randoms = [name for name in names if name[len(prefix):].startswith('random_')]
            protected = [name for name in names
                         if name[len(prefix):].startswith('protected_random_')]
            result[fmt][str(r)] = {
                'full_precision_envelope': fmt in design['precision_formats'],
                'same_budget_comparisons': comparisons,
                'sensitivity_reliably_below_all_random': all(
                    comparisons[name]['metrics']['content_nll']['status'] == 'reliably_lower'
                    for name in randoms) if fmt in design['precision_formats'] else None,
                'sensitivity_reliably_below_all_protected_random': all(
                    comparisons[name]['metrics']['content_nll']['status'] == 'reliably_lower'
                    for name in protected) if fmt in design['precision_formats'] else None,
            }
    return result


def layer_summary(
    rows: list[dict[str, Any]], probe: dict[str, Any], design: dict[str, Any],
    raw: dict[str, Any], matched_local: list[float] | None,
) -> dict[str, Any]:
    """保存可靠单层、两半秩相关及 S 相关；不足两层的相关系数保留未定义。"""
    names = [name for name in rows[0]['precisions']['current'][0]['schemes']
             if name.startswith('single_layer_')]
    names.sort(key=lambda name: int(name.split('_')[-1]))
    base = design['single_layer_format'] + '_k8v4'
    selected = variants(rows, design['precisions'])
    reliable = {name: paired_comparison(rows, name, base, selected, design) for name in names}
    local = [probe['formats'][design['single_layer_format']]['layers'][
        int(name.split('_')[-1])]['sensitivity'] for name in names]
    modes = {}
    for mode in design['precisions']:
        layer_rows = []
        for name in names:
            per_question = effects(rows, name, base, 'content_nll', [(mode, 0)])[:, 0]
            all_path = effects(rows, name, base, 'all_path_nll', [(mode, 0)])[:, 0]
            kl = [item[mode]['single_layers'][name]['content_kl_baseline']
                  for item in raw['questions']]
            layer_rows.append({'layer': int(name.split('_')[-1]), 'name': name,
                               'mean_delta_content_nll': float(per_question.mean()),
                               'mean_delta_all_path_nll': float(all_path.mean()),
                               'mean_content_kl_baseline': float(np.mean(kl)),
                               'per_question_delta_content_nll': per_question.tolist(),
                               'per_question_content_kl_baseline': kl})
        content = [row['mean_delta_content_nll'] for row in layer_rows]
        kl = [row['mean_content_kl_baseline'] for row in layer_rows]
        significant = [i for i, name in enumerate(names)
                       if reliable[name]['metrics']['content_nll']['status'] != 'unresolved']
        halves = design['ranking_halves']
        half_content = [[float(np.mean([row['per_question_delta_content_nll'][i]
                                        for i in indices])) for row in layer_rows]
                        for indices in halves]
        half_kl = [[float(np.mean([row['per_question_content_kl_baseline'][i]
                                  for i in indices])) for row in layer_rows] for indices in halves]
        modes[mode] = {
            'layers': layer_rows, 'half_content_nll_spearman': spearman(*half_content),
            'half_content_kl_spearman': spearman(*half_kl),
            'calibration_s_vs_content_nll_spearman': spearman(local, content),
            'calibration_s_vs_content_kl_spearman': spearman(local, kl),
            'reliable_layer_count': len(significant),
            'calibration_s_vs_reliable_content_nll_spearman': spearman(
                [local[i] for i in significant], [content[i] for i in significant]),
            'matched_local_s_vs_content_nll_spearman':
            spearman(matched_local, content) if matched_local else None,
            'matched_local_s_vs_content_kl_spearman':
            spearman(matched_local, kl) if matched_local else None,
        }
    return {'baseline': base, 'reliability': reliable, 'precisions': modes,
            'signal_status': 'insufficient_reliable_layer_signal' if all(
                value['metrics']['content_nll']['status'] == 'unresolved'
                for value in reliable.values()) else 'reliable_layer_effects_present'}
