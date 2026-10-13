"""从完整词表原始实体复算增量、绝对叠加残差及逐 token 路径审核。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.static_layer_allocation.code.content_review import compare
from experiments.static_layer_allocation.code.metrics import vocabulary_kl
from experiments.static_layer_allocation.code.numerics_metrics import (
    centered_energy,
    incremental_comparison,
    path_metrics,
    score_answer,
)


def file_sha(path: Path) -> str:
    """分块校验大实体，避免校验本身复制完整 logits。"""
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def kl_mean(base: np.ndarray, actual: np.ndarray, positions: list[int]) -> float:
    """先平均题内指定位置的完整词表 KL，单位 nat/token。"""
    return float(np.mean([vocabulary_kl(base[i], actual[i]) for i in positions]))


def additive(
    base: np.ndarray, singles: dict[int, np.ndarray], actual: np.ndarray,
    layers: list[int], positions: list[int],
) -> dict[str, float | None]:
    """绝对残差能量先题内平均；相对能量单独记录，不代替绝对闭合。"""
    b = base[positions].astype(np.float64)
    a = actual[positions].astype(np.float64)
    added = b.copy()
    for layer in layers:
        added += singles[layer][positions].astype(np.float64) - b
    residual = centered_energy(a - added)
    actual_energy = centered_energy(a - b)
    sum_kl = sum(kl_mean(base, singles[layer], positions) for layer in layers)
    actual_kl = kl_mean(base, actual, positions)
    return {
        'absolute_centered_residual_mse': residual,
        'absolute_centered_residual_rms': residual ** 0.5,
        'actual_increment_centered_mse': actual_energy,
        'actual_increment_centered_rms': actual_energy ** 0.5,
        'residual_over_actual_energy': residual / actual_energy if actual_energy else None,
        'actual_kl_baseline': actual_kl,
        'additive_kl_baseline': kl_mean(b, added, list(range(len(positions)))),
        'actual_to_additive_kl': kl_mean(a, added, list(range(len(positions)))),
        'sum_single_layer_kl': sum_kl, 'sum_single_kl_minus_actual': sum_kl - actual_kl,
    }


def audit_entities(
    root: Path, row: dict[str, Any], tokenizer: Any, checked: dict[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    """逐值复算全部重复的路径指标；完全一致实体只加载并校验一次。"""
    source = row['source_entity']
    compare(file_sha(root / source['path']), source['sha256'], 'fixed prefill entity')
    retained = {}
    for mode, repeats in row['precisions'].items():
        previous = None
        previous_path = None
        for repeat in repeats:
            entity = repeat['logit_entity']
            path = root / entity['path']
            if entity['path'] not in checked:
                compare(file_sha(path), entity['sha256'], 'complete logits bytes')
                compare(path.stat().st_size, entity['bytes'], 'complete logits size')
                checked[entity['path']] = entity['sha256']
            compare(checked[entity['path']], entity['sha256'], 'aliased logits hash')
            if previous_path == entity['path']:
                values = previous
            else:
                packed = torch.load(path, weights_only=False, map_location='cpu')
                compare(packed['input_sha256'], row['input_sha256'], 'actual logits input')
                compare(packed['targets'], row['answer_ids'], 'full canonical path targets')
                values = {name: tensor.numpy() for name, tensor in packed['logits'].items()}
            names = repeat['schemes']
            if previous is not None:
                equal = all(np.array_equal(values[name], previous[name]) for name in names)
                compare(equal, repeat['identical_to_previous'], 'repeat actual value equality')
            for name, measured in names.items():
                z = values[name]
                if z.dtype != np.float32 or not np.isfinite(z).all():
                    raise RuntimeError('原始 logits dtype 或有限性不符')
                expected = path_metrics(z, row['answer_ids'], tokenizer, row['content_indices'])
                compare({key: measured[key] for key in expected}, expected, 'complete path metrics')
                scored = score_answer(measured['generation']['prediction'], row['answer'])
                compare({key: measured['generation'][key] for key in scored}, scored,
                        'uniform exact normalized scoring')
                if repeat['repeat'] == 0 and measured['all_strictly_positive']:
                    compare(measured['generation']['token_ids'], row['answer_ids'],
                            'positive canonical path equals actual greedy')
                first = measured['first_nonpositive']
                generated = measured['generation']['token_ids']
                if repeat['repeat'] == 0 and first and first['margin'] < 0:
                    i = first['answer_index']
                    if generated[:i] == row['answer_ids'][:i]:
                        compare(generated[i], first['greedy_id'], 'first actual negative fork')
            if repeat['repeat'] == 0:
                retained[mode] = values
            previous, previous_path = values, entity['path']
        if len(repeats) == 2:
            compare(repeats[1]['identical_to_previous'], True, 'two-repeat stopping condition')
        elif len(repeats) != 6:
            raise RuntimeError('重复次数不符合冻结扩展规则')
    return retained


def question_analysis(
    row: dict[str, Any], raw: dict[str, dict[str, np.ndarray]],
    allocation: dict[str, Any], design: dict[str, Any],
) -> dict[str, Any]:
    """同精度 B/A 配对增量；完整路径和内容位置分别计算闭合。"""
    base_name = design['single_layer_format'] + '_k8v4'
    positions = list(range(len(row['answer_ids'])))
    content = row['content_indices']
    high = raw['residual_fp32']
    diagnostic_names = list(high)
    result = {'id': row['id']}
    for mode, values in raw.items():
        base = values[base_name]
        singles = {int(name.split('_')[-1]): values[name] for name in values
                   if name.startswith('single_layer_')}
        result[mode] = {
            'single_layers': {f'single_layer_{layer}': {
                'content_kl_baseline': kl_mean(base, value, content),
                'all_path_kl_baseline': kl_mean(base, value, positions),
            } for layer, value in singles.items()},
            'increments_vs_residual_fp32': {
                name: {scope: incremental_comparison(base[chosen], values[name][chosen],
                                                     high[base_name][chosen], high[name][chosen])
                       for scope, chosen in (('content', content), ('all_path', positions))}
                for name in diagnostic_names if name not in ('bf16', 'kv8', base_name)
            }, 'additive': {}, 'distributions': {},
        }
        for name, value in values.items():
            fmt = allocation['allocations'].get(name, {}).get('format')
            baseline = values[fmt + '_k8v4'] if fmt else base
            result[mode]['distributions'][name] = {
                scope + '_kl_bf16': kl_mean(values['bf16'], value, chosen)
                for scope, chosen in (('content', content), ('all_path', positions))}
            result[mode]['distributions'][name].update({
                scope + '_kl_baseline': kl_mean(baseline, value, chosen)
                for scope, chosen in (('content', content), ('all_path', positions))})
        if len(singles) == 32:
            for name, spec in allocation['allocations'].items():
                if spec['format'] == design['single_layer_format'] and spec.get('r'):
                    result[mode]['additive'][name] = {
                        'content': additive(base, singles, values[name],
                                            spec['k4_layers'], content),
                        'all_path': additive(base, singles, values[name],
                                             spec['k4_layers'], positions),
                    }
    return result


def raw_summary(questions: list[dict[str, Any]], design: dict[str, Any]) -> dict[str, Any]:
    """题目等权汇总，绝对 RMS 使用先平均题内 MSE 再开平方的口径。"""
    result = {'questions': questions, 'precisions': {}}
    for mode in design['precisions']:
        additive_means = {}
        for name in questions[0][mode]['additive']:
            additive_means[name] = {}
            for scope in ('content', 'all_path'):
                rows = [row[mode]['additive'][name][scope] for row in questions]
                means = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
                         if any(row[key] is not None for row in rows) else None
                         for key in rows[0]}
                means['root_mean_absolute_centered_residual_mse'] = (
                    means['absolute_centered_residual_mse'] ** 0.5)
                means['root_mean_actual_increment_centered_mse'] = (
                    means['actual_increment_centered_mse'] ** 0.5)
                additive_means[name][scope] = means
        increments = {}
        for name in questions[0][mode]['increments_vs_residual_fp32']:
            increments[name] = {}
            for scope in ('content', 'all_path'):
                rows = [row[mode]['increments_vs_residual_fp32'][name][scope] for row in questions]
                increments[name][scope] = {
                    key: float(np.mean([row[key] for row in rows if row[key] is not None]))
                    if any(row[key] is not None for row in rows) else None for key in rows[0]}
        distributions = {
            name: {key: float(np.mean([row[mode]['distributions'][name][key]
                                      for row in questions])) for key in values}
            for name, values in questions[0][mode]['distributions'].items()}
        result['precisions'][mode] = {'additive': additive_means, 'increments': increments,
                                     'distributions': distributions}
    return result
