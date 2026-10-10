"""在计算节点审核全部原始 logits，再产出配对数值诊断的正式小型结果。"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from kvstudy.paths import project_root
from experiments.static_layer_allocation.code.content import save
from experiments.static_layer_allocation.code.content_review import compare
from experiments.static_layer_allocation.code.numerics import (
    DESIGN, FORMAL, RUNTIME, protected_allocations, read, sha,
)
from experiments.static_layer_allocation.code.numerics_logits import (
    audit_entities, file_sha, question_analysis, raw_summary,
)
from experiments.static_layer_allocation.code.numerics_metrics import score_answer
from experiments.static_layer_allocation.code.numerics_summary import (
    budget_comparisons, generation_divergence, layer_summary, path_summary,
)
from experiments.static_layer_allocation.code.protocol import digest, load_plan
from phase1.representation import MODELS


def check_execution(root: Path, execution: dict[str, Any], design: dict[str, Any]) -> None:
    """执行文件按实际归档字节核对，不以服务器 Git 状态替代源码身份。"""
    compare(execution['design_digest'], digest(design), 'executed frozen numerical design')
    compare(execution['tf32_matmul'], False, 'TF32 matrix calculation')
    compare(execution['tf32_cudnn'], False, 'TF32 cudnn')
    compare(execution['autocast'], False, 'implicit autocast')
    compare(execution['float32_matmul_precision'], 'highest', 'actual FP32 setting')
    archive = root / RUNTIME / 'execution_sources' / execution['commit']
    for name, checksum in execution['code_sha256'].items():
        compare(file_sha(archive / name), checksum, 'actual archived code: ' + name)


def check_allocations(
    root: Path, model: str, allocation: dict[str, Any], design: dict[str, Any],
) -> None:
    """独立复建校准保护层，核对同预算成本及原位宽表不变。"""
    folder = root / FORMAL / model
    base, probe = read(folder / 'allocations.json'), read(folder / 'probe.json')
    geometry = read(root / MODELS[model] / 'config.json')
    expected = protected_allocations(base, probe, load_plan(root), geometry, design)
    compare({key: allocation[key] for key in expected}, expected, 'frozen protected allocations')
    for name, spec in base['allocations'].items():
        compare(allocation['allocations'][name], spec, 'original frozen bit table')
    for name in allocation['protected_random_names']:
        spec = allocation['allocations'][name]
        reference = base['allocations'][f"{spec['format']}_k4_layers_{spec['r']}_sensitivity"]
        compare(spec['cost'], reference['cost'], 'same budget full bytes')


def check_rows(
    root: Path, model: str, rows: list[dict[str, Any]], design: dict[str, Any],
    allocation: dict[str, Any],
) -> None:
    """验证题来源、实际 token、输入字节、重复矩阵及原生成结果逐项一致。"""
    dataset = read(root / design['dataset'])
    expected = [item for item in dataset['rows'] if item['role'] == 'validation']
    compare([row['id'] for row in rows], [item['id'] for item in expected], 'held out ordering')
    compare(len(rows), design['validation_n'], 'held out n')
    prior = {row['id']: row for row in read(root / FORMAL / model / 'gate.json')['rows']}
    for index, (row, item) in enumerate(zip(rows, expected)):
        record = item['models'][model]
        compare(row['question_index'], index, 'question ordinal')
        compare(row['input_sha256'], record['sha256'], 'frozen actual input')
        compare(file_sha(root / record['path']), record['sha256'], 'actual input bytes')
        compare(row['answer_ids'][:-1], record['answer_ids'], 'actual answer tokenization')
        compare(row['content_indices'], record['content_indices'], 'actual content positions')
        compare(row['key'], item['key'], 'readable actual query')
        compare(row['answer'], item['value'], 'actual mapping answer')
        current = row['precisions']['current'][0]['schemes']
        singles = {name for name in current if name.startswith('single_layer_')}
        compare(len(singles), 32 if model == 'llama' else 2, 'single layer scope')
        all_names = {'bf16', 'kv8', *allocation['allocations'], *singles}
        compare(set(current), all_names, 'all current formats and protected random schemes')
        diagnostic = {'bf16', 'kv8', *singles, *[
            name for name, spec in allocation['allocations'].items()
            if spec['format'] in design['precision_formats']]}
        for mode, repeats in row['precisions'].items():
            compare([run['repeat'] for run in repeats], list(range(len(repeats))),
                    'continuous repeat indices')
            if len(repeats) not in (design['repeats_initial'], design['repeats_if_nonidentical']):
                raise RuntimeError('重复次数不符合冻结规则')
            for repeat in repeats:
                names = all_names if mode == 'current' and repeat['repeat'] == 0 else diagnostic
                compare(set(repeat['schemes']), names, 'actual measurement scope')
        for name, old in prior[row['id']]['schemes'].items():
            measured = current[name]
            history = row['historical_device_comparison']['schemes'][name]
            compare(history['same_prediction'],
                    measured['generation']['prediction'] == old['prediction'],
                    'separate device prediction comparison')
            compare(history['content_nll_difference'], measured['content_nll'] - old['nll'],
                    'separate device NLL comparison')


def compact_question(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """回传逐 token 小型测量；完整自由生成与原始 logits 由服务器索引保留。"""
    compact = copy.deepcopy(row)
    compact['question_entity'] = source
    for runs in compact['precisions'].values():
        for run in runs:
            for scheme in run['schemes'].values():
                scheme['first_generation_divergence'] = generation_divergence(row, scheme)
                generated = scheme['generation']
                generated['generated_tokens'] = len(generated['token_ids'])
                del generated['prediction'], generated['token_ids']
    return compact


def check_rescoring(root: Path, model: str, rescoring: dict[str, Any]) -> None:
    """所有旧方案与映射配对同时用精确规范化重评，保留原严格匹配。"""
    gate = read(root / FORMAL / model / 'gate.json')
    compare(rescoring['source_gate']['sha256'], sha(root / FORMAL / model / 'gate.json'),
            'rescoring original gate source')
    for old, new in zip(gate['rows'], rescoring['rows']):
        for name, measured in old['schemes'].items():
            scored = score_answer(measured['prediction'], old['answer'])
            compare(new['schemes'][name], scored, 'all old output scores')
            compare(scored['formatted_correct'], measured['free_exact'], 'retained old strict EM')
    mapping = read(root / FORMAL / model / 'mapping.json')
    for old, new in zip(mapping['rows'], rescoring['mapping_rows']):
        for name in ('original', 'swapped'):
            case = old['cases'][name]
            compare(new['scores'][name], score_answer(case['prediction'], case['answer']),
                    'mapping pair normalized scores')


def run(commit: str) -> None:
    """先逐题审核原始实体；所有模型通过后保存统一审核，不生成图片或报告。"""
    root = project_root()
    design, plan = read(root / DESIGN), load_plan(root)
    compare(sha(root / design['base_plan']), design['base_plan_file_sha256'], 'base plan source')
    formal = root / RUNTIME / 'reviewed'
    formal.mkdir(parents=True, exist_ok=True)
    audit_models, entities = {}, {}
    for model in design['models']:
        folder = root / RUNTIME / 'results' / model
        allocation = read(folder / 'numerical_allocations.json')
        execution = read(folder / 'execution.json')
        check_execution(root, execution, design)
        check_allocations(root, model, allocation, design)
        rows = sorted([read(path) for path in (folder / 'questions').glob('*.json')],
                      key=lambda row: row['question_index'])
        check_rows(root, model, rows, design, allocation)
        tokenizer = AutoTokenizer.from_pretrained(root / MODELS[model])
        compare(rows[0]['answer_ids'][-1],
                tokenizer.convert_tokens_to_ids(design['termination'][model]),
                'canonical scoring termination')
        raw_questions = []
        for row in rows:
            raw = audit_entities(root, row, tokenizer, entities)
            raw_questions.append(question_analysis(row, raw, allocation, design))
            source_path = folder / 'questions' / (row['id'] + '.json')
            source = {'path': source_path.relative_to(root).as_posix(),
                      'sha256': file_sha(source_path), 'bytes': source_path.stat().st_size}
            save(formal / model / 'numerical_questions' / source_path.name,
                 compact_question(row, source))
            del raw
            print('RAW_QUESTION_REVIEWED', model, row['question_index'] + 1, flush=True)
        raw = raw_summary(raw_questions, design)
        probe = read(root / FORMAL / model / 'probe.json')
        old_single = read(root / FORMAL / model / 'single_layer.json') if model == 'llama' else None
        matched = [item['matched_local_sensitivity'] for item in old_single['layers']] \
            if old_single else None
        summary = {
            'model': model, 'analysis_commit': commit, 'execution_commit': execution['commit'],
            'design_digest': digest(design), 'question_ids': [row['id'] for row in rows],
            'statistical_unit': 'question', 'validation_n': len(rows),
            'path_means': path_summary(rows),
            'budget_comparisons': budget_comparisons(rows, allocation, plan, design),
            'single_layer': layer_summary(rows, probe, design, raw, matched),
            'raw_diagnostics': raw,
            'repeats': {row['id']: {mode: {
                'n': len(runs), 'all_pairwise_consecutive_equal':
                all(run['identical_to_previous'] for run in runs[1:])}
                for mode, runs in row['precisions'].items()} for row in rows},
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in rows),
            'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in rows),
        }
        save(formal / model / 'numerics_summary.json', summary)
        rescoring = read(folder / 'rescoring.json')
        check_rescoring(root, model, rescoring)
        for name in ('rescoring.json', 'numerical_allocations.json', 'execution.json'):
            target = 'numerics_execution.json' if name == 'execution.json' else name
            save(formal / model / target, read(folder / name))
        audit_models[model] = {
            'status': 'passed', 'n': len(rows), 'path_questions_raw_recomputed': len(rows),
            'historical_output_matches': sum(
                item['same_prediction'] for row in rows
                for item in row['historical_device_comparison']['schemes'].values()),
            'historical_output_comparisons': sum(
                len(row['historical_device_comparison']['schemes']) for row in rows),
            'protected_random_configurations': 27,
            'single_layers': 32 if model == 'llama' else 2,
            'content_correct_bf16_rescored': sum(row['schemes']['bf16']['content_correct']
                                                for row in rescoring['rows']),
            'format_compliant_bf16_rescored': sum(row['schemes']['bf16']['format_compliant']
                                                 for row in rescoring['rows']),
        }
        print('NUMERICS_MODEL_REVIEWED', model, flush=True)
    sources = {path.relative_to(formal).as_posix(): file_sha(path)
               for path in sorted(formal.rglob('*.json')) if path.name != 'numerics_review.json'}
    if any((formal / name).stat().st_size > 10 * 2**20 for name in sources):
        raise RuntimeError('正式小型实体超过回传大小上限')
    save(formal / 'numerics_review.json', {
        'status': 'passed', 'date': datetime.now(timezone.utc).isoformat(),
        'analysis_commit': commit, 'design_digest': digest(design),
        'design_sha256': sha(root / DESIGN), 'models': audit_models,
        'source_files': sources, 'raw_logit_entities_verified': entities,
        'review_code_sha256': sha(Path(__file__)),
        'numeric_envelope': 'observed_three_conditions_and_repeats_not_a_universal_error_bound',
    })
    print('NUMERICS_REVIEW_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    run(parser.parse_args().commit)
