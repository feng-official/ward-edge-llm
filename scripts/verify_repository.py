#!/usr/bin/env python3
"""Offline validation of the upload package; never contacts a model or network."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '__pycache__', '.venv', 'venv', 'outputs', 'cache', 'tmp', '.pytest_cache'}
SELF_REPORTS = {'reports/package-manifest.json', 'reports/local-validation.json'}
EXPECTED_COUNTS = {
    'fall_suspected': 50, 'seizure': 50, 'fall_prediction': 50, 'nurse_call': 40,
    'bed_leave': 40, 'abnormal_posture': 40, 'long_still': 40, 'night_wandering': 35,
    'door_departure': 35, 'bedsore_risk': 35, 'environment_anomaly': 30,
    'device_fault': 30, 'node_offline': 25,
}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(part in IGNORED for part in p.relative_to(ROOT).parts))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def prompts(rows):
    return {m['content'].strip() for row in rows for m in row.get('messages', []) if m['role'] == 'user'}


def refresh_manifest():
    entries = []
    for p in files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in SELF_REPORTS:
            continue
        entries.append({'path': rel, 'size_bytes': p.stat().st_size, 'sha256': digest(p)})
    output = ROOT / 'reports/package-manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'package': 'ward-edge-llm', 'prepared_date': '2026-10-27',
                                  'weight_included': False, 'files': entries,
                                  'excluded_from_manifest': sorted(SELF_REPORTS)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def f1(rows):
    values = []
    for label in sorted({r['label_expected'] for r in rows}):
        tp = sum(r['label_expected'] == label and r['label_prediction'] == label for r in rows)
        fp = sum(r['label_expected'] != label and r['label_prediction'] == label for r in rows)
        fn = sum(r['label_expected'] == label and r['label_prediction'] != label for r in rows)
        values.append(2 * tp / max(1, 2 * tp + fp + fn))
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-manifest', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.refresh_manifest:
        refresh_manifest()
    checks = []

    def check(name, passed, detail):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    inventory = files()
    parse_errors = []
    python_count = 0
    json_count = 0
    for p in inventory:
        try:
            if p.suffix == '.py':
                ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
                python_count += 1
            elif p.suffix == '.json':
                json.loads(p.read_text(encoding='utf-8-sig'))
                json_count += 1
            elif p.suffix == '.jsonl':
                read_jsonl(p)
                json_count += 1
        except (SyntaxError, UnicodeError, ValueError) as exc:
            parse_errors.append(f'{p.relative_to(ROOT)}: {type(exc).__name__}')
    check('python_and_json_parse', not parse_errors,
          {'python_files': python_count, 'json_jsonl_files': json_count, 'errors': parse_errors})

    base = ROOT / 'datasets/ward-nlu-500-v1'
    rows = read_jsonl(base / 'nlu-dataset-v1.0.0.jsonl')
    train = read_jsonl(base / 'train-v1.0.0.jsonl')
    test = read_jsonl(base / 'test-v1.0.0.jsonl')
    check('dataset_counts', len(rows) == 500 and len(train) == 400 and len(test) == 100,
          {'total': len(rows), 'train': len(train), 'test': len(test)})
    check('event_distribution', dict(Counter(r['event_type'] for r in rows)) == EXPECTED_COUNTS,
          dict(Counter(r['event_type'] for r in rows)))
    check('unique_ids_and_split_files', len({r['id'] for r in rows}) == len(rows)
          and train == [r for r in rows if r['split'] == 'train']
          and test == [r for r in rows if r['split'] == 'test'], 'Canonical source and split files must agree.')
    overlaps = {key: len({(r['context']['scenario_family'] if key == 'scenario_family' else r[key]) for r in train}
                         & {(r['context']['scenario_family'] if key == 'scenario_family' else r[key]) for r in test})
                for key in ['id', 'description', 'scenario_family']}
    check('train_test_isolation', not any(overlaps.values()), overlaps)
    sft_train = read_jsonl(base / 'qwen-sft-train-v1.0.0.jsonl')
    sft_test = read_jsonl(base / 'qwen-sft-test-v1.0.0.jsonl')
    shape_ok = all([m.get('role') for m in r.get('messages', [])] == ['system', 'user', 'assistant']
                   and all(isinstance(m.get('content'), str) and m['content'].strip() for m in r['messages'])
                   for r in sft_train + sft_test)
    check('sft_count_and_shape', len(sft_train) == 800 and len(sft_test) == 200 and shape_ok,
          {'train': len(sft_train), 'test': len(sft_test)})
    mixed = read_jsonl(base / 'distillation/v4/mixed-train-v4.1.0.jsonl')
    mixed_manifest = json.loads((base / 'distillation/v4/manifest-v4.1.0.json').read_text(encoding='utf-8'))
    check('mixed_training_count_and_hash', len(mixed) == 5000
          and digest(base / 'distillation/v4/mixed-train-v4.1.0.jsonl') == mixed_manifest['output_sha256'],
          {'rows': len(mixed), 'historical_manifest_hash_matches': digest(base / 'distillation/v4/mixed-train-v4.1.0.jsonl') == mixed_manifest['output_sha256']})
    new_test = read_jsonl(base / 'evaluation/ward-task-50-v1/ward-task-50-v1.jsonl')
    student = read_jsonl(base / 'distillation/student-train-v2.jsonl')
    duplicate_counts = {
        'sft_train_vs_test': len(prompts(sft_train) & prompts(sft_test)),
        'mixed_vs_sft_test': len(prompts(mixed) & prompts(sft_test)),
        'ward50_vs_saved_train_test': len(prompts(new_test) & prompts(sft_train + sft_test + student + mixed)),
    }
    check('exact_prompt_overlap', not any(duplicate_counts.values()), duplicate_counts)
    test_ids = {r['id'] for r in test}
    teachers = read_jsonl(base / 'distillation/teacher-train-v1.jsonl')
    teacher_overlap = set()
    for row in teachers:
        teacher_overlap |= {x.strip() for x in row.get('source_record_id', '').split(',')} & test_ids
    check('teacher_sources_exclude_heldout_ids', not teacher_overlap, {'overlap_count': len(teacher_overlap)})

    domain_files = [base / 'distillation/v4/domain-v4-q6-full.json', base / 'evaluation/ward-task-50-v1/results-v4-q6.json']
    for path in domain_files:
        report = json.loads(path.read_text(encoding='utf-8'))
        clouds = [r for r in report['samples'] if r['task'] == 'cloud_judgment']
        events = [r for r in report['samples'] if r['task'] == 'event_enhancement']
        recomputed = {'cloud_judgment_accuracy': sum(r['label_expected'] == r['label_prediction'] for r in clouds) / len(clouds),
                      'cloud_judgment_macro_f1': f1(clouds),
                      'event_urgency_accuracy': sum(r['urgency_expected'] == r['urgency_prediction'] for r in events) / len(events),
                      'format_valid_rate': sum(r['format_valid'] for r in report['samples']) / len(report['samples'])}
        check(f'historical_metrics/{path.name}', all(math.isclose(v, report['metrics'][k], abs_tol=1e-8) for k, v in recomputed.items()), recomputed)

    general = json.loads((base / 'distillation/v4/general-v4-q6-full.json').read_text(encoding='utf-8'))
    journal = read_jsonl(base / 'distillation/v4/general-v4-q6-full.samples.jsonl')
    # The original run_command accumulates previous[id] = item; latest record wins.
    raw = list({row['id']: row for row in journal}.values())
    general_recomputed = {task: sum(r.get('relaxed_correct', r.get('correct', False)) for r in raw if r['task'] == task)
                          / sum(r['task'] == task for r in raw) for task in ['gsm8k', 'cruxeval_o', 'bbh']}
    check('historical_general_raw_metrics', len(raw) == 600 and all(math.isclose(value, general['task_metrics'][task]['semantic_accuracy'], abs_tol=1e-8)
          for task, value in general_recomputed.items()),
          {'raw_journal_rows': len(journal), 'unique_questions': len(raw), 'latest_record_per_id_accuracy': general_recomputed})

    log = read_jsonl(ROOT / 'reports/training-v4/training_log.jsonl')
    check('training_log', len(log) == 313 and [r['update'] for r in log] == list(range(1, 314)), {'optimizer_updates': len(log)})
    spec = importlib.util.spec_from_file_location('ward_cli', ROOT / 'scripts/ward_cli.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    demo = (ROOT / 'examples/event.json').read_text(encoding='utf-8')
    built = {}
    for task in cli.TASKS:
        source = demo if task in cli.EVENT_TASKS else 'B01于14:32坐起，事件尚待确认。'
        messages = cli.messages_for(task, source)
        built[task] = len(messages) == 2 and all(m['content'].strip() for m in messages)
    check('seven_task_prompt_builder', all(built.values()), built)

    suspicious = []
    secret_patterns = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                       r'\bgh[pousr]_[A-Za-z0-9]{24,}\b', r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
                       r'\bsk-[A-Za-z0-9_-]{24,}\b', r'\bhf_[A-Za-z0-9]{25,}\b']
    for p in inventory:
        if p.name == '.env' or (p.name.startswith('.env.') and p.name != '.env.example') or p.suffix in {'.pem', '.key'} or p.name == 'api_key':
            suspicious.append({'file': p.relative_to(ROOT).as_posix(), 'type': 'credential_file'})
        if p.suffix in {'.py', '.json', '.jsonl', '.md', '.txt', '.example', '.yml'}:
            content = p.read_text(encoding='utf-8-sig')
            if any(re.search(pattern, content) for pattern in secret_patterns):
                suspicious.append({'file': p.relative_to(ROOT).as_posix(), 'type': 'token_pattern'})
    check('credential_pattern_scan', not suspicious, {'findings': suspicious})

    manifest_path = ROOT / 'reports/package-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        mismatches = [entry['path'] for entry in manifest['files']
                      if not (ROOT / entry['path']).is_file() or digest(ROOT / entry['path']) != entry['sha256']]
        tracked = {p.relative_to(ROOT).as_posix() for p in inventory} - SELF_REPORTS
        expected = {entry['path'] for entry in manifest['files']}
        check('package_manifest_hashes', not mismatches and tracked == expected,
              {'files': len(expected), 'mismatches': mismatches, 'inventory_difference': sorted(tracked ^ expected)})
    else:
        check('package_manifest_hashes', False, 'Missing reports/package-manifest.json')

    report = {'kind': 'offline_repository_validation_not_model_inference',
              'passed': all(row['passed'] for row in checks), 'checks': checks,
              'known_missing_assets': ['final GGUF', 'LoRA weights', 'historical frozen benchmark.jsonl'],
              'cloud_server_contacted': False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in checks:
        print(f"{'PASS' if row['passed'] else 'FAIL'} {row['name']}")
    print(f"{sum(r['passed'] for r in checks)}/{len(checks)} offline checks passed")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
