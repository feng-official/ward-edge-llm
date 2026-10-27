#!/usr/bin/env python3
"""Reduce an append-only evaluation journal using the evaluator's latest-id rule."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'datasets/ward-nlu-500-v1/distillation/v4/general-v4-q6-full.samples.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/saved-samples-audit')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    final = {row['id']: row for row in rows}
    ordered = sorted(final.values(), key=lambda row: row['id'])
    tasks = sorted({row['task'] for row in ordered})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'general-v4-q6-final600.samples.jsonl'
    with output.open('w', encoding='utf-8', newline='\n') as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'source': args.input.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'raw_rows': len(rows), 'unique_question_ids': len(final), 'superseded_rows': len(rows) - len(final),
              'rule': 'Keep last record per id, identical to evaluate_general_retention.run_command previous[id] assignment.',
              'output': output.name,
              'accuracy': {task: sum(row.get('relaxed_correct', row.get('correct', False)) for row in ordered if row['task'] == task)
                           / sum(row['task'] == task for row in ordered) for task in tasks}}
    (args.output_dir / 'samples-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
