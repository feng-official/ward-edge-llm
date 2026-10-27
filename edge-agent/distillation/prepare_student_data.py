#!/usr/bin/env python3
"""Combine teacher responses with native project prompts and rebalance judgments."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, type=Path)
    p.add_argument("--native", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    teacher = read(args.teacher)
    native = read(args.native)
    rows = teacher + native
    # The held-out set contains all three judgment classes. Repeat native-format
    # reject/escalate samples so the student cannot minimize loss by always confirming.
    minority = []
    for row in native:
        if row.get("task") != "cloud_judgment":
            continue
        label = row["messages"][-1]["content"].split("|", 1)[0]
        if label in {"reject", "escalate"}:
            duplicate = json.loads(json.dumps(row, ensure_ascii=False))
            duplicate["id"] += "-balanced"
            minority.append(duplicate)
    rows.extend(minority)
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "rows": len(rows),
        "teacher_rows": len(teacher),
        "native_rows": len(native),
        "balanced_duplicates": len(minority),
        "task_counts": Counter(r.get("task", "unknown") for r in rows),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
