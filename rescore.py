#!/usr/bin/env python3
"""Re-score saved results through the current checkers — no model calls.

Scoring logic evolves (e.g. infra-vs-quality decoupling) independently of the
expensive model runs. This replays the saved RunResults in results/<run>.jsonl
through the up-to-date checkers and rewrites results/<run>_scores.jsonl with the
new Score (including `status`), so old runs benefit from scorer fixes for free.

Usage:
  python rescore.py results/bench_all.jsonl [more.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from telemetry.schema import RunResult
from runner import load_tasks, get_checker


def _task_index() -> dict[str, dict]:
    return {t["id"]: t for t in load_tasks()}


def rescore(path: Path, tasks: dict[str, dict]) -> None:
    scores_path = Path(str(path).replace(".jsonl", "_scores.jsonl"))
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    counts: dict[str, int] = {"pass": 0, "fail": 0, "infra_error": 0, "skipped": 0}
    for d in rows:
        task = tasks.get(d["task_id"])
        if task is None:
            counts["skipped"] += 1
            continue
        result = RunResult.from_dict(d)
        checker = get_checker(task.get("family", "ops"))
        score = checker.score(task, result)
        out.append(score.to_json())
        counts[score.status] = counts.get(score.status, 0) + 1
    scores_path.write_text("\n".join(out) + "\n")
    print(f"  {path.name}: rescored {len(out)} -> {scores_path.name}  {dict(counts)}")


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        print("usage: python rescore.py <results.jsonl> [...]", file=sys.stderr)
        sys.exit(1)
    tasks = _task_index()
    print(f"Loaded {len(tasks)} tasks; rescoring {len(files)} file(s):")
    for f in files:
        if f.exists():
            rescore(f, tasks)
        else:
            print(f"  SKIP (missing): {f}")


if __name__ == "__main__":
    main()
