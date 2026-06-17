#!/usr/bin/env python3
"""Benchmark runner — task × system × variant matrix.

Writes one JSONL record per (task, system, variant) to results/<run_id>.jsonl
and a corresponding scores JSONL to results/<run_id>_scores.jsonl.

Usage:
  # Run Lumi on all ops tasks (canonical + paraphrases):
  python runner.py --system lumi --family ops

  # Run Lumi on a single task:
  python runner.py --system lumi --task ops_001

  # Replay pre-recorded accGPT results:
  python runner.py --system accgpt --family ops

  # Record new accGPT results interactively:
  python runner.py --system accgpt --family cds --record

  # Dry-run: check tasks load and schema validates, don't call any LLM:
  python runner.py --dry-run

Environment:
  LUMI_MODEL       override the Lumi model (default: litellm/gpt-4.1)
  LUMI_CONFIG_DIR  override lumi-assistant config path
  ACCGPT_RECORD    set to 1 to enter accGPT record mode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Allow imports from the benchmark package root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from telemetry.schema import RunResult, Score, INFRA_ERROR
from checkers.ops_condor import OpsCondorChecker
from checkers.cds_recall import CDSRecallChecker
from systems import make_adapter, repeats_for, system_names

TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_tasks(family: str | None = None, task_id: str | None = None) -> list[dict]:
    tasks = []
    for yaml_path in sorted(TASKS_DIR.rglob("*.yaml")):
        with open(yaml_path) as f:
            task = yaml.safe_load(f)
        if family and task.get("family") != family:
            continue
        if task_id and task.get("id") != task_id:
            continue
        tasks.append(task)
    return tasks


def get_variants(task: dict) -> list[tuple[str, str]]:
    """Return list of (variant_id, prompt_text) pairs."""
    variants = [("canonical", task["prompt"].strip())]
    for p in task.get("paraphrases", []):
        variants.append((p["id"], p["text"].strip()))
    return variants


def get_checker(family: str):
    if family == "ops":
        return OpsCondorChecker()
    elif family == "cds":
        return CDSRecallChecker()
    else:
        raise ValueError(f"Unknown family: {family}")


def get_adapter(system: str, record: bool = False):
    """Instantiate an adapter from the systems.yaml registry."""
    return make_adapter(system, record=record)


def run_benchmark(
    systems: list[str],
    tasks: list[dict],
    record: bool = False,
    dry_run: bool = False,
    repeats_override: dict[str, int] | None = None,
    run_id: str | None = None,
) -> tuple[list[RunResult], list[Score]]:
    repeats_override = repeats_override or {}
    if run_id is None:
        run_id = f"run_{int(time.time())}"

    results_path = RESULTS_DIR / f"{run_id}.jsonl"
    scores_path = RESULTS_DIR / f"{run_id}_scores.jsonl"

    # Truncate any prior output for this run_id so re-runs are idempotent
    # (records below are streamed in append mode). Without this, re-running
    # the same slice silently doubles the rows in the file.
    if not dry_run:
        open(results_path, "w").close()
        open(scores_path, "w").close()

    all_results: list[RunResult] = []
    all_scores: list[Score] = []

    for system in systems:
        adapter = get_adapter(system, record=record) if not dry_run else None
        repeats = repeats_override.get(system) or repeats_for(system)

        for task in tasks:
            family = task.get("family", "ops")
            checker = get_checker(family)
            variants = get_variants(task)

            for variant_id, prompt in variants:
                for rep in range(repeats):
                    vid = variant_id if repeats == 1 else f"{variant_id}_r{rep}"
                    print(
                        f"  [{system}] {task['id']} / {variant_id}"
                        + (f" rep{rep}" if repeats > 1 else ""),
                        flush=True,
                    )

                    if dry_run:
                        print(f"    [dry-run] would call adapter.run()")
                        continue

                    result = adapter.run(task, vid, prompt)
                    score = checker.score(task, result)

                    all_results.append(result)
                    all_scores.append(score)

                    # Stream to JSONL immediately
                    with open(results_path, "a") as f:
                        f.write(result.to_json() + "\n")
                    with open(scores_path, "a") as f:
                        f.write(score.to_json() + "\n")

                    status = {"pass": "PASS", "fail": "FAIL"}.get(
                        score.status, score.status.upper()
                    )
                    tokens = (f"{result.input_tokens}in/{result.output_tokens}out"
                              f" [{result.tokens_source}]")
                    print(f"    {status}  tokens={tokens}  latency={result.latency_ms}ms")
                    if result.error:
                        print(f"    NOTE: {result.error}")

    print(f"\nResults → {results_path}")
    print(f"Scores  → {scores_path}")
    return all_results, all_scores


def print_summary(scores: list[Score]) -> None:
    from collections import defaultdict
    from telemetry.stats import summarize

    by_family_system: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in scores:
        # Infer family from task_id prefix
        family = s.task_id.split("_")[0]
        by_family_system[(family, s.system)].append(s.status)

    print("\n" + "=" * 64)
    print("SUMMARY (pass rate [95% CI] per family per system; infra excluded)")
    print("=" * 64)
    for (family, system), statuses in sorted(by_family_system.items()):
        st = summarize(statuses)
        infra = f"  (+{st.n_infra} infra)" if st.n_infra else ""
        print(f"  {family:4s} / {system:8s}  {st.n_pass}/{st.n_scored}  "
              f"{st.pct()}{infra}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lumi vs accGPT benchmark runner")
    registered = system_names()
    parser.add_argument("--system", nargs="+", default=registered[:1],
                        choices=[*registered, "both", "all"],
                        help="System(s) to benchmark (from systems.yaml); "
                             "'all'/'both' = every registered system")
    parser.add_argument("--family", choices=["ops", "cds"],
                        help="Filter to one task family")
    parser.add_argument("--task", help="Filter to a single task ID")
    parser.add_argument("--record", action="store_true",
                        help="accGPT: enter interactive record mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate tasks/schema without calling any LLM")
    parser.add_argument("--repeats", default=None,
                        help="Override repeats, e.g. 'lumi=1,accgpt=3' "
                             "(default: per-system value from systems.yaml)")
    parser.add_argument("--run-id", default=None,
                        help="Override run ID (default: run_<timestamp>)")
    parser.add_argument("--model", default=None,
                        help="Override Lumi model (e.g. litellm/gpt-4.1)")
    args = parser.parse_args()

    if args.model:
        os.environ["LUMI_MODEL"] = args.model

    if "all" in args.system or "both" in args.system:
        systems = system_names()
    else:
        systems = args.system

    repeats_override: dict[str, int] = {}
    if args.repeats:
        for kv in args.repeats.split(","):
            k, _, v = kv.partition("=")
            if k.strip() and v.strip():
                repeats_override[k.strip()] = int(v)

    tasks = load_tasks(family=args.family, task_id=args.task)
    if not tasks:
        print("No tasks found matching filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Benchmark: {len(tasks)} tasks × {len(systems)} system(s)")
    print(f"Systems: {systems}")
    if args.family:
        print(f"Family filter: {args.family}")
    if args.task:
        print(f"Task filter: {args.task}")
    print()

    results, scores = run_benchmark(
        systems=systems,
        tasks=tasks,
        record=args.record,
        dry_run=args.dry_run,
        repeats_override=repeats_override,
        run_id=args.run_id,
    )

    if scores:
        print_summary(scores)


if __name__ == "__main__":
    main()
