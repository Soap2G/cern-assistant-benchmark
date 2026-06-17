#!/usr/bin/env python3
"""Smoke test: validate schema, task loading, checkers, and cost model.

Does NOT call any LLM. Run this first to confirm the harness is healthy.

Usage:
  cd /eos/user/g/gguerrie/benchmark
  python smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import yaml


def test_schema() -> None:
    from telemetry.schema import RunResult, Score
    r = RunResult(
        task_id="ops_001", system="lumi", variant_id="canonical",
        answer_artifact="executable = test.sh\n+JobFlavour = \"longlunch\"\nqueue 1",
        input_tokens=1000, output_tokens=100, retrieval_tokens=None,
        latency_ms=2000, raw_response="raw", error=None,
    )
    assert r.to_json()
    r2 = RunResult.from_dict(r.to_dict())
    assert r2.task_id == "ops_001"
    print("  [OK] schema.RunResult")

    s = Score(task_id="ops_001", system="lumi", variant_id="canonical",
              passed=True, metrics={"dry_run_ok": True})
    assert s.to_json()
    print("  [OK] schema.Score")


def test_task_loading() -> None:
    from runner import load_tasks
    tasks = load_tasks()
    assert tasks, "No tasks found"
    families = {t.get("family") for t in tasks}
    print(f"  [OK] Loaded {len(tasks)} tasks: {sorted(families)}")
    for t in tasks:
        assert "id" in t, f"Missing id in {t}"
        assert "family" in t, f"Missing family in {t}"
        assert "prompt" in t, f"Missing prompt in {t}"
    print(f"  [OK] All task fields validated")


def test_ops_checker_pass() -> None:
    from telemetry.schema import RunResult
    from checkers.ops_condor import OpsCondorChecker

    task = {
        "id": "ops_001", "family": "ops",
        "gold_attrs": ["executable", "+jobflavour", "output", "error", "log"],
        "gold_attr_values": {"+jobflavour": "longlunch"},
    }
    good_artifact = (
        "executable = run_analysis.sh\n"
        "+JobFlavour = \"longlunch\"\n"
        "output = logs/run.out\n"
        "error = logs/run.err\n"
        "log = logs/run.log\n"
        "queue 1\n"
    )
    r = RunResult(
        task_id="ops_001", system="lumi", variant_id="canonical",
        answer_artifact=good_artifact,
        input_tokens=1000, output_tokens=100, retrieval_tokens=None,
        latency_ms=500, raw_response=good_artifact, error=None,
    )
    checker = OpsCondorChecker()
    score = checker.score(task, r)
    assert score.passed, f"Expected PASS, got metrics={score.metrics}"
    print("  [OK] OpsCondorChecker PASS case")


def test_ops_checker_fail() -> None:
    from telemetry.schema import RunResult
    from checkers.ops_condor import OpsCondorChecker

    task = {
        "id": "ops_001", "family": "ops",
        "gold_attrs": ["executable", "+jobflavour"],
        "gold_attr_values": {},
    }
    bad_artifact = "# missing executable\nqueue 1\n"
    r = RunResult(
        task_id="ops_001", system="lumi", variant_id="canonical",
        answer_artifact=bad_artifact,
        input_tokens=1000, output_tokens=50, retrieval_tokens=None,
        latency_ms=500, raw_response=bad_artifact, error=None,
    )
    checker = OpsCondorChecker()
    score = checker.score(task, r)
    assert not score.passed, f"Expected FAIL, got passed=True"
    print("  [OK] OpsCondorChecker FAIL case")


def test_cds_checker() -> None:
    from telemetry.schema import RunResult
    from checkers.cds_recall import CDSRecallChecker

    task = {
        "id": "cds_001", "family": "cds",
        "gold_recids": [1471451, 9999999],
        "k": 5, "threshold": 0.5,
        "staleness_marker": False,
    }
    r = RunResult(
        task_id="cds_001", system="lumi", variant_id="canonical",
        answer_artifact="[1471451, 1234567]",
        input_tokens=5000, output_tokens=30, retrieval_tokens=None,
        latency_ms=3000, raw_response="[1471451, 1234567]", error=None,
    )
    checker = CDSRecallChecker()
    score = checker.score(task, r)
    assert score.passed, f"Expected PASS: {score.metrics}"
    assert score.metrics["recall_at_k"] == 0.5
    print("  [OK] CDSRecallChecker PASS case")


def test_cost_model() -> None:
    from cost.model import load_params, rag_diff, skills_diff, breakeven_N

    p = load_params()
    N = p["queries_per_year"]
    churn = p["reindexes_per_year"]
    r = rag_diff(p, N, churn)
    s = skills_diff(p, N)
    be = breakeven_N(p, churn)
    assert r >= 0 and s >= 0
    be_str = f"{be:.0f}" if be != float("inf") else "inf"
    print(f"  [OK] cost model: RAG=${r:.2f}, Skills=${s:.2f}, breakeven_N={be_str}")


def main() -> None:
    tests = [
        test_schema,
        test_task_loading,
        test_ops_checker_pass,
        test_ops_checker_fail,
        test_cds_checker,
        test_cost_model,
    ]
    print("Running smoke tests...")
    failures = 0
    for t in tests:
        name = t.__name__
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback; traceback.print_exc()
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures}/{len(tests)} tests")
        sys.exit(1)
    else:
        print(f"ALL PASSED ({len(tests)}/{len(tests)})")


if __name__ == "__main__":
    main()
