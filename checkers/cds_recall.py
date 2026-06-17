"""CDS checker — hit@k against a broadened set of valid record IDs.

The adapter answer is expected to be a JSON array of integer recids,
e.g. [1471452, 1471453].

Gold sets are BROAD: each task's gold_recids lists every canonical record that
correctly answers the query (the paper plus its CONF/PUB-note and channel
variants). A returned recid counts as correct if it is anywhere in that set.

Pass criteria: at least `min_hits` of the returned top-k recids are in the gold
set (default min_hits=1 — "did it find a relevant record"). precision@k and
recall@k are still reported for analysis.

Staleness marker: if task.staleness_marker=true, the gold set holds records that
post-date accGPT's last reindex. Lumi queries live CDS and finds them; RAG
misses them → staleness surfaces as a quality regression.
"""
from __future__ import annotations

import json
import re
from typing import Any

from checkers.base import Checker, infra_failure
from telemetry.schema import RunResult, Score


def _parse_recids(raw: str) -> list[int]:
    """Extract a list of integers from the answer artifact.

    Handles:
      - Plain JSON array: [123, 456]
      - Embedded in text: "The records are [123, 456]."
      - Newline-separated integers
    """
    # Try direct JSON parse
    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [int(x) for x in data]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON array in the text
    m = re.search(r"\[[\d\s,]+\]", text)
    if m:
        try:
            data = json.loads(m.group(0))
            return [int(x) for x in data]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fall back: extract all integers
    nums = re.findall(r"\b\d{5,8}\b", text)  # CDS recids are 7-digit numbers
    return [int(n) for n in nums]


def _precision_recall_at_k(
    returned: list[int], gold: list[int], k: int
) -> tuple[float, float]:
    returned_at_k = returned[:k]
    gold_set = set(gold)
    if not gold_set:
        return 0.0, 0.0
    matched = [r for r in returned_at_k if r in gold_set]
    precision = len(matched) / max(len(returned_at_k), 1)
    recall = len(matched) / len(gold_set)
    return precision, recall


class CDSRecallChecker(Checker):
    def score(self, task: dict[str, Any], result: RunResult) -> Score:
        task_id = task["id"]
        system = result.system
        variant_id = result.variant_id
        gold_recids: list[int] = task.get("gold_recids", [])
        k: int = task.get("k", 5)
        threshold: float = task.get("threshold", 0.8)
        min_hits: int = task.get("min_hits", 1)
        staleness = task.get("staleness_marker", False)

        # Infra failures (no artifact + adapter error) are excluded from quality;
        # a non-blocking error alongside a real artifact falls through and is
        # scored on quality (the recids stand).
        infra = infra_failure(result)
        if infra is not None:
            infra.metrics["staleness_marker"] = staleness
            return infra
        if not result.answer_artifact.strip():
            return Score(
                task_id=task_id,
                system=system,
                variant_id=variant_id,
                passed=False,
                metrics={"error": "empty artifact", "staleness_marker": staleness},
            )

        returned = _parse_recids(result.answer_artifact)
        if not gold_recids:
            # No gold set defined — can't score
            return Score(
                task_id=task_id,
                system=system,
                variant_id=variant_id,
                passed=False,
                metrics={
                    "error": "no gold_recids defined for this task",
                    "returned": returned,
                    "staleness_marker": staleness,
                },
            )

        precision, recall = _precision_recall_at_k(returned, gold_recids, k)
        matched = [r for r in returned[:k] if r in set(gold_recids)]
        passed = len(matched) >= min_hits

        return Score(
            task_id=task_id,
            system=system,
            variant_id=variant_id,
            passed=passed,
            metrics={
                "hits": len(matched),
                "min_hits": min_hits,
                "precision_at_k": round(precision, 4),
                "recall_at_k": round(recall, 4),
                "k": k,
                "threshold": threshold,
                "matched": matched,
                "returned": returned,
                "gold_recids": gold_recids,
                "staleness_marker": staleness,
                **({"infra_note": result.error} if result.error else {}),
            },
        )
