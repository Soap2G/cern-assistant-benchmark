"""Checker abstract base class + shared infra/quality triage."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from telemetry.schema import RunResult, Score, INFRA_ERROR


class Checker(ABC):
    @abstractmethod
    def score(self, task: dict[str, Any], result: RunResult) -> Score:
        """Apply family-specific quality gate. Never raises."""
        ...


def infra_failure(result: RunResult) -> Score | None:
    """Return an `infra_error` Score if the run failed for infrastructure
    reasons, else None so the caller proceeds to quality scoring.

    Rule: a run is an INFRA failure only when it produced no usable artifact AND
    the adapter reported an error (network/timeout/auth/crash). That is distinct
    from a *quality* miss — a present-but-wrong or empty-but-deliberate answer.
    A non-blocking error alongside a real artifact (e.g. Lumi token-fetch lag)
    is NOT an infra failure: the answer stands and is scored on quality, with the
    note carried in metrics.

    Infra failures are excluded from the quality denominator by the report, so a
    transient outage never masquerades as a model getting the answer wrong.
    """
    has_artifact = bool((result.answer_artifact or "").strip())
    if result.error and not has_artifact:
        return Score(
            task_id=result.task_id,
            system=result.system,
            variant_id=result.variant_id,
            passed=False,
            status=INFRA_ERROR,
            metrics={"error": result.error, "infra": True},
        )
    return None
