"""Shared dataclasses — the contract every adapter and checker conforms to."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any
import json


def _only_known(cls, d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the dataclass doesn't declare, so older/newer JSONL rows load.

    Schema evolves over time (new token-accounting fields, Score.status); a
    record written by a different version must still round-trip without raising.
    Unknown keys are dropped; missing keys fall back to field defaults.
    """
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in known}


# Score statuses
PASS = "pass"
FAIL = "fail"
INFRA_ERROR = "infra_error"  # run failed for infrastructure reasons; excluded from quality


@dataclass
class RunResult:
    task_id: str
    system: str           # "lumi" | "accgpt" | ...
    variant_id: str       # e.g. "canonical" | "p1" | "p2"
    answer_artifact: str  # raw text: submit-file text or JSON array of recids
    input_tokens: int     # effective input tokens (reported if available, else estimate/0)
    output_tokens: int    # effective output tokens
    retrieval_tokens: int | None  # None when system can't expose it
    latency_ms: int
    raw_response: str
    error: str | None = None

    # ----- uniform token accounting (telemetry/tokens.py) -----
    # What the provider actually reported (None/0 when the API hides usage).
    input_tokens_reported: int | None = None
    output_tokens_reported: int | None = None
    # tiktoken count of the exact prompt we sent (always observable).
    client_input_tokens: int | None = None
    # "reported" | "estimated" | "mixed" | "unavailable" — provenance of the
    # effective input/output token counts above.
    tokens_source: str = "reported"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        return cls(**_only_known(cls, d))


@dataclass
class Score:
    task_id: str
    system: str
    variant_id: str
    passed: bool
    # "pass" | "fail" | "infra_error". `passed` stays for backward compatibility
    # (passed == status == "pass"); `status` is authoritative for reporting so
    # infra failures can be excluded from the quality denominator.
    status: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    # ops: {}
    # cds: {"precision_at_k": float, "recall_at_k": float, "matched": list[int]}

    def __post_init__(self) -> None:
        # Keep status and passed consistent regardless of which the caller set.
        if not self.status:
            self.status = PASS if self.passed else FAIL
        self.passed = self.status == PASS

    @property
    def is_infra_error(self) -> bool:
        return self.status == INFRA_ERROR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Score":
        return cls(**_only_known(cls, d))
