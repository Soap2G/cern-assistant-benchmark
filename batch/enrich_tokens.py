#!/usr/bin/env python3
"""Post-hoc token enrichment for benchmark result slices (tiktoken, o200k_base).

Why: accGPT's API always returns usage=0, and a few long Lumi runs fail token
extraction. Answers (output) are fully observable, so we tokenize them directly.
Input is subtler:

  * Lumi  — opencode already reports the REAL processed input (full agent
            context: system prompt + skills + tool results). We KEEP that
            measured value; we never overwrite it. The few holes (0) are left
            as 0 so the report's median excludes them rather than being biased.
  * accGPT — the API hides the RAG-retrieved context. We can only observe the
            prompt we send (~tens of tokens). We record that as
            `client_input_tokens` for the Pareto plot, and ALSO put it in
            `input_tokens` so the point isn't pinned at x=0. The cost MODEL must
            NOT treat this as accGPT's processed context (its real context is
            the hidden retrieved chunks, ~ctx_tokens_rag); report.py is guarded
            so a small observed accGPT median can't override that estimate.

Usage:
  python batch/enrich_tokens.py results/bench_lumi_ops.jsonl [more.jsonl ...]
  python batch/enrich_tokens.py --all      # enrich the 4 standard slices
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
TASKS = ROOT / "tasks"

from telemetry import tokens  # single tokenizer + resolution policy

# accGPT appends a family-specific artifact instruction to every prompt; mirror
# it so the reconstructed client input matches what was actually sent.
ARTIFACT_SUFFIX = {
    "ops": (
        "\n\nIMPORTANT: Output ONLY the complete HTCondor submit file as "
        "plain text (no markdown fences, no explanation). "
        "Start directly with the first key=value line."
    ),
    "cds": (
        "\n\nIMPORTANT: Output ONLY a JSON array of integer CDS record IDs "
        "that match the query, e.g. [1234567, 2345678]. "
        "No explanations, no markdown."
    ),
}


def _load_prompts() -> dict[tuple[str, str], tuple[str, str]]:
    """Map (task_id, variant_base) -> (prompt_text, family)."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for yml in TASKS.glob("*/*.y*ml"):
        t = yaml.safe_load(yml.read_text())
        if not isinstance(t, dict) or "id" not in t:
            continue
        tid = t["id"]
        fam = t.get("family", "ops")
        out[(tid, "canonical")] = (t["prompt"].strip(), fam)
        for p in t.get("paraphrases", []):
            out[(tid, p["id"])] = (p["text"].strip(), fam)
    return out


def _variant_base(variant_id: str) -> str:
    """'p1_r2' -> 'p1', 'canonical' -> 'canonical'."""
    import re
    return re.sub(r"_r\d+$", "", variant_id)


def enrich_file(path: Path, prompts: dict[tuple[str, str], tuple[str, str]]) -> None:
    """Backfill token accounting on an existing slice using the SAME policy the
    adapters apply live (telemetry.tokens.resolve), so re-running the benchmark
    and back-filling old data give identical numbers."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    changed = 0
    for r in rows:
        system = r["system"]
        answer = r.get("answer_artifact") or r.get("raw_response") or ""

        key = (r["task_id"], _variant_base(r["variant_id"]))
        prompt_text, fam = prompts.get(key, ("", r.get("family", "ops")))
        sent = prompt_text + (ARTIFACT_SUFFIX.get(fam, "") if system == "accgpt" else "")

        # Treat the slice's stored counts as the provider-reported values (old
        # adapters wrote reported usage straight into input/output_tokens).
        rep_in = r.get("input_tokens_reported")
        if rep_in is None:
            rep_in = r.get("input_tokens") or None
        rep_out = r.get("output_tokens_reported")
        if rep_out is None:
            rep_out = r.get("output_tokens") or None

        acct = tokens.resolve(
            reported_input=rep_in,
            reported_output=rep_out,
            prompt=sent,
            answer=answer,
            estimate_input_from_prompt=(system == "accgpt"),
        )
        before = (r.get("input_tokens"), r.get("output_tokens"), r.get("tokens_source"))
        r.update(acct)
        if (r["input_tokens"], r["output_tokens"], r["tokens_source"]) != before:
            changed += 1

    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  {path.name}: {len(rows)} rows, {changed} updated")


def main() -> None:
    args = sys.argv[1:]
    if "--all" in args or not args:
        files = [
            RESULTS / f"bench_{sys_}_{fam}.jsonl"
            for sys_ in ("lumi", "accgpt")
            for fam in ("ops", "cds")
        ]
    else:
        files = [Path(a) for a in args]

    prompts = _load_prompts()
    print(f"Loaded {len(prompts)} (task, variant) prompts; enriching {len(files)} slice(s):")
    for f in files:
        if f.exists():
            enrich_file(f, prompts)
        else:
            print(f"  SKIP (missing): {f}")


if __name__ == "__main__":
    main()
