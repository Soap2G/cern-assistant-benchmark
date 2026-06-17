"""Ops checker — validates HTCondor submit files via condor_submit -dry-run.

Pass criteria:
  1. condor_submit -dry-run exits 0
  2. All required attributes from task.gold_attrs are present in the
     submit file (case-insensitive key match)
  3. Optional task.gold_attr_values enforces specific values
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from checkers.base import Checker, infra_failure
from telemetry.schema import RunResult, Score

CONDOR_BIN = "condor_submit"


def _extract_submit_file(raw: str) -> str:
    """Pull the submit file out of raw answer text.

    Supports two formats:
      - Raw submit file (starts with 'executable' or '#')
      - Code-block wrapped: ```\\n...\\n```
    """
    # Try to find a code block first
    m = re.search(r"```[^\n]*\n(.*?)```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Otherwise assume the whole thing is a submit file
    return raw.strip()


def _parse_classads(subfile: str) -> dict[str, str]:
    """Parse key=value lines from a condor submit file."""
    attrs: dict[str, str] = {}
    for line in subfile.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            attrs[key.strip().lower()] = val.strip().strip('"')
    return attrs


def _validate_via_bindings(sanitized: str) -> tuple[bool, str] | None:
    """Validate submit-file syntax with the HTCondor Python bindings.

    `htcondor.Submit(text)` parses the full submit description without
    contacting the schedd or the (flaky) Kerberos credmon. Returns
    (passed, output) on a definitive parse result, or None if the bindings
    aren't importable (caller falls back to condor_submit).
    """
    try:
        import htcondor  # system bindings (py3.9): module name "htcondor"
    except Exception:
        try:
            import htcondor2 as htcondor  # pip wheel (htcondor >=24): "htcondor2"
        except Exception:
            return None
    try:
        sub = htcondor.Submit(sanitized)
        keys = ", ".join(sorted(k.lower() for k in dict(sub).keys()))
        return True, f"htcondor.Submit parsed OK; keys: {keys}"
    except Exception as e:
        return False, f"htcondor.Submit parse error: {e}"


def _dry_run(subfile_text: str) -> tuple[bool, str]:
    """Validate a submit file's syntax/structure.

    Primary path is the HTCondor Python bindings (`htcondor.Submit`), which
    parse the submit description deterministically and credmon-free. CERN's
    condor_submit auto-requests Kerberos credentials on every invocation, and a
    stalled credmon makes `condor_submit -dry-run` hang ~20s and bail — so the
    bindings are the robust validator here. Falls back to `condor_submit
    -dry-run` only if the bindings are unavailable.

    The executable is rewritten to /bin/true so validation never fails merely
    because the (fictional) script doesn't exist on disk.

    Returns (passed, output_text).
    """
    sanitized = re.sub(
        r"^(executable\s*=\s*).*$",
        r"\g<1>/bin/true",
        subfile_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    via_bindings = _validate_via_bindings(sanitized)
    if via_bindings is not None:
        return via_bindings

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sub", delete=False, prefix="lumi_bench_"
    ) as f:
        f.write(sanitized)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [CONDOR_BIN, "-dry-run", "/dev/null", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except FileNotFoundError:
        return False, "condor_submit not found — not on a submit host"
    except subprocess.TimeoutExpired:
        return False, "condor_submit timeout"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class OpsCondorChecker(Checker):
    def score(self, task: dict[str, Any], result: RunResult) -> Score:
        task_id = task["id"]
        system = result.system
        variant_id = result.variant_id

        # Infra failures (no artifact + adapter error) are excluded from quality.
        infra = infra_failure(result)
        if infra is not None:
            return infra
        if not result.answer_artifact.strip():
            # Empty answer with no infra error = a genuine quality miss (refusal
            # / non-answer), not an outage.
            return Score(
                task_id=task_id,
                system=system,
                variant_id=variant_id,
                passed=False,
                metrics={"error": "empty artifact"},
            )

        subfile = _extract_submit_file(result.answer_artifact)
        dry_ok, dry_out = _dry_run(subfile)
        attrs = _parse_classads(subfile)

        # Check required attrs
        gold_attrs = [a.lower() for a in task.get("gold_attrs", [])]
        missing = [a for a in gold_attrs if a not in attrs]

        # Check required values
        gold_values = {
            k.lower(): str(v).lower()
            for k, v in task.get("gold_attr_values", {}).items()
        }
        value_failures = {
            k: {"expected": v, "got": attrs.get(k, "<missing>")}
            for k, v in gold_values.items()
            if attrs.get(k, "").lower() != v
        }

        passed = dry_ok and not missing and not value_failures

        return Score(
            task_id=task_id,
            system=system,
            variant_id=variant_id,
            passed=passed,
            metrics={
                "dry_run_ok": dry_ok,
                "dry_run_output": dry_out[:500],
                "missing_attrs": missing,
                "value_failures": value_failures,
                "attrs_found": list(attrs.keys()),
                # non-blocking adapter note (e.g. token-fetch lag) if any
                **({"infra_note": result.error} if result.error else {}),
            },
        )
