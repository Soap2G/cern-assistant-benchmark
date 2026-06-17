"""Lumi adapter — headless opencode invocation.

Flow:
  1. Build full prompt (task prompt + artifact extraction instruction)
  2. Invoke `opencode run --title <unique-tag> <prompt>` with OPENCODE_CONFIG_DIR set
  3. Find OUR session by its unique title tag (concurrency-safe — no 'most recent'
     guessing, which races with flush timing and any parallel opencode calls)
  4. Export the session and parse input/output token counts (retry until flushed)
  5. Return RunResult with answer_artifact = stripped model output
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from telemetry.schema import RunResult
from telemetry import tokens

LUMI_CONFIG_DIR = os.environ.get(
    "LUMI_CONFIG_DIR",
    "/afs/cern.ch/user/g/gguerrie/lumi-assistant/config",
)
LUMI_MODEL = os.environ.get("LUMI_MODEL", "litellm/gpt-4.1")
LUMI_KEY_FILE = os.environ.get(
    "LUMI_KEY_FILE", "/eos/user/g/gguerrie/lumi_assistant/key.txt"
)
OPENCODE_BIN = os.environ.get(
    "OPENCODE_BIN", "/cvmfs/sw.escape.eu/lumi/latest/bin/opencode"
)
LUMI_TIMEOUT = int(os.environ.get("LUMI_TIMEOUT", "300"))

# Artifact extraction suffixes appended to the raw prompt
_ARTIFACT_SUFFIX = {
    "ops": (
        "\n\nIMPORTANT: Output ONLY the complete HTCondor submit file as a "
        "plain text block (no markdown fences, no explanation). "
        "Start directly with the first key=value line."
    ),
    "cds": (
        "\n\nIMPORTANT: Output ONLY a JSON array of integer CDS record IDs "
        "that match the query, e.g. [1234567, 2345678]. "
        "No explanations, no markdown."
    ),
}


def _load_key() -> str:
    path = Path(LUMI_KEY_FILE)
    if path.exists():
        return path.read_text().strip()
    return os.environ.get("LITELLM_API_KEY", "")


# Dedicated local data dir for the benchmark's opencode sessions. Keeps the
# sqlite DB on fast local disk (not AFS) and isolated from the user's polluted
# session history, so `export` reads are consistent.
LUMI_DATA_HOME = os.environ.get("LUMI_DATA_HOME", "/tmp/lumi-bench-data")


def _opencode_env() -> dict:
    env = os.environ.copy()
    env["LITELLM_API_KEY"] = _load_key()
    env["OPENCODE_CONFIG_DIR"] = LUMI_CONFIG_DIR
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
    env["XDG_DATA_HOME"] = LUMI_DATA_HOME
    os.makedirs(LUMI_DATA_HOME, exist_ok=True)
    return env


def _list_sessions() -> list[tuple[str, str]]:
    """Return (session_id, title) pairs from `opencode session list`.

    The list columns are: Session ID | Title | Updated. We slice by the fixed
    Session-ID width so multi-word titles stay intact.
    """
    result = subprocess.run(
        [OPENCODE_BIN, "session", "list"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_opencode_env(),
        cwd="/tmp",
    )
    pairs = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("ses_"):
            continue
        sid = stripped.split()[0]
        # Title is the remainder after the id, minus the trailing 'Updated' column.
        rest = stripped[len(sid):].strip()
        pairs.append((sid, rest))
    return pairs


def _run_opencode(prompt: str, title_tag: str) -> tuple[str, int]:
    """Run opencode headlessly with a unique session title.

    The title_tag uniquely identifies the session this invocation creates, so
    token accounting is robust to flush-timing races and any concurrent opencode
    calls. Returns (stdout, elapsed_ms).
    """
    t0 = time.monotonic()
    result = subprocess.run(
        [OPENCODE_BIN, "run", "--title", title_tag, "--model", LUMI_MODEL, prompt],
        capture_output=True,
        text=True,
        timeout=LUMI_TIMEOUT,
        env=_opencode_env(),
        cwd="/tmp",
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    # stdout = model answer; stderr = opencode migration/header noise
    return result.stdout, elapsed_ms


def _export_session(session_id: str) -> dict:
    result = subprocess.run(
        [OPENCODE_BIN, "export", session_id],
        capture_output=True,
        text=True,
        timeout=60,
        env=_opencode_env(),
        cwd="/tmp",
    )
    return _parse_export_json(result.stdout)


def _parse_export_json(stdout: str) -> dict:
    """Parse export stdout, tolerating a non-JSON preamble.

    `opencode export` can prepend noise (e.g. the one-time DB-migration banner)
    before the JSON object. Try a direct parse, then fall back to slicing from
    the first '{' to the last '}'.
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(stdout[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def _resolve_session_by_title(title_tag: str) -> str | None:
    """Find the session whose title contains our unique tag. Retry until it appears."""
    for _ in range(6):
        for sid, title in _list_sessions():
            if title_tag in title:
                return sid
        time.sleep(1.0)
    return None


def _fetch_tokens_with_retry(session_id: str) -> tuple[int, int]:
    """Export the session and parse tokens, retrying until they flush to non-zero.

    The token metadata is written shortly after `opencode run` returns; an
    immediate export reads zeros. Retry with a fixed 2s cadence (≈30s budget),
    which covers the flush lag of large tool-heavy sessions.
    """
    for attempt in range(15):
        data = _export_session(session_id)
        inp, out = _parse_tokens(data)
        if inp > 0 or out > 0:
            return inp, out
        time.sleep(2.0)
    return 0, 0


def _parse_tokens(session_data: dict) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from session export."""
    input_tokens = 0
    output_tokens = 0
    for msg in session_data.get("messages", []):
        info = msg.get("info", {})
        if info.get("role") == "assistant":
            tokens = info.get("tokens", {})
            input_tokens += tokens.get("input", 0)
            output_tokens += tokens.get("output", 0)
    return input_tokens, output_tokens


def _extract_answer(raw: str, family: str) -> str:
    """Strip ANSI codes and opencode's "> build · model" header from stdout."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    lines = clean.splitlines()
    # The first non-empty line is typically "> build · gpt-4.1" — drop it
    content_lines = []
    skipping_header = True
    for line in lines:
        stripped = line.strip()
        if skipping_header:
            if not stripped or stripped.startswith(">") or stripped.startswith("█"):
                continue
            skipping_header = False
        content_lines.append(line)
    return "\n".join(content_lines).strip()


class LumiAdapter:
    def __init__(self, record: bool = False, model: str | None = None, **_ignored):
        # `record` and extra params are accepted for a uniform registry-driven
        # constructor signature; Lumi has no record/replay mode.
        global LUMI_MODEL
        if model:
            LUMI_MODEL = model

    def run(self, task: dict[str, Any], variant_id: str, prompt: str) -> RunResult:
        family = task.get("family", "ops")
        full_prompt = prompt + _ARTIFACT_SUFFIX.get(family, "")
        error = None
        raw = ""
        elapsed_ms = 0
        input_tokens = 0
        output_tokens = 0
        title_tag = f"bench-{uuid.uuid4().hex[:10]}"

        try:
            raw, elapsed_ms = _run_opencode(full_prompt, title_tag)
        except subprocess.TimeoutExpired:
            error = f"opencode timeout after {LUMI_TIMEOUT}s"
        except Exception as e:
            error = str(e)

        answer = _extract_answer(raw, family) if not error else ""

        # Fetch token counts: resolve the session WE created (by its unique title
        # tag), then retry the export until the token metadata flushes (it lags
        # `run` by a moment). This is COST measurement and is non-blocking: if it
        # fails we still return the answer and let the token layer mark input as
        # unavailable (the report excludes it from the median, not from quality).
        token_note = None
        if not error:
            try:
                sid = _resolve_session_by_title(title_tag)
                if sid:
                    input_tokens, output_tokens = _fetch_tokens_with_retry(sid)
                if input_tokens == 0 and output_tokens == 0:
                    token_note = "token-fetch: tokens stayed 0 after retries"
            except Exception as e:
                token_note = f"token-fetch: {e}"

        # Uniform accounting. Lumi's real input is opencode's processed context,
        # which only the provider can report — the bare prompt is not a valid
        # proxy, so input is NOT estimated from it (stays unavailable on a miss).
        acct = tokens.resolve(
            reported_input=input_tokens or None,
            reported_output=output_tokens or None,
            prompt=full_prompt,
            answer=answer,
            estimate_input_from_prompt=False,
        )

        return RunResult(
            task_id=task["id"],
            system="lumi",
            variant_id=variant_id,
            answer_artifact=answer,
            retrieval_tokens=None,
            latency_ms=elapsed_ms,
            raw_response=raw,
            # `error` is reserved for blocking failures (no answer). A token-fetch
            # miss only rides along as a non-blocking note when an answer exists;
            # an empty answer is left to score as a plain quality miss, never infra.
            error=error or (token_note if answer.strip() else None),
            **acct,
        )
