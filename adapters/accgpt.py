"""accGPT adapter — OpenWebUI API or manual record/replay fallback.

Modes (in priority order):

  API MODE (default):
    Sends the prompt to the accGPT OpenWebUI API and captures the response.
    Uses /api/v1/chat/completions (NOT /openai/chat/completions which is blocked).
    Note: accGPT returns usage tokens as 0 — token counts will always be 0.

  RECORD MODE (--record flag or env ACCGPT_RECORD=1):
    Prints the prompt so the user can paste it into the accGPT web UI,
    reads the answer from stdin, and saves to
    results/accgpt_manual/{task_id}__{variant_id}.json.

  REPLAY MODE (fallback if API probe fails):
    Reads from results/accgpt_manual/{task_id}__{variant_id}.json.
    Returns an error RunResult if the file doesn't exist yet.

Environment variables:
  ACCGPT_API_URL  — OpenWebUI base URL (default: https://accgpt-ui.app.cern.ch)
  ACCGPT_API_KEY  — Bearer token. If unset, read from ACCGPT_KEY_FILE.
  ACCGPT_KEY_FILE — Path to a file holding the bearer token (kept out of git).
  ACCGPT_MODEL    — Model ID as shown in OpenWebUI (default: accgpt)
  ACCGPT_RECORD   — Set to 1 to force record mode
  ACCGPT_MANUAL_DIR — Override directory for manual recordings
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from telemetry.schema import RunResult
from telemetry import tokens

ACCGPT_API_URL = os.environ.get("ACCGPT_API_URL", "https://accgpt-ui.app.cern.ch")


def _load_api_key() -> str:
    """Bearer token from ACCGPT_API_KEY, else a key file (kept out of git)."""
    key = os.environ.get("ACCGPT_API_KEY")
    if key:
        return key.strip()
    key_file = os.environ.get(
        "ACCGPT_KEY_FILE", "/eos/user/g/gguerrie/lumi_assistant/accgpt_key.txt"
    )
    try:
        return Path(key_file).read_text().strip()
    except OSError:
        return ""


ACCGPT_API_KEY = _load_api_key()
ACCGPT_MODEL = os.environ.get("ACCGPT_MODEL", "accgpt")
ACCGPT_COMPLETIONS_PATH = "/api/v1/chat/completions"
ACCGPT_MODELS_PATH = "/api/v1/models"
ACCGPT_TIMEOUT = int(os.environ.get("ACCGPT_TIMEOUT", "120"))
# accgpt-ui is load-balanced across replicas; some intermittently reject the
# key with 403 "Use of API key is not enabled in the environment." Retrying
# lands on a healthy replica, so both the probe and each call retry generously.
ACCGPT_MAX_RETRIES = int(os.environ.get("ACCGPT_MAX_RETRIES", "6"))
ACCGPT_RETRY_SLEEP = float(os.environ.get("ACCGPT_RETRY_SLEEP", "2.0"))
_TRANSIENT_403 = "use of api key is not enabled"

MANUAL_DIR = Path(os.environ.get(
    "ACCGPT_MANUAL_DIR",
    "/eos/user/g/gguerrie/benchmark/results/accgpt_manual",
))

_ARTIFACT_SUFFIX = {
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


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ACCGPT_API_KEY}"}


def _strip_sources(text: str) -> str:
    """Remove accGPT's trailing '### Sources' section from the answer."""
    import re
    # Strip everything from "### Sources" or "*No relevant documents*" preamble onward
    text = re.sub(r"\n*\*No relevant documents.*?\*\n*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n*\*Hopefully.*?\*\n*", "", text, flags=re.DOTALL)
    # Remove trailing sources block
    text = re.sub(r"\n*#{1,3}\s*Sources\b.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _record_path(task_id: str, variant_id: str) -> Path:
    return MANUAL_DIR / f"{task_id}__{variant_id}.json"


def _probe_api() -> bool:
    """Return True if the API is reachable and key auth is enabled.

    Retries several times: the first TLS handshake on CERN's network can time
    out, and the load balancer intermittently routes to a replica that 403s
    with "Use of API key is not enabled". A retry lands on a healthy replica.
    """
    for attempt in range(ACCGPT_MAX_RETRIES):
        try:
            r = requests.get(
                f"{ACCGPT_API_URL}{ACCGPT_MODELS_PATH}",
                headers=_headers(),
                timeout=20,
            )
            if r.ok and "data" in r.json():
                return True
        except Exception:
            pass
        if attempt < ACCGPT_MAX_RETRIES - 1:
            time.sleep(ACCGPT_RETRY_SLEEP)
    return False


def _list_models() -> list[str]:
    try:
        r = requests.get(
            f"{ACCGPT_API_URL}{ACCGPT_MODELS_PATH}",
            headers=_headers(),
            timeout=15,
        )
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


def _post_once(prompt: str) -> tuple[str, int, int, int]:
    """Single POST to accGPT completions. Raises on HTTP error."""
    t0 = time.monotonic()
    r = requests.post(
        f"{ACCGPT_API_URL}{ACCGPT_COMPLETIONS_PATH}",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "model": ACCGPT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=ACCGPT_TIMEOUT,
    )
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    r.raise_for_status()
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return answer, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), elapsed_ms


def _is_transient(exc: Exception) -> bool:
    """True if the error is a per-replica flake worth retrying."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        if exc.response.status_code == 403 and _TRANSIENT_403 in exc.response.text.lower():
            return True
        # 502/503/504 from the LB are also transient.
        return exc.response.status_code in (429, 500, 502, 503, 504)
    # Connection resets / TLS warmup / timeouts.
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _call_api(prompt: str) -> tuple[str, int, int, int]:
    """POST to accGPT with retries over the load balancer's transient errors.

    Returns (answer, input_tokens, output_tokens, latency_ms).
    accGPT always returns usage tokens as 0 — that is expected behaviour.
    Raises the last exception if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(ACCGPT_MAX_RETRIES):
        try:
            return _post_once(prompt)
        except Exception as e:  # noqa: BLE001 — re-raised below if non-transient/exhausted
            last_exc = e
            if not _is_transient(e) or attempt == ACCGPT_MAX_RETRIES - 1:
                raise
            time.sleep(ACCGPT_RETRY_SLEEP)
    assert last_exc is not None
    raise last_exc


class AccGPTAdapter:
    def __init__(self, record: bool = False, model: str | None = None, **_ignored):
        self.record = record or bool(os.environ.get("ACCGPT_RECORD"))
        if model:
            global ACCGPT_MODEL
            ACCGPT_MODEL = model
        self._api_available: bool | None = None

    def _use_api(self) -> bool:
        if self._api_available is None:
            self._api_available = _probe_api()
            if self._api_available:
                print("[accgpt] API mode active", flush=True)
            else:
                print("[accgpt] API probe failed — falling back to record/replay", flush=True)
        return self._api_available

    def run(self, task: dict[str, Any], variant_id: str, prompt: str) -> RunResult:
        task_id = task["id"]
        family = task.get("family", "ops")
        full_prompt = prompt + _ARTIFACT_SUFFIX.get(family, "")

        if not self.record and self._use_api():
            return self._run_api(task_id, variant_id, full_prompt)

        path = _record_path(task_id, variant_id)
        if self.record:
            return self._record(task, variant_id, full_prompt, path)
        return self._replay(task_id, variant_id, path)

    def _run_api(self, task_id: str, variant_id: str, prompt: str) -> RunResult:
        error = None
        answer = ""
        rep_in = 0
        rep_out = 0
        elapsed_ms = 0

        try:
            answer, rep_in, rep_out, elapsed_ms = _call_api(prompt)
        except requests.HTTPError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:400]}"
        except Exception as e:
            error = str(e)

        artifact = _strip_sources(answer)
        # Uniform accounting. accGPT always reports usage=0, so input is estimated
        # from the prompt we send (its server-side RAG retrieval context stays
        # hidden — documented in cost/params.yaml via ctx_tokens_rag).
        acct = tokens.resolve(
            reported_input=rep_in or None,
            reported_output=rep_out or None,
            prompt=prompt,
            answer=artifact,
            estimate_input_from_prompt=True,
        )

        return RunResult(
            task_id=task_id,
            system="accgpt",
            variant_id=variant_id,
            answer_artifact=artifact,
            retrieval_tokens=None,
            latency_ms=elapsed_ms,
            raw_response=answer,
            error=error,
            **acct,
        )

    def _record(
        self, task: dict[str, Any], variant_id: str, prompt: str, path: Path
    ) -> RunResult:
        task_id = task["id"]
        print("\n" + "=" * 70)
        print(f"TASK: {task_id}  VARIANT: {variant_id}")
        print("=" * 70)
        print("Copy this prompt into the accGPT web UI:")
        print("-" * 70)
        print(prompt)
        print("-" * 70)
        print("Paste accGPT's answer below, then press Ctrl-D (or type END on its own line):")

        lines = []
        try:
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
        except EOFError:
            pass

        answer = "\n".join(lines).strip()
        record = {
            "task_id": task_id,
            "variant_id": variant_id,
            "answer_artifact": answer,
            "input_tokens": None,
            "output_tokens": None,
            "retrieval_tokens": None,
            "latency_ms": 0,
            "raw_response": answer,
            "error": None,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        MANUAL_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(f"Saved → {path}")

        return RunResult(
            task_id=task_id,
            system="accgpt",
            variant_id=variant_id,
            answer_artifact=answer,
            input_tokens=0,
            output_tokens=0,
            retrieval_tokens=None,
            latency_ms=0,
            raw_response=answer,
            error=None,
        )

    def _replay(self, task_id: str, variant_id: str, path: Path) -> RunResult:
        if not path.exists():
            return RunResult(
                task_id=task_id,
                system="accgpt",
                variant_id=variant_id,
                answer_artifact="",
                input_tokens=0,
                output_tokens=0,
                retrieval_tokens=None,
                latency_ms=0,
                raw_response="",
                error="not recorded: run with --record or check API connectivity",
            )
        data = json.loads(path.read_text())
        return RunResult(
            task_id=data["task_id"],
            system="accgpt",
            variant_id=data["variant_id"],
            answer_artifact=data.get("answer_artifact", ""),
            input_tokens=data.get("input_tokens") or 0,
            output_tokens=data.get("output_tokens") or 0,
            retrieval_tokens=data.get("retrieval_tokens"),
            latency_ms=data.get("latency_ms", 0),
            raw_response=data.get("raw_response", ""),
            error=data.get("error"),
        )


def list_models() -> None:
    models = _list_models()
    if models:
        print("Available models:")
        for m in models:
            print(f"  {m}")
    else:
        print("Could not list models. Check API connectivity.")


if __name__ == "__main__":
    import sys
    if "--list-models" in sys.argv:
        list_models()
    else:
        print("Usage: python adapters/accgpt.py --list-models")
