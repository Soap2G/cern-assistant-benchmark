"""Uniform token accounting — one tokenizer, one resolution policy.

Every adapter resolves its token counts through `resolve()` so the benchmark
treats "tokens" identically across systems and is explicit about what is
*measured* vs *estimated*:

  * output tokens are always observable (we have the answer text), so a missing
    provider count is filled by tokenizing the answer;
  * input tokens are subtler. For an agent like Lumi the real input is the full
    processed context (system prompt + skills + tool results), which only the
    provider can report — the bare user prompt is NOT a valid proxy, so when the
    provider count is missing we mark it "unavailable" rather than inventing one.
    For a plain chat/RAG endpoint the prompt we send *is* the observable input,
    so it can be estimated by tokenizing that prompt (with the caveat that any
    server-side retrieval context stays hidden).

Tokenizer: tiktoken o200k_base (GPT-4.x family). Import is guarded so the module
degrades gracefully (estimates become 0, sources fall back to reported) on hosts
without tiktoken.
"""
from __future__ import annotations

from typing import Any

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:  # tiktoken absent or encoding download blocked
    _ENC = None

# token-source labels
REPORTED = "reported"
ESTIMATED = "estimated"
MIXED = "mixed"
UNAVAILABLE = "unavailable"


def count_tokens(text: str | None) -> int:
    """tiktoken length of `text` (0 if tokenizer unavailable or text empty)."""
    if not text or _ENC is None:
        return 0
    return len(_ENC.encode(text))


def _pos(n: int | None) -> bool:
    return bool(n and n > 0)


def resolve(
    *,
    reported_input: int | None,
    reported_output: int | None,
    prompt: str,
    answer: str,
    estimate_input_from_prompt: bool,
) -> dict[str, Any]:
    """Resolve effective token counts + provenance for one run.

    Args:
        reported_input/reported_output: provider-reported usage (None/0 if hidden).
        prompt: the exact prompt text sent to the system.
        answer: the system's answer text.
        estimate_input_from_prompt: True when the sent prompt is a valid proxy for
            input (plain chat/RAG); False for agents whose real context is injected
            server-/client-side and only the provider can report it (e.g. Lumi).

    Returns a dict of fields ready to splat onto RunResult:
        input_tokens, output_tokens, input_tokens_reported,
        output_tokens_reported, client_input_tokens, tokens_source.
    """
    obs_in = count_tokens(prompt)
    obs_out = count_tokens(answer)

    # OUTPUT — always estimable from the answer.
    if _pos(reported_output):
        eff_out, src_out = int(reported_output), REPORTED
    else:
        eff_out, src_out = obs_out, ESTIMATED

    # INPUT — provider count preferred; estimate only when the prompt is a valid
    # proxy; otherwise mark unavailable (do NOT substitute the bare prompt).
    if _pos(reported_input):
        eff_in, src_in = int(reported_input), REPORTED
    elif estimate_input_from_prompt:
        eff_in, src_in = obs_in, ESTIMATED
    else:
        eff_in, src_in = 0, UNAVAILABLE

    if src_in == src_out:
        source = src_in
    elif UNAVAILABLE in (src_in, src_out):
        # surface the weakest provenance explicitly
        source = f"{src_in}/{src_out}"
    else:
        source = MIXED

    return {
        "input_tokens": eff_in,
        "output_tokens": eff_out,
        "input_tokens_reported": reported_input,
        "output_tokens_reported": reported_output,
        "client_input_tokens": obs_in,
        "tokens_source": source,
    }
