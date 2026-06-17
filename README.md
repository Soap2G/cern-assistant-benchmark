# CERN model/framework benchmark

A reduced, **verifiable** capability + cost benchmark for AI assistants on
CERN-specific tasks. It compares systems (currently **Lumi** — skills+docs via
headless `opencode` — and **accGPT** — RAG) on two task families with
ground-truth scoring, and pairs quality with a token/cost model.

The design bet: credibility comes from *verifiable oracles*, not LLM-judges. A
HTCondor submit file either parses or it doesn't; a CDS recid is either in the
gold set or not.

## Layout

```
runner.py            # task × system × variant matrix; streams results + scores
report.py            # Pareto plots, break-even surface, summary (Wilson CIs)
rescore.py           # re-score saved results through current checkers (no model calls)
systems.yaml         # system registry: name -> adapter, repeats, params
systems.py           # loads adapters from the registry (config-driven)
adapters/            # one module per system (lumi, accgpt); return a RunResult
checkers/            # family scorers: ops_condor (parse), cds_recall (hit@k)
                     #   base.py: infra-vs-quality triage shared by checkers
telemetry/
  schema.py          # RunResult / Score dataclasses (Score.status: pass|fail|infra_error)
  tokens.py          # one tokenizer + one token-resolution policy
  stats.py           # Wilson score intervals
cost/                # model.py + params.yaml (RAG vs skills differential cost)
tasks/<family>/*.yaml# task definitions: prompt, paraphrases, gold answers
batch/               # HTCondor submission + post-run enrich/merge/report
```

## Run

```bash
VENV=/eos/user/g/gguerrie/bench-venv/bin/python

# one system, one family
$VENV runner.py --system lumi --family ops --run-id myrun

# all registered systems
$VENV runner.py --system all --run-id myrun

# report (Pareto + break-even + summary with 95% CIs)
$VENV report.py results/myrun.jsonl
```

Unattended batch (4 slices = system × family) is in `batch/` — see
`batch/submit.sh` and `batch/merge_report.sh`.

## Key concepts

- **Infra vs quality.** A run that produced no artifact *and* errored
  (network/timeout/auth) is an `infra_error`, excluded from the quality
  denominator — a transient outage never counts as the model getting it wrong.
  A non-blocking error (e.g. Lumi token-fetch lag) alongside a real answer is
  scored normally.
- **Token accounting** (`telemetry/tokens.py`) is uniform and labels provenance:
  `reported` (provider usage), `estimated` (tiktoken of observable text), or
  `unavailable`. Output is always estimable; agent input is reported-only (the
  bare prompt is not a valid proxy for injected context); accGPT's RAG context
  is server-side and hidden (the cost model uses an estimate, `ctx_tokens_rag`).
- **Cost model** (`cost/params.yaml`) is parametric. The infra-cost numbers are
  placeholders until filled with real CERN figures — treat the dollar outputs as
  illustrative until then.

## Adding a system

Add an entry to `systems.yaml` and an adapter module whose class exposes
`run(task, variant_id, prompt) -> RunResult` and a constructor accepting
`(record=False, **params)`. No runner changes needed.

## Secrets

No tokens live in the repo. The LiteLLM key and the accGPT bearer token are read
from files outside the tree (`ACCGPT_KEY_FILE`, `LUMI_KEY_FILE` / `LITELLM_API_KEY`).
