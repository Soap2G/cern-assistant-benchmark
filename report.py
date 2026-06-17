#!/usr/bin/env python3
"""Generate benchmark report: accuracy-vs-tokens Pareto + break-even surface.

Usage:
  python report.py results/run_<id>.jsonl
  python report.py results/run_<id>.jsonl --cost-params cost/params.yaml
  python report.py results/run_<id>.jsonl --output-dir report_out/

Outputs (saved to --output-dir or results/):
  pareto_ops.png
  pareto_cds.png
  breakeven_surface.png
  summary_table.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from telemetry.schema import RunResult, Score
from telemetry.stats import summarize
from cost.model import load_params, rag_diff, skills_diff, breakeven_surface, cds_externality_ratio, sensitivity_table


def _status(s: dict) -> str:
    """Score status, tolerating older score rows that predate the status field."""
    st = s.get("status")
    if st:
        return st
    return "pass" if s.get("passed") else "fail"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def pair_results_scores(
    results: list[dict], scores: list[dict]
) -> list[tuple[dict, dict]]:
    score_index = {
        (s["task_id"], s["system"], s["variant_id"]): s for s in scores
    }
    pairs = []
    for r in results:
        key = (r["task_id"], r["system"], r["variant_id"])
        s = score_index.get(key)
        if s:
            pairs.append((r, s))
    return pairs


def plot_pareto(
    pairs: list[tuple[dict, dict]],
    family: str,
    output_path: Path,
    cern_N: int | None = None,
) -> None:
    """Accuracy (pass rate) vs median input_tokens — per-system scatter."""
    by_system: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    for r, s in pairs:
        if r["task_id"].startswith(family) and _status(s) != "infra_error":
            by_system[r["system"]].append((r["input_tokens"], _status(s) == "pass"))

    if not any(by_system.values()):
        print(f"  [report] No data for family={family}, skipping Pareto plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"lumi": "#2196F3", "accgpt": "#FF5722"}
    markers = {"lumi": "o", "accgpt": "s"}

    for system, pts in by_system.items():
        if not pts:
            continue
        token_counts = [p[0] for p in pts]
        passed = [p[1] for p in pts]
        # Group by unique token count and compute pass rate
        token_buckets: dict[int, list[bool]] = defaultdict(list)
        for tok, ok in zip(token_counts, passed):
            token_buckets[tok].append(ok)

        xs = sorted(token_buckets.keys())
        ys = [np.mean(token_buckets[x]) for x in xs]
        ax.scatter(
            xs, ys,
            color=colors.get(system, "grey"),
            marker=markers.get(system, "^"),
            s=80, alpha=0.8, label=system, zorder=3,
        )

    ax.set_xlabel("Input tokens (context size)")
    ax.set_ylabel("Pass rate")
    ax.set_title(f"Accuracy vs Context Tokens — {family.upper()} tasks")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_breakeven(
    params: dict,
    output_path: Path,
    cern_N: float | None = None,
    cern_churn: float | None = None,
) -> None:
    """Break-even surface: shade where Skills+Docs is cheaper."""
    surf = breakeven_surface(params, grid_size=300)
    N_grid = surf["N_grid"]
    C_grid = surf["churn_grid"]
    rag_cheaper = surf["rag_cheaper"]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Shade where skills is cheaper (NOT rag_cheaper)
    ax.contourf(
        N_grid, C_grid, (~rag_cheaper).astype(float),
        levels=[0.5, 1.5], colors=["#4CAF50"], alpha=0.25,
    )
    ax.contourf(
        N_grid, C_grid, rag_cheaper.astype(float),
        levels=[0.5, 1.5], colors=["#F44336"], alpha=0.15,
    )
    # Break-even line
    ax.contour(
        N_grid, C_grid, rag_cheaper.astype(float),
        levels=[0.5], colors=["black"], linewidths=2,
    )

    # CERN real point
    if cern_N and cern_churn:
        ax.scatter(
            [cern_N], [cern_churn],
            color="gold", s=200, marker="*", zorder=5,
            label=f"CERN (N={cern_N:,}, churn={cern_churn}/yr)",
        )

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.4, label="Skills+Docs cheaper"),
        Patch(facecolor="#F44336", alpha=0.4, label="RAG cheaper"),
        plt.Line2D([0], [0], color="black", lw=2, label="Break-even line"),
    ]
    if cern_N:
        legend_elements.append(
            plt.Line2D([0], [0], marker="*", color="w",
                       markerfacecolor="gold", markersize=14, label="CERN point")
        )
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_xscale("log")
    ax.set_xlabel("Annual query volume N")
    ax.set_ylabel("Corpus reindexes per year (churn)")
    ax.set_title("Break-even surface: Skills+Docs vs RAG differential cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def print_summary_table(
    pairs: list[tuple[dict, dict]],
    output_path: Path,
    params: dict,
) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("QUALITY SUMMARY  —  pass rate [95% Wilson CI];  infra failures excluded")
    lines.append("=" * 70)

    by_fs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r, s in pairs:
        family = r["task_id"].split("_")[0]
        by_fs[(family, r["system"])].append(_status(s))

    for (family, system), statuses in sorted(by_fs.items()):
        st = summarize(statuses)
        infra = f"  (+{st.n_infra} infra-excluded)" if st.n_infra else ""
        lines.append(f"  {family:4s} / {system:8s}  {st.n_pass}/{st.n_scored}  "
                     f"{st.pct()}{infra}")

    lines.append("")
    lines.append("TOKEN ECONOMY (median input tokens per query; provenance shown)")
    lines.append("-" * 60)
    # Group effective input tokens by system, but only over runs whose input is
    # actually known (source 'reported'/'estimated'); 'unavailable' rows (e.g.
    # Lumi token-fetch misses) are excluded rather than counted as 0.
    by_system: dict[str, list[int]] = defaultdict(list)
    src_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r, _ in pairs:
        src = r.get("tokens_source", "reported")
        src_counts[r["system"]][src] += 1
        if "unavailable" not in src and r.get("input_tokens"):
            by_system[r["system"]].append(r["input_tokens"])
    for system in sorted(src_counts):
        toks = by_system.get(system, [])
        med = int(np.median(toks)) if toks else 0
        srcs = ", ".join(f"{k}:{v}" for k, v in sorted(src_counts[system].items()))
        lines.append(f"  {system:10s}  median {med:,} input tok/query "
                     f"(n={len(toks)}; sources: {srcs})")

    lines.append("")
    lines.append("COST MODEL SNAPSHOT (from params.yaml + measured tokens)")
    lines.append("-" * 50)
    N = params["queries_per_year"]
    churn = params["reindexes_per_year"]
    lines.append(f"  N={N} queries/yr, churn={churn} reindexes/yr")
    lines.append(f"  RAG differential cost:    ${rag_diff(params, N, churn):,.2f}/yr")
    lines.append(f"  Skills differential cost: ${skills_diff(params, N):,.2f}/yr")
    lines.append(f"  Skills cheaper: {skills_diff(params, N) < rag_diff(params, N, churn)}")
    lines.append(f"  CDS externality ratio: {cds_externality_ratio(params):.1f}x")

    lines.append("")
    lines.append("SENSITIVITY (token price sweep)")
    lines.append("-" * 50)
    for row in sensitivity_table(params, N, churn):
        lines.append(
            f"  {row['label']:6s}  ${row['price_per_Mtok']:.2f}/Mtok  "
            f"RAG=${row['rag_diff_usd']:,.0f}  Skills=${row['skills_diff_usd']:,.0f}  "
            f"skills_cheaper={row['skills_cheaper']}"
        )

    lines.append("=" * 70)
    text = "\n".join(lines)
    print(text)
    output_path.write_text(text)
    print(f"  Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("results_jsonl", help="Path to results JSONL file")
    parser.add_argument("--scores-jsonl", default=None,
                        help="Path to scores JSONL (default: auto-detected from results path)")
    parser.add_argument("--cost-params", default=str(ROOT / "cost" / "params.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--cern-n", type=float, default=None,
                        help="CERN real query volume for break-even plot")
    parser.add_argument("--cern-churn", type=float, default=None,
                        help="CERN real reindex cadence for break-even plot")
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    scores_path = Path(
        args.scores_jsonl
        or str(results_path).replace(".jsonl", "_scores.jsonl")
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"Loading results from {results_path}")
    results = load_jsonl(results_path)
    print(f"Loading scores from {scores_path}")
    scores = load_jsonl(scores_path) if scores_path.exists() else []

    pairs = pair_results_scores(results, scores)
    print(f"Paired {len(pairs)} result+score records")

    params = load_params(Path(args.cost_params))

    # Update measured tokens in params if available
    lumi_toks = [r["input_tokens"] for r, _ in pairs if r["system"] == "lumi" and r["input_tokens"]]
    accgpt_toks = [r["input_tokens"] for r, _ in pairs if r["system"] == "accgpt" and r["input_tokens"]]
    if lumi_toks:
        params["ctx_tokens_skills"] = int(np.median(lumi_toks))
        print(f"  Updated ctx_tokens_skills = {params['ctx_tokens_skills']:,} (measured)")
    # NOTE: accGPT's API does not expose the RAG-retrieved context it actually
    # feeds the LLM; r["input_tokens"] for accGPT is only the observable client
    # prompt (~tens of tokens). That is NOT the processed context, so it must not
    # override the ctx_tokens_rag estimate (doing so would invert the cost model).
    # We keep the params.yaml estimate and only adopt a measured median if it is
    # plausibly a real retrieved-context size.
    if accgpt_toks and int(np.median(accgpt_toks)) >= params.get("ctx_tokens_rag", 0):
        params["ctx_tokens_rag"] = int(np.median(accgpt_toks))
        print(f"  Updated ctx_tokens_rag = {params['ctx_tokens_rag']:,} (measured)")
    else:
        print(f"  Kept ctx_tokens_rag = {params['ctx_tokens_rag']:,} (estimate; "
              f"accGPT retrieval context not exposed by API)")

    run_stem = results_path.stem

    print("\nGenerating Pareto plots...")
    plot_pareto(pairs, "ops", out_dir / f"{run_stem}_pareto_ops.png")
    plot_pareto(pairs, "cds", out_dir / f"{run_stem}_pareto_cds.png")

    print("\nGenerating break-even surface...")
    plot_breakeven(
        params,
        out_dir / f"{run_stem}_breakeven.png",
        cern_N=args.cern_n or params.get("queries_per_year"),
        cern_churn=args.cern_churn or params.get("reindexes_per_year"),
    )

    print("\nSummary table:")
    print_summary_table(pairs, out_dir / f"{run_stem}_summary.txt", params)


if __name__ == "__main__":
    main()
