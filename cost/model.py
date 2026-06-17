"""Cost model — parametrized RAG vs Skills differential cost.

Implements the model from the benchmark plan:

  RAG_diff    = C_vectorstore·T
              + C_crawl·crawls
              + C_index·reindexes           # corpus_tokens × p_embed
              + N·(q_embed_tokens·p_embed + retrieval_compute)
              + [externality: CDS request volume]

  Skills_diff = C_authoring + C_maint·T
              + N·(ctx_tokens_skills − ctx_tokens_rag)·p_input_token

Break-even surface: solve RAG_diff = Skills_diff over (N, churn) grid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

PARAMS_FILE = Path(__file__).parent / "params.yaml"


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def rag_diff(p: dict[str, Any], N: float, churn: float | None = None) -> float:
    """Annual differential cost for RAG system (USD).

    Args:
        p: params dict
        N: queries per year
        churn: reindexes per year (overrides p if provided)
    """
    reindexes = churn if churn is not None else p["reindexes_per_year"]

    C_vectorstore = p["vectorstore_monthly_usd"] * 12
    C_crawl = p["crawl_cost_per_run_usd"] * p["crawls_per_year"]
    C_index = p["corpus_tokens"] * p["embed_price_per_token"] * reindexes
    C_per_query = (
        p["q_embed_tokens"] * p["embed_price_per_token"]
        + p["retrieval_compute_usd_per_query"]
    )
    return C_vectorstore + C_crawl + C_index + N * C_per_query


def skills_diff(p: dict[str, Any], N: float) -> float:
    """Annual differential cost for Skills+Docs system (USD).

    The recurring per-query penalty is the extra context tokens loaded
    by Lumi's skills vs RAG's retrieved chunks.
    """
    C_authoring = p["authoring_hours"] * p["hourly_rate_usd"]
    C_maint = p["maint_hours_per_year"] * p["hourly_rate_usd"]
    delta_ctx = p["ctx_tokens_skills"] - p["ctx_tokens_rag"]
    C_per_query = delta_ctx * p["input_token_price_usd"]
    return C_authoring + C_maint + N * C_per_query


def breakeven_N(p: dict[str, Any], churn: float) -> float:
    """Solve RAG_diff(N, churn) = skills_diff(N) for N.

    Returns float('inf') if skills is always cheaper or no crossing exists.
    """
    # skills_diff = authoring + maint + N·(delta_ctx·p_tok)
    # rag_diff    = vectorstore + crawl + churn·index + N·(q_embed + retrieval)
    # Solve: rag_diff = skills_diff for N
    # (rag_per_q - skills_per_q)·N = (skills_fixed - rag_fixed)
    skills_fixed = (
        p["authoring_hours"] * p["hourly_rate_usd"]
        + p["maint_hours_per_year"] * p["hourly_rate_usd"]
    )
    rag_fixed = (
        p["vectorstore_monthly_usd"] * 12
        + p["crawl_cost_per_run_usd"] * p["crawls_per_year"]
        + p["corpus_tokens"] * p["embed_price_per_token"] * churn
    )
    skills_per_q = (p["ctx_tokens_skills"] - p["ctx_tokens_rag"]) * p["input_token_price_usd"]
    rag_per_q = (
        p["q_embed_tokens"] * p["embed_price_per_token"]
        + p["retrieval_compute_usd_per_query"]
    )
    denominator = rag_per_q - skills_per_q
    if abs(denominator) < 1e-12:
        return float("inf")
    N_cross = (skills_fixed - rag_fixed) / denominator
    return N_cross if N_cross > 0 else float("inf")


def breakeven_surface(
    p: dict[str, Any],
    N_range: tuple[float, float] = (100, 500_000),
    churn_range: tuple[float, float] = (1, 52),
    grid_size: int = 200,
) -> dict[str, Any]:
    """Compute 2D break-even surface.

    Returns dict with keys: N_grid, churn_grid, rag_cheaper (bool mask),
    breakeven_N_per_churn.
    """
    N_vals = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), grid_size)
    churn_vals = np.linspace(churn_range[0], churn_range[1], grid_size)
    N_grid, C_grid = np.meshgrid(N_vals, churn_vals)

    rag = np.vectorize(lambda N, c: rag_diff(p, N, c))(N_grid, C_grid)
    sk = np.vectorize(lambda N: skills_diff(p, N))(N_grid)
    rag_cheaper = rag < sk

    be_Ns = [breakeven_N(p, c) for c in churn_vals]

    return {
        "N_grid": N_grid,
        "churn_grid": C_grid,
        "rag_cheaper": rag_cheaper,
        "breakeven_N_per_churn": np.array(be_Ns),
        "churn_vals": churn_vals,
        "N_vals": N_vals,
    }


def cds_externality_ratio(p: dict[str, Any]) -> float:
    """RAG CDS requests / Lumi live requests per year.

    RAG: cds_requests_per_reindex × reindexes_per_year
    Lumi: queries_per_year × cds_fraction
    """
    rag_requests = p["cds_requests_per_reindex"] * p["reindexes_per_year"]
    lumi_requests = p["queries_per_year"] * p["cds_fraction"]
    if lumi_requests == 0:
        return float("inf")
    return rag_requests / lumi_requests


def sensitivity_table(p: dict[str, Any], N: float, churn: float) -> list[dict]:
    """Sweep input_token_price over the configured range."""
    sweep = p.get("input_token_price_sweep", {})
    rows = []
    for label, price in sweep.items():
        pc = dict(p, input_token_price_usd=price)
        rows.append({
            "label": label,
            "price_per_Mtok": price * 1e6,
            "rag_diff_usd": rag_diff(pc, N, churn),
            "skills_diff_usd": skills_diff(pc, N),
            "skills_cheaper": skills_diff(pc, N) < rag_diff(pc, N, churn),
        })
    return rows


if __name__ == "__main__":
    p = load_params()
    N = p["queries_per_year"]
    churn = p["reindexes_per_year"]
    print(f"RAG diff (N={N}, churn={churn}):    ${rag_diff(p, N, churn):.2f}")
    print(f"Skills diff (N={N}):                ${skills_diff(p, N):.2f}")
    print(f"Skills cheaper: {skills_diff(p, N) < rag_diff(p, N, churn)}")
    print(f"CDS externality ratio: {cds_externality_ratio(p):.1f}×")
    print(f"Break-even N (churn={churn}): {breakeven_N(p, churn):.0f} queries/yr")
    print("\nSensitivity table:")
    for row in sensitivity_table(p, N, churn):
        print(f"  {row['label']:6s}  ${row['price_per_Mtok']:.2f}/Mtok  "
              f"RAG=${row['rag_diff_usd']:.0f}  Skills=${row['skills_diff_usd']:.0f}  "
              f"skills_cheaper={row['skills_cheaper']}")
