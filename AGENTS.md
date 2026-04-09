# AGENTS.md — AI Agent Onboarding

> Context file for AI coding agents (GitHub Copilot, OpenAI Codex, Claude, etc.)
> working inside this repository.

## Mission

This is the **public replication package** for the paper *"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"* (Vilkov, 2026). The repo enables three tiers of engagement:

1. **Instant replication** — run analysis scripts against shipped derived data to reproduce every table and figure.
2. **Rebuild from source** — use Massive or ThetaData API adapters to reconstruct raw data from scratch.
3. **Explore and extend** — use AI-optimized context to understand, critique, or extend the analysis.

## Paper Summary (One Paragraph)

The paper studies realized payoffs of S&P 500 zero-days-to-expiration (0DTE) options and standard multi-leg structures from 09/2016 to 01/2026. A positive 0DTE variance risk premium exists but is small after realistic frictions. Strategy PNL distributions are wide, tail-heavy, and regime-dependent — dominated by directional and skewness realizations rather than stable mean carry. Yet disciplined 10:00 ET conditional rules under a strict out-of-sample protocol deliver economically meaningful net performance for selected strategies (put ratio spreads SR ≈ 1.26, iron butterflies SR ≈ 0.82) and diversified baskets (SR 1.01–1.27). The practical implication: 0DTE is better viewed as a tightly risk-budgeted tactical overlay than a standing carry strategy.

## Three-Tier Data Model

```
Tier 1 (shipped):    data/*.parquet, data/*.csv
                     ↓ analysis scripts read these directly
                     output/tables/*.tex, output/figures/*.pdf

Tier 2 (rebuild):    API key → code/ingest/{massive,thetadata}/
                     → raw snapshots → code/build_data.py
                     → data/*.parquet (same schema as Tier 1)

Tier 3 (explore):    docs/agent-context/* + AGENTS.md + CLAUDE.md
                     → LLM understands paper claims, methods, variables
```

## Repository File Map

```
code/
  config.py              Path configuration (replaces internal zEnvmt setup)
  run_replication.py     Single entry point: runs all analysis in order
  analysis/
    _paths.py            Shared path resolution helpers
    option_strats_uncond_analysis.py    → Tables 1a–1e (unconditional returns)
    compute_implementable_pnl.py       → Table 2 (execution costs, turnover)
    compute_tail_risk_diagnostics.py   → Table 3 (tail risk, capital at risk)
    compute_clustered_inference_mht.py → Table 4 (clustered SE, BHY correction)
    compute_structural_break_2022.py   → Table 5 (structural break test)
    compute_vix_regime_conditioning.py → Table 6 (VIX regime splits)
    compute_conditional_oos_protocol.py → Table 7 (OOS conditional results)
    compute_conditional_model_zoo.py    → Tables 8–9 (model comparison zoo)
    compute_conditional_oos_investment_ts.py → OOS investment time series
    build_conditional_target_choice_table.py → Target choice comparison
    derive_binary_decision_summary.py   → Binary decision framework
    moneyness_selection.py              → Representative moneyness selection
    plot_conditional_topk_basket_legs.py → Top-K basket composition
    figs_strats.py                      → All figures (payoffs, bars, time series)
  ingest/
    massive/             Massive API download adapter
    thetadata/           ThetaData API download adapter

data/                    Shipped derived panels (Git LFS)
  data_opt.parquet       Interpolated option panel (moneyness 0.98–1.02)
  data_structures.parquet Strategy panel (7 types × moneyness × dates)
  vix.parquet            VIX + intraday moments
  slopes.parquet         Volatility surface slopes (PIT)
  future_moments_SPX.parquet  Forward realized moments (SPX)
  future_moments_VIX.parquet  Forward realized moments (VIX)
  ALL_eod.csv            End-of-day reference prices

output/
  tables/                Generated LaTeX tables
  figures/               Generated PDF/PNG figures

tests/
  reference/tables/      Ground-truth LaTeX for parity checks
  test_replication.py    Byte-level parity test

tools/
  doctor.py              Environment health checker
```

## Key Variables and Concepts

| Variable | Definition | Source |
|----------|-----------|--------|
| IV | Implied variance (VIX methodology on 0DTE SPXW) | data_opt |
| RV | Realized variance (sum of squared 1-min returns to close) | vix, future_moments |
| VRP | Variance risk premium = IV − RV | Computed |
| IS / RS | Implied / Realized skewness (semivariance difference) | data_opt, vix |
| SRP | Skewness risk premium = IS − RS | Computed |
| Mid | Interpolated mid-quote price / spot | data_opt |
| BAS | Bid-ask spread / spot | data_opt |
| PNL | Payoff − Mid (spot-relative, × 100 for %) | data_opt, data_structures |
| Moneyness | Strike / Spot, grid 0.98–1.02, step 0.001 | data_opt |

### Strategy Types (7 Templates)

1. **Strangle / Straddle** — long OTM put + long OTM call (or both ATM)
2. **Iron Butterfly / Condor** — short ATM strangle + long OTM wings
3. **Risk Reversal** — long OTM call + short OTM put
4. **Bull Call Spread** — long lower-strike call + short higher-strike call
5. **Bear Put Spread** — long higher-strike put + short lower-strike put
6. **Call Ratio Spread** — long 1 ATM call + short 2 OTM calls
7. **Put Ratio Spread** — long 1 ATM put + short 2 OTM puts

## Pipeline Flow

```
[Tier 2 only]
Massive/ThetaData API → raw snapshots → code/build_data.py → data/*.parquet

[Tier 1 — everyone]
data/*.parquet
  → code/analysis/option_strats_uncond_analysis.py  → output/tables/0dte_stratret*.tex
  → code/analysis/compute_implementable_pnl.py      → output/tables/0dte_implementable_pnl.tex
  → code/analysis/compute_tail_risk_diagnostics.py   → output/tables/0dte_tail_risk*.tex
  → code/analysis/compute_clustered_inference_mht.py → output/tables/0dte_inference*.tex
  → code/analysis/compute_structural_break_2022.py   → output/tables/0dte_structbreak*.tex
  → code/analysis/compute_vix_regime_conditioning.py → output/tables/0dte_vix_regime*.tex
  → code/analysis/compute_conditional_oos_protocol.py → output/tables/0dte_conditional_oos*.tex
  → code/analysis/compute_conditional_model_zoo.py    → output/tables/0dte_conditional_model_zoo*.tex
  → code/analysis/figs_strats.py                      → output/figures/*.pdf

[Verify]
python tests/test_replication.py  → compares output/tables/ vs tests/reference/tables/
```

## Conventions for AI Agents

1. **Path resolution**: All scripts use `code/analysis/_paths.py` which resolves the repo root from the file's location or `ODTE_REPO_ROOT` env var. Do not hardcode absolute paths.

2. **Data access pattern**: Scripts load data via `pd.read_parquet(data_dir / "filename.parquet")` or `pd.read_csv(data_dir / "filename.csv")`. The `data_dir` always points to `<repo_root>/data/`.

3. **Output pattern**: Tables are written as `.tex` files to `output/tables/`. Figures are written as `.pdf` files to `output/figures/`.

4. **No proprietary data**: Never attempt to read `.h5` files (HDF5) — the repo uses `.parquet` exclusively. The original HDF5 moments file has been converted to `future_moments_SPX.parquet` and `future_moments_VIX.parquet`.

5. **Testing**: After modifying any analysis script, run `python tests/test_replication.py` to verify output parity.

6. **Dependencies**: All required packages are in `requirements.txt`. Do not add heavy ML frameworks unless the user explicitly requests Tier 2 model training.
