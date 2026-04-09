# CLAUDE.md — Claude Project Context

## What Is This Repository?

Public replication package for *"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"* (Vilkov, 2026). Contains analysis code, derived data panels, and AI-optimized documentation for reproducing every exhibit in the paper.

## Quick Orientation

- **Paper topic**: Realized payoffs of S&P 500 zero-days-to-expiration (0DTE) options and multi-leg strategies. Tests whether 0DTE offers repeatable carry or is better used as a tactical overlay.
- **Sample**: SPXW options, Sep 2016 – Jan 2026, focusing on 10:00 ET entry held to 16:00 ET close.
- **Main finding**: Unconditional 0DTE is weak after costs and tail risk; conditional OOS rules on selected strategies and baskets achieve SR 1.0–1.3.

## How to Run

```bash
pip install -r requirements.txt
python tools/doctor.py           # verify setup
python code/run_replication.py   # reproduce all tables + figures
python tests/test_replication.py # verify parity with reference
```

## Key Commands

| Task | Command |
|------|---------|
| Reproduce everything | `python code/run_replication.py` |
| Single table family | `python code/analysis/<script>.py` |
| Check environment | `python tools/doctor.py` |
| Run parity tests | `python tests/test_replication.py` |

## Code Structure

All analysis scripts live in `code/analysis/` and follow a uniform pattern:
1. Parse args (optional `--project-root`, `--data-dir`)
2. Load data from `data/` via `_paths.py` helpers
3. Compute statistics
4. Write LaTeX tables to `output/tables/` or PDF figures to `output/figures/`

Central config: `code/config.py` (RepoConfig dataclass with path properties).

## Data Files

All in `data/` directory (Git LFS tracked):
- `data_opt.parquet` — interpolated option panel (mid, spread, Greeks, payoff, PNL)
- `data_structures.parquet` — strategy-level panel (7 types, flow/liquidity features)
- `vix.parquet` — VIX and intraday implied/realized moments
- `slopes.parquet` — Volatility surface slopes (point-in-time)
- `future_moments_SPX.parquet` — forward realized moments for SPX
- `future_moments_VIX.parquet` — forward realized moments for VIX
- `ALL_eod.csv` — end-of-day reference prices

## Seven Strategy Types

1. Strangle/Straddle
2. Iron Butterfly/Condor
3. Risk Reversal
4. Bull Call Spread
5. Bear Put Spread
6. Call Ratio Spread
7. Put Ratio Spread

## Important Context for Edits

- Scripts use relative imports via `_paths.py` — run from `code/analysis/` or set `ODTE_REPO_ROOT`
- Data is `.parquet` format (not HDF5) — moments were converted from `.h5` to two parquet files
- LaTeX tables use `booktabs` style (`\toprule`, `\midrule`, `\bottomrule`)
- Figures use `matplotlib` with PDF backend
- After any code change, run `python tests/test_replication.py` to verify

## Style Conventions

- Python 3.10+, type hints encouraged
- `from __future__ import annotations` at top of all modules
- `pandas` for data manipulation, `statsmodels` for regression/inference
- `argparse` for CLI in each script
- No notebook dependencies — all scripts are standalone `.py` files
