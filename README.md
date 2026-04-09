# 0DTE Trading Rules: Replication Package

**Paper:** *0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing*
**Author:** Grigory Vilkov (Frankfurt School of Finance & Management)
**Sample:** S&P 500 SPXW 0DTE options, September 2016 – January 2026

---

## What This Repository Does

This is the **AI-augmented replication package** for the paper. It ships:

| Tier | What you get | What you need |
|------|-------------|---------------|
| **Tier 1 — Instant replication** | Pre-built derived data panels + analysis scripts → reproduce every table and figure | Python ≥ 3.10, `pip install -r requirements.txt` |
| **Tier 2 — Rebuild from source** | Ingest adapters for [Massive](https://massive.com) or [ThetaData](https://thetadata.net) → rebuild raw data from scratch | Tier 1 + API subscription |
| **Tier 3 — Explore & extend** | AI-optimized context documents + agent skills → ask an LLM to explain, critique, or extend the analysis | Tier 1 + any code-capable LLM |

Raw Cboe bar files are proprietary and not redistributed. The shipped derived panels (`data/`) contain all interpolated option-level, strategy-level, and moment variables needed to run every exhibit.

---

## Quick Start

```bash
# 1. Clone (includes ~400 MB of LFS data)
git clone https://github.com/vilkovgr/0dte-strategies.git
cd 0dte-strategies

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify environment
python tools/doctor.py

# 4. Reproduce all tables and figures
python code/run_replication.py
```

Output lands in `output/tables/` and `output/figures/`. Compare against `tests/reference/` to verify parity.

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions, troubleshooting, and platform-specific notes.

---

## Repository Layout

```
0dte-strategies/
├── code/
│   ├── config.py                  # Central path configuration
│   ├── run_replication.py         # Single-command replication entry point
│   ├── analysis/                  # One script per table/figure family
│   │   ├── option_strats_uncond_analysis.py
│   │   ├── compute_implementable_pnl.py
│   │   ├── compute_clustered_inference_mht.py
│   │   ├── compute_structural_break_2022.py
│   │   ├── compute_tail_risk_diagnostics.py
│   │   ├── compute_vix_regime_conditioning.py
│   │   ├── compute_conditional_oos_protocol.py
│   │   ├── compute_conditional_model_zoo.py
│   │   ├── compute_conditional_oos_investment_ts.py
│   │   ├── build_conditional_target_choice_table.py
│   │   ├── derive_binary_decision_summary.py
│   │   ├── moneyness_selection.py
│   │   ├── plot_conditional_topk_basket_legs.py
│   │   └── figs_strats.py
│   └── ingest/                    # Raw data rebuilders (Tier 2)
│       ├── massive/
│       └── thetadata/
├── data/                          # Derived data panels (Git LFS)
│   ├── data_opt.parquet           # Interpolated option-level panel
│   ├── data_structures.parquet    # Strategy-level panel
│   ├── vix.parquet                # VIX and intraday moment series
│   ├── slopes.parquet             # Volatility surface slopes (PIT)
│   ├── future_moments_SPX.parquet # Forward-looking realized moments (SPX)
│   ├── future_moments_VIX.parquet # Forward-looking realized moments (VIX)
│   └── ALL_eod.csv                # End-of-day reference prices
├── output/
│   ├── tables/                    # Generated LaTeX tables
│   └── figures/                   # Generated PDF/PNG figures
├── tests/
│   ├── reference/tables/          # Ground-truth tables for parity checks
│   └── test_replication.py        # Automated parity test
├── tools/
│   └── doctor.py                  # Environment and data health check
├── docs/                          # AI context and paper documentation
│   ├── agent-context/             # Structured context for LLM agents
│   └── manifests/                 # Exhibit-to-code mappings
├── AGENTS.md                      # AI agent onboarding (GitHub Copilot / Codex)
├── CLAUDE.md                      # Claude-specific project context
├── QUICKSTART.md                  # Detailed setup guide
├── requirements.txt               # Python dependencies
├── .env.example                   # Template for API keys
└── LICENSE                        # MIT
```

---

## Data Description

### Shipped Panels (Tier 1)

| File | Rows | Description |
|------|------|-------------|
| `data_opt.parquet` | ~3.5M | Interpolated 0DTE option observations (moneyness 0.98–1.02, step 0.001) with mid, spread, Greeks, payoff, PNL |
| `data_structures.parquet` | ~700K | Strategy-level panel: 7 structure types × moneyness configs × dates, with PNL, flow, and liquidity features |
| `vix.parquet` | ~30K | Intraday VIX levels and implied/realized moment time series |
| `slopes.parquet` | ~2.4K | Volatility surface slopes (point-in-time, intraday) |
| `future_moments_{SPX,VIX}.parquet` | ~2.4K each | Forward-looking realized variance, semivariance, skewness (daily) |
| `ALL_eod.csv` | ~2.4K | End-of-day SPX, VIX, and related reference prices |

### Rebuilding from Source (Tier 2)

If you hold a Massive or ThetaData subscription, you can rebuild the raw data:

```bash
# Using Massive
python code/ingest/massive/download_spxw.py --start 2016-09-01 --end 2026-02-01

# Using ThetaData
python code/ingest/thetadata/download_spxw.py --start 2016-09-01 --end 2026-02-01

# Then build derived panels
python code/build_data.py --source massive  # or --source thetadata
```

---

## Reproducing Specific Exhibits

Each script maps to a family of paper exhibits. Run any individually:

```bash
cd code/analysis

# Table 1: Unconditional strategy returns (full sample + sub-periods)
python option_strats_uncond_analysis.py

# Table 2: Implementable PNL diagnostics
python compute_implementable_pnl.py

# Table 3: Tail risk diagnostics
python compute_tail_risk_diagnostics.py

# Table 4: Clustered inference with multiple-hypothesis testing
python compute_clustered_inference_mht.py

# Table 5: Structural break around 2022 daily-expiry expansion
python compute_structural_break_2022.py

# Table 6: VIX regime conditioning
python compute_vix_regime_conditioning.py

# Tables 7–9: Conditional OOS protocol and model zoo
python compute_conditional_oos_protocol.py
python compute_conditional_model_zoo.py

# Figures: Strategy payoff diagrams, bar charts, time series
python figs_strats.py
```

---

## Testing

Verify that your output matches the paper's published tables:

```bash
python tests/test_replication.py
```

The harness compares generated LaTeX tables byte-for-byte against `tests/reference/tables/`.

---

## AI Agent Integration (Tier 3)

This repo is designed to work with AI coding assistants. See:

- **[AGENTS.md](AGENTS.md)** — Mission, data model, and file map for GitHub Copilot / Codex agents
- **[CLAUDE.md](CLAUDE.md)** — Claude-specific onboarding and skills

Example prompts for an AI assistant:

> "Reproduce Table 4 and explain what the multiple-testing adjustment does."
>
> "Which strategies have positive Sharpe ratios after transaction costs?"
>
> "Run the conditional OOS protocol and show me the top-5 strategy basket."
>
> "What would change if I moved the entry time from 10:00 to 13:00 ET?"

---

## Citation

```bibtex
@article{Vilkov2026_0DTE,
  author  = {Vilkov, Grigory},
  title   = {0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing},
  year    = {2026},
  note    = {Working paper, Frankfurt School of Finance \& Management}
}
```

---

## License

Code: [MIT](LICENSE). Data panels are derived from proprietary sources and provided for academic replication only; redistribution of raw exchange data is not permitted.
