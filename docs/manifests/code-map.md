# Code Map

## Analysis Scripts → Paper Exhibits

| Script | Paper Section | Exhibits Produced |
|--------|--------------|-------------------|
| `option_strats_uncond_analysis.py` | §3 Unconditional Results | Tables 1a–1e (stratret by subperiod), moment-explanation regressions, all bar-chart and time-series figures |
| `compute_implementable_pnl.py` | §3.4 Implementability | Table 2: execution cost layers, turnover proxy, ES₁% |
| `compute_tail_risk_diagnostics.py` | §3.5 Tail Risk | Table 3: tail risk diagnostics (ES, max DD, worst day/5-day) |
| `compute_clustered_inference_mht.py` | §4.1 Inference Robustness | Table 4: clustered-SE regressions + Benjamini-Hochberg q-values |
| `compute_structural_break_2022.py` | §3.4 Regime Stability | Table 5: pre/post-2022 PNL shift test |
| `compute_vix_regime_conditioning.py` | §5.1 VIX Regime Filter | Table 6: strategy PNL by VIX tercile |
| `compute_conditional_oos_protocol.py` | §5.2 Strict OOS Protocol | Table 7: strategy-level OOS hit rate, Brier, SR |
| `compute_conditional_model_zoo.py` | §5.2 Model Comparison | Tables 8–9: model family zoo (tree, linear, NN families) |
| `compute_conditional_oos_investment_ts.py` | §5.3 OOS Portfolio | Table 9: portfolio implementation; cumulative PNL figure |
| `build_conditional_target_choice_table.py` | §5.2 Target Choice | Table 8: return vs. binary target comparison |
| `derive_binary_decision_summary.py` | §5.2 Binary Decision | Binary decision detail tables |
| `moneyness_selection.py` | §5.2 Representative Moneyness | Table: moneyness config selection |
| `plot_conditional_topk_basket_legs.py` | §5.3 Basket Composition | Figure: top-K basket leg decomposition |
| `figs_strats.py` | §3 | All strategy bar charts, payoff diagrams, time series |

## Data Flow

```
data/data_opt.parquet ─────────────┬─→ option_strats_uncond_analysis.py
                                   ├─→ compute_implementable_pnl.py
                                   ├─→ compute_tail_risk_diagnostics.py
                                   └─→ figs_strats.py

data/data_structures.parquet ──────┬─→ option_strats_uncond_analysis.py
                                   ├─→ compute_clustered_inference_mht.py
                                   ├─→ compute_structural_break_2022.py
                                   ├─→ compute_vix_regime_conditioning.py
                                   ├─→ compute_conditional_oos_protocol.py
                                   └─→ compute_conditional_model_zoo.py

data/vix.parquet ──────────────────┬─→ option_strats_uncond_analysis.py
                                   ├─→ compute_clustered_inference_mht.py
                                   ├─→ compute_vix_regime_conditioning.py
                                   ├─→ compute_conditional_oos_protocol.py
                                   └─→ compute_conditional_model_zoo.py

data/future_moments_SPX.parquet ───┬─→ option_strats_uncond_analysis.py
                                   ├─→ compute_clustered_inference_mht.py
                                   ├─→ compute_conditional_oos_protocol.py
                                   └─→ compute_conditional_model_zoo.py

data/slopes.parquet ───────────────┬─→ compute_conditional_oos_protocol.py
                                   └─→ compute_conditional_model_zoo.py

data/ALL_eod.csv ──────────────────→ option_strats_uncond_analysis.py
```
