# Paper Digest: 0DTE Trading Rules

> AI-optimized summary of *"0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"* (Vilkov, 2026).
> This document is designed for LLM agents to quickly understand the paper's claims, methods, and evidence.

---

## Metadata

- **Title:** 0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing
- **Author:** Grigory Vilkov (Frankfurt School of Finance & Management)
- **Date:** March 2026
- **Sample:** SPXW 0DTE options, September 2016 – January 2026
- **Keywords:** 0DTE, tactical overlay, tail risk, out-of-sample timing, option implementation, volatility trading

## Abstract

We study realized payoffs of S&P 500 zero-days-to-expiration (0DTE) options and standard multi-leg structures from 09/2016 to 01/2026 to ask whether 0DTE is a repeatable source of net premium or mainly a tactical instrument. A positive 0DTE variance risk premium exists, but at same-day horizons it is small after realistic frictions. Across individual options and strategy templates, net PNL distributions are wide, state-dependent, and dominated by tail risk rather than stable mean carry. Strategy PNL loads more on directional and skewness realizations than on realized variance alone. Yet disciplined 10:00 ET rules estimated under a strict out-of-sample protocol deliver economically meaningful net performance for selected strategies and diversified baskets. Practically, 0DTE is better viewed as a tightly risk-budgeted tactical overlay than a standing carry strategy.

---

## Section-by-Section Summary

### §1 Introduction

**Central question:** Do same-day SPX option positions deliver a repeatable net edge after realistic execution costs, or do they mainly load portfolios with concentrated intraday tail risk?

**Three key findings:**
1. A positive 0DTE variance risk premium exists, but it is economically small at same-day horizons.
2. Strategy-level payoff distributions are wide, tail-heavy, and unstable across regimes.
3. Conditional timing works better as directional classification than return prediction. Selected strategies achieve SR 1.0–1.3 after costs.

**Portfolio takeaway:** Unconditional 0DTE exposure is weak after costs and tail risk; the credible use is a small, governed tactical sleeve.

### §2 Data, Implementation, and Variable Construction

**Data sources:**
- Cboe 30-minute option bars (SPXW, European, cash-settled at 16:00 ET)
- ThetaData 1-minute SPX/VIX bars for realized moments
- Sample: Sep 2016 – Jan 2026

**Construction choices:**
- Moneyness = Strike / Spot, range [0.98, 1.02], grid step 0.001
- Akima interpolation over moneyness (cross-sectional, not across time)
- Entry time: 10:00 ET (main text); appendix covers 13:00, 15:00, 16:00-prev
- PNL = (Payoff − Mid) / Spot × 100 (percentage of underlying)

**Key variables:**
- IV: Implied variance from 0DTE SPXW (VIX methodology)
- RV: Sum of squared 1-minute log returns to close
- VRP = IV − RV (variance risk premium)
- IS, RS: Implied/realized skewness (semivariance difference)
- SRP = IS − RS (skewness risk premium)

### §3 Unconditional 0DTE Results

**§3.1 Variance Risk Premium:**
- Positive and statistically significant, but median VRP ≈ 0.0011% of spot at 10:00 ET
- Too small to monetize after realistic trading frictions

**§3.2 Individual Options:**
- Returns highly volatile, especially OTM
- Selling slightly OTM calls/puts is positive in up to 75% of observations
- Mean returns often not distinguishable from zero

**§3.3 Strategy Performance:**
- 7 strategy templates examined across moneyness configurations
- Distributions are wide; no strategy is risk-free
- Risk reversals are the only structure with consistently positive mean/median PNL
- Iron butterflies/condors show short-vol benefits
- Spreads and ratio spreads highly variable

**§3.4 Regime Stability & Implementability:**
- Pre/post 2022 PNL sign changes exist but are statistically weak
- After execution costs (half-spread + 0.5bp), most strategies have poor mean-to-tail ratios
- Turnover is small but capital usage (ES₁%) is large relative to mean PNL

**§3.5 Tail Risk:**
- ES₁% ranges from 0.58% to 1.58% of underlying
- Worst-day outcomes are severe for all strategies
- Mean PNL alone is an insufficient summary

### §4 What Drives Strategy PNL?

- Realized skewness (RS) is a much stronger driver than realized variance (RV) for asymmetric strategies
- RS explains 20–40% of PNL variation for directional spreads and risk reversals
- Adding IV and IS improves R² by 2–7 percentage points
- All results survive Benjamini-Hochberg multiple-testing correction

### §5 Conditional Signals and Portfolio Use

**§5.1 VIX Regime Filter:**
- Clear regime heterogeneity: downside structures earn more in high-VIX
- Simple but not sufficient for implementation

**§5.2 Strict OOS Protocol:**
- Binary direction prediction >> return magnitude prediction
- Logistic benchmark with L2 regularization
- Features: 10:00 ET implied state + lagged realized + lagged strategy PNL
- Expanding or rolling (252-day) windows; OOS from April 2019
- **Best individual strategies:** Put ratio spread SR ≈ 1.26; Iron butterfly SR ≈ 0.82

**§5.3 Portfolio Implementation:**
- Equal-weight baskets of strategy-specific signals
- **Top-3 by SR: SR ≈ 1.27** | Top-3 by mean: SR ≈ 1.17 | All-strategies: SR ≈ 1.01
- Diversification materially smooths strategy-specific drawdowns

### §6 Conclusion

0DTE is better viewed as a tightly risk-budgeted tactical overlay than a standing carry strategy. Selected strategies under disciplined OOS rules can deliver meaningful performance, but:
- Implementation must account for execution costs and tail risk
- Signals should be transparent and real-time
- Diversification across strategy families is more defensible than reliance on any single structure
- Ongoing OOS monitoring is essential

---

## Exhibit Map

| Exhibit | Type | Description | Replication Script |
|---------|------|-------------|--------------------|
| Figure 1 | VRP bars | Variance risk premiums by intraday bar | `figs_strats.py` |
| Figure 2 | Option prices | 0DTE option prices and time value at 10:00 ET | `figs_strats.py` |
| Figure 3 | Option PNL | Individual option returns and PNL | `figs_strats.py` |
| Figure 4 | Strategy PNL | Unconditional strategy PNL bar charts | `figs_strats.py` |
| Figure 5 | Time series | 63-day moving average PNL with SPX/VIX overlay | `figs_strats.py` |
| Figure 6 | Payoff diagrams | Strategy payoff profiles | `figs_strats.py` |
| Figure 7 | OOS cumulative | Cumulative net PNL of baskets | `compute_conditional_oos_investment_ts.py` |
| Table 1 | Summary stats | Unconditional strategy PNL (full sample + subperiods) | `option_strats_uncond_analysis.py` |
| Table 2 | Implementability | Net PNL, turnover, capital usage by cost layer | `compute_implementable_pnl.py` |
| Table 3 | Tail risk | ES₁%, max drawdown, worst day | `compute_tail_risk_diagnostics.py` |
| Table 4 | Inference | Clustered SE + BHY multiple-testing correction | `compute_clustered_inference_mht.py` |
| Table 5 | Structural break | Pre/post 2022 test | `compute_structural_break_2022.py` |
| Table 6 | VIX regimes | Strategy PNL by implied-variance tercile | `compute_vix_regime_conditioning.py` |
| Table 7 | OOS timing | Strategy-level OOS hit rate, Brier, SR | `compute_conditional_oos_protocol.py` |
| Table 8 | Target choice | Return vs. binary target comparison | `build_conditional_target_choice_table.py` |
| Table 9 | Portfolio | Basket performance with ES, max DD | `compute_conditional_oos_investment_ts.py` |

---

## Key Equations

**Variance Risk Premium:**
\[ VRP_t = IV_t - RV_t \]

**Skewness Risk Premium:**
\[ SRP_t = IS_t - RS_t = VRP^{up}_t - VRP^{dn}_t \]

**Option PNL (spot-relative):**
\[ PNL_t(M, Type) = Payoff_t(M, Type) - Mid_t(M, Type) \]

**Conditional signal (logistic):**
\[ p_{s,t} = \Pr(y^{net}_{s,t} > 0 \mid X_{s,t}) = \Lambda(\alpha_s + \beta_s^\top X_{s,t}) \]

**Hard mapping:** \( w_t = \text{sign}(p_t - 0.5) \in \{-1, +1\} \)

**Soft mapping:** \( w_t = 2p_t - 1 \in [-1, +1] \)

---

## Limitations

1. SPXW only; no multi-asset or cross-exchange analysis
2. Hold-to-close only; no dynamic intraday hedging
3. No causal market-impact design
4. Conditioning restricted to pre-10:00 ET information
5. Strategy construction uses interpolated data; results may differ with actual exchange quotes
