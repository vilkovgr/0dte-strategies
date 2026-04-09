# Paper Reading Guide for AI Agents

## Paper: "0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing"

### One-Sentence Summary
The paper studies whether S&P 500 zero-days-to-expiration (0DTE) options and multi-leg strategies deliver repeatable net performance, finding that unconditional positions are weak after costs and tail risk, but selected strategies under strict out-of-sample conditional rules achieve Sharpe ratios above 1.0.

### Structure

| Section | Title | Key Question |
|---------|-------|-------------|
| §1 | Introduction | What is the practical use case for 0DTE? |
| §2 | Data, Implementation, and Variable Construction | How are options, strategies, and conditioning variables built? |
| §3 | Unconditional 0DTE Results | Do unconditional positions generate net carry? |
| §3.1 | Variance Risk Premium Is Not Enough | Is the 0DTE VRP large enough to trade? |
| §3.2 | Individual Options | How do naked 0DTE calls/puts perform? |
| §3.3 | Option Strategies | How do multi-leg structures compare? |
| §3.4 | Regime Stability / Implementability | Do results shift after 2022? How do costs affect PNL? |
| §3.5 | Tail Risk and Capital at Risk | How severe is the downside? |
| §4 | What Drives Strategy PNL? | Which moments explain cross-day PNL variation? |
| §4.1 | Inference Robustness | Do results survive clustered SE and multiple-testing correction? |
| §5 | Conditional Signals and Portfolio Use | Can real-time signals improve timing? |
| §5.1 | VIX Regime Filter | Does a simple state partition help? |
| §5.2 | Strict OOS Protocol | Do logistic models beat unconditional at 10:00 ET? |
| §5.3 | OOS Portfolio Implementation | How do diversified baskets perform? |
| §6 | Conclusion | 0DTE is a tactical overlay, not standing carry. |

### Three Core Claims

1. **Unconditional 0DTE is weak.** The variance risk premium exists but is economically small (~0.001% of spot at 10:00 ET). After half-spread + 0.5bp costs, most strategies have poor mean-to-tail ratios.

2. **PNL is driven by directional skewness, not variance.** Realized skewness (RS) is a stronger driver than realized variance (RV) for most strategies. This means predicting *direction* matters more than predicting *volatility*.

3. **Conditional timing works for selected strategies.** Under strict out-of-sample logistic classification, put ratio spreads achieve SR ≈ 1.26, iron butterflies ≈ 0.82, and top-3 baskets reach SR 1.01–1.27.

### Key Methodological Choices

- **Moneyness grid**: Strike/Spot from 0.98 to 1.02, step 0.001 (Akima interpolation)
- **Entry time**: 10:00 ET (main text); 13:00, 15:00, 16:00-prev in appendix
- **Settlement**: 16:00 ET same day (European, cash-settled)
- **Execution cost model**: Mid benchmark → + half-spread → + 0.5bp slippage
- **Conditional protocol**: Expanding or rolling (252-day), logistic with L2, OOS from April 2019
- **No data snooping**: Features use only pre-10:00 ET information; lagged RV/RS are shifted by 1 day

### What an Agent Should Know to Discuss Results

1. Strategy PNL is in **percentage of underlying** (spot-relative), not dollar terms
2. Sharpe ratios are **annualized** by √252
3. The conditional models predict **direction** (sign of net PNL), not magnitude
4. "Soft mapping" means confidence-weighted position sizing: w = 2p - 1
5. "Hard mapping" means binary ±1 position: w = sign(p - 0.5)
6. The model zoo compares Ridge, Elastic Net, Random Forest, LightGBM, XGBoost, and neural networks
7. All cross-strategy comparisons use **representative moneyness** (one near-ATM config per strategy)

---

### Notation Glossary

| Symbol | Type | Meaning | Defined |
|--------|------|---------|---------|
| $S_t$ | Scalar | S&P500 index level at time $t$ | §2 |
| $S_T$ | Scalar | Index level at expiration (16:00 ET) | §2 |
| $M$ | Scalar | Moneyness: strike / spot | §2 |
| $IV_t$ | Scalar | Model-free implied variance from $t$ to expiration (VIX methodology on SPXW 0DTE) | Eq. (1) |
| $IV^{up}_t, IV^{dn}_t$ | Scalar | Up/down implied semivariances (OTM calls / OTM puts) | Eq. (1)–(2) |
| $IS_t$ | Scalar | Implied skewness proxy: $IV^{up}_t - IV^{dn}_t$ | Eq. (2) |
| $RV_t$ | Scalar | Realized variance: sum of squared 1-min log returns from $t$ to 16:00 ET | Eq. (3) |
| $RV^{up}_t, RV^{dn}_t$ | Scalar | Realized up/down semivariances | Eq. (3)–(4) |
| $RS_t$ | Scalar | Realized skewness proxy: $RV^{up}_t - RV^{dn}_t$ | Eq. (4) |
| $VRP_t$ | Scalar | Variance risk premium: $IV_t - RV_t$ | Eq. (5) |
| $SRP_t$ | Scalar | Skewness risk premium: $IS_t - RS_t = VRP^{up}_t - VRP^{dn}_t$ | Eq. (6) |
| $Payoff_t(M, Type)$ | Scalar | Terminal payoff of option $(M, Type)$ at expiration | Eq. (7) |
| $PNL_t(M, Type)$ | Scalar | $Payoff_t - Mid_t$, option profit relative to spot | Eq. (8) |
| $Ret_t(M, Type)$ | Scalar | Return: $PNL_t / Mid_t$ | Eq. (9) |
| $h_{i,l}$ | Scalar | Half-spread execution-cost proxy for leg $l$ of strategy $i$ | Eq. (10) |
| $d_{i,l}$ | Scalar | Displayed depth at quoted prices | Eq. (11) |
| $\rho_{i,l}$ | Scalar | Relative spread: $bas_l / \|mid_l\|$ | Eq. (12) |
| $v_{i,l}$ | Scalar | Traded volume at strategy leg | Eq. (13) |
| $f^\Delta_{i,l}$ | Scalar | Delta-weighted flow notional | Eq. (14) |
| $f^\Gamma_{i,l}$ | Scalar | Gamma-weighted flow notional | Eq. (15) |
| $f^\nu_{i,l}$ | Scalar | Vega-weighted flow notional | Eq. (16) |
| $g^{OI,n}_{i,l}$ | Scalar | Signed OI-weighted gamma exposure (GEX proxy) | Eq. (17) |
| $g^{OI,a}_{i,l}$ | Scalar | Absolute OI-weighted gamma exposure | Eq. (18) |
| $g^{\Gamma,n}_{i,l}, g^{\Gamma,a}_{i,l}$ | Scalar | Signed/absolute structural gamma from leg mix | Eqs. (19)–(20) |
| $B^\Gamma_i$ | Scalar | GEX balance: normalized sign of OI gamma | Eq. (21) |
| $R^\Gamma_i$ | Scalar | Flow-pressure: traded gamma scaled by OI gamma base | Eq. (21) |
| $T_i$ | Scalar | Liquidity tightness: cost per depth unit | Eq. (21) |
| $M_{i,t}$ | Scalar | Moneyness center of strategy | Eq. (22) |
| $P^{(1)}, \bar P^{(5)}, \sigma^{(5)}_P$ | Scalar | Lagged strategy PNL, 5-day mean, 5-day std | Eq. (22) |
| $y^{net}_{s,t}$ | Scalar | Net strategy return after costs | §5.2 |
| $c_{s,t}$ | Scalar | Implementation cost: half-spread + 0.5 bp | §5.2 |
| $p_{s,t}$ | Scalar | Predicted probability $\Pr(y^{net}>0 \mid X)$ from logistic model | §5.2 |
| $w_{s,t}$ | Scalar | Trading position: $\mathrm{sign}(\hat p - 0.5)$ (hard) or $2\hat p - 1$ (soft) | §5.2 |
| $q_l$ | Scalar | Signed leg quantity ($>0$ long, $<0$ short) | §2 |
| $\mathcal{L}_{s,m}$ | Set | Legs in strategy $s$ with moneyness configuration $m$ | §2 |

### Equation Quick-Reference

| Eq. | Name | What it defines | Where used empirically |
|-----|------|-----------------|----------------------|
| (1) | Implied variance | VIX-style IV from SPXW 0DTE OTM options | Figure 1, all market-state controls |
| (2) | Implied skewness | $IS_t = IV^{up}_t - IV^{dn}_t$ | Conditional predictor, Table 4 |
| (3) | Realized variance | Sum of squared 1-min log returns to 16:00 ET | Table 4 (PNL drivers) |
| (4) | Realized skewness | $RS_t = RV^{up}_t - RV^{dn}_t$ | Table 4 (dominant driver for directional strategies) |
| (5) | Variance risk premium | $VRP_t = IV_t - RV_t$ | Figure 1, §3.1 |
| (6) | Skewness risk premium | $SRP_t = IS_t - RS_t$ | Table 4 |
| (7) | Option payoff | Terminal payoff for call/put at moneyness $M$ | All option-level analysis |
| (8) | Option PNL | Payoff minus mid price, spot-relative | Tables 1–3, Figures 2–4 |
| (9) | Option return | PNL / mid price | Figure 3 |
| (10)–(12) | Liquidity features | Half-spread, depth, relative spread per leg | Conditional model zoo |
| (13)–(16) | Flow features | Volume, delta/gamma/vega weighted flow per leg | Conditional model zoo |
| (17)–(20) | GEX features | Signed/absolute OI-weighted and structural gamma | Conditional model zoo |
| (21) | Transformed ratios | $B^\Gamma$, $R^\Gamma$, $T$: normalized balance, flow-pressure, tightness | Conditional model zoo |
| (22) | Baseline features | Moneyness center, lagged PNL, cross-sectional scaling | Conditional model zoo, Table 7 |

### Dataset Divergence Log

| Dimension | Cboe 30-min SPXW bars | ThetaData 1-min bars | Cboe Open-Close Volume |
|-----------|----------------------|---------------------|----------------------|
| Coverage | 09/2016 – 01/2026 | 09/2016 – 01/2026 | Jan 2021 – Jun 2023 |
| Frequency | 30-min bars | 1-min OHLC | 10-min cumulative |
| Content | NBBO quotes, sizes, OHLC, volume, Greeks, underlying levels | SPX and VIX prices | Volume by trader type, buy/sell direction |
| Used for | Option cross-section, strategy construction, implied moments (Eq. 1), Greeks, conditional features | Realized variance (Eq. 3), realized skewness (Eq. 4), semivariances | Not used in main paper (used in companion gamma-channel papers) |
| Key limitation | 30-min resolution caps intraday granularity of strategy signals | No option-level data; underlying prices only | Short sample; C1 exchange only |
| Shipped as | `data_opt.parquet`, `data_structures.parquet`, `vix.parquet`, `slopes.parquet` | `future_moments_SPX.parquet`, `future_moments_VIX.parquet` | Not shipped (companion papers only) |

### Suggested Reading Paths

**Path A: "What does this paper find?" (5 minutes)**
1. Abstract (p. 1): three findings + portfolio takeaway
2. Figure 1 (VRP is small) → Table 1 (unconditional PNL is wide)
3. Table 7 (OOS timing: selected strategies work) → Table 9 (baskets: SR 1.01–1.27)
4. §6 Conclusion: tactical overlay, not carry strategy

**Path B: "Is unconditional 0DTE profitable?" (10 minutes)**
1. §3.1: VRP exists but is small
2. §3.2: Individual option PNL distributions → Figure 3
3. §3.3: Strategy PNL → Table 1, Figure 4
4. §3.4: Execution costs → Table 2
5. §3.5: Tail risk → Table 3

**Path C: "What drives strategy PNL?" (10 minutes)**
1. §4: Moment regressions → Table 4 (RV weak, RS dominant)
2. §4.1: MHT-robust inference → Table 5
3. Figure 5 (time variation in strategy PNL with SPX/VIX overlay)

**Path D: "Does conditional timing work?" (15 minutes)**
1. §5.1: VIX regime filter → Table 6
2. §5.2: OOS protocol + target design → Table 8 (binary > return target)
3. Table 7: Strategy-level OOS results (expanding + rolling windows)
4. §5.3: Portfolio implementation → Table 9, Figure 7

**Path E: "What predictors are used?" (5 minutes)**
1. §2 "Construction of Conditional Features" (pp. 5–7): Eqs. (10)–(22)
2. Appendix Table A1: Full feature dictionary
3. §5.2: Only 10:00 ET information, no look-ahead

**Path F: "How robust are the results?" (10 minutes)**
1. §3.4: Structural break test → Table 5
2. §4.1: MHT correction → Table 4
3. Table 7: Expanding vs rolling OOS windows
4. Table 8: Model-zoo comparison (ridge, elastic net, RF, LightGBM, XGBoost)
5. Appendix: Alternative entry times (16:00 prev, 13:00, 15:00), subperiod tables
