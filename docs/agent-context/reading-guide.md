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
