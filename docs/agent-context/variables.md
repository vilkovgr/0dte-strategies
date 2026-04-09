# Variable Definitions

## Implied Measures (from 0DTE SPXW options at bar time t)

| Variable | Formula | Description |
|----------|---------|-------------|
| IV_t | VIX methodology on 0DTE SPXW | Integrated implied variance from t to 16:00 ET |
| IV^up_t | OTM calls only | Upside implied semivariance |
| IV^dn_t | OTM puts only | Downside implied semivariance |
| IS_t | IV^up_t - IV^dn_t | Implied skewness proxy |

## Realized Measures (from SPX 1-minute bars)

| Variable | Formula | Description |
|----------|---------|-------------|
| RV_t | Σ r²_{t,t+1} | Realized variance from t to 16:00 ET |
| RV^up_t | Σ r²_{t,t+1} × 1(r>0) | Realized upside semivariance |
| RV^dn_t | Σ r²_{t,t+1} × 1(r<0) | Realized downside semivariance |
| RS_t | RV^up_t - RV^dn_t | Realized skewness proxy |

## Premium Measures

| Variable | Formula | Description |
|----------|---------|-------------|
| VRP_t | IV_t - RV_t | Variance risk premium (ex post) |
| VRP^up_t | IV^up_t - RV^up_t | Upside semivariance premium |
| VRP^dn_t | IV^dn_t - RV^dn_t | Downside semivariance premium |
| SRP_t | IS_t - RS_t = VRP^up_t - VRP^dn_t | Skewness risk premium |

## Option-Level Variables

| Variable | Description | Source |
|----------|-------------|--------|
| Mid | Mid-quote price / spot | data_opt |
| BAS | Bid-ask spread / spot | data_opt |
| Delta (Δ) | Option delta | data_opt |
| Gamma (Γ) | Option gamma | data_opt |
| Vega (ν) | Option vega | data_opt |
| Moneyness | Strike / Spot (range 0.98–1.02) | data_opt |
| Payoff | max(S_T/S_t - M, 0) for calls; max(M - S_T/S_t, 0) for puts | Computed |
| PNL | Payoff - Mid | Computed |
| Return | PNL / Mid | Computed |

## Strategy-Level Conditional Features (at 10:00 ET)

### Liquidity features (per leg l)
- **h** (half-spread): |q_l| × BAS_l / 2
- **d** (displayed depth): |q_l| × (BidSz_l + AskSz_l)
- **ρ** (relative spread): |q_l| × BAS_l / |Mid_l|

### Flow features (per leg l)
- **v** (volume): |q_l| × Vol_l
- **f^Δ** (delta notional): |q_l| × Vol_l × |Δ_l| × S_l
- **f^Γ** (gamma notional): |q_l| × Vol_l × Γ_l × S_l²
- **f^ν** (vega notional): |q_l| × Vol_l × ν_l × S_l

### Lagged features
- **PNL_l1**: Yesterday's strategy net PNL
- **PNL_mean5_l1**: 5-day rolling mean of PNL
- **PNL_std5_l1**: 5-day rolling std of PNL
- **SPX_lrv (lagged)**: Yesterday's realized variance
- **SPX_lrv_skew (lagged)**: Yesterday's realized skewness
- **SPX_lret (lagged)**: Yesterday's log return

## Volatility Surface Slopes (PIT)
- **slope_up**: Upside IV slope (OTM calls vs ATM), point-in-time
- **slope_dn**: Downside IV slope (OTM puts vs ATM), point-in-time
