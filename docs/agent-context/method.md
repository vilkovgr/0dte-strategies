# Methodology Summary

## Data Construction

### Source Data
- **Cboe 30-minute option bars** with NBBO quotes, sizes, OHLC, volume, underlying levels
- **ThetaData 1-minute SPX/VIX bars** for realized moment construction
- Root: SPXW (European, cash-settled at 16:00 ET)
- Sample: September 2016 – January 2026

### Interpolation Pipeline
1. Filter same-day-expiry SPXW options at each 30-min bar
2. Compute moneyness = Strike / Spot
3. Retain options with moneyness in [0.98, 1.02]
4. Scale mid prices and spreads by spot
5. Interpolate over moneyness grid (step 0.001) using Akima 1D interpolation
6. Separate calls and puts; compute payoffs, intrinsic value, time value, PNL

### Strategy Construction
- Match legs by (datetime, option_type, moneyness)
- Strategy PNL = signed sum of leg-level PNL weighted by leg quantities
- All quantities in spot-relative units (× 100 for percentage)

## Unconditional Analysis (§3)

- Report full-sample and subperiod statistics for all strategy types
- Three execution tiers: mid, mid + half-spread, mid + half-spread + 0.5bp
- Tail risk metrics: ES₁%, max drawdown, worst day, worst 5-day
- Moment regressions with combo fixed effects and date-clustered SE

## Conditional Analysis (§5)

### Protocol Design
- **Features**: 10:00 ET implied state (IV, IS, slope_up, slope_dn) + lagged realized (RV, RS, return) + lagged strategy PNL
- **Target**: Binary — 1 if net PNL > 0, else 0
- **Model**: Logistic regression with L2 regularization (benchmark)
- **Windows**: Expanding (from start) or rolling (252 trading days)
- **OOS start**: April 2019 (after 252-day burn-in)
- **Standardization**: Features standardized within training window only

### Signal → Position Mapping
- **Hard**: w = sign(p̂ - 0.5) ∈ {-1, +1}
- **Soft**: w = 2p̂ - 1 ∈ [-1, +1]
- Realized trading return: r = w × net_PNL

### Model Zoo (Extended)
- Ridge, Elastic Net (linear)
- Random Forest, LightGBM, XGBoost, CatBoost (tree-based)
- Neural networks with linear + ridge output layers
- Each model tested under return-target and binary-target designs

### Portfolio Construction
- Strategy-specific signals combined via equal-weight baskets
- Top-3 by mean net PNL, top-3 by SR, and all-strategies
- No cross-strategy pooling in model estimation
