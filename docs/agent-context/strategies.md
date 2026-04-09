# Strategy Templates

## Seven Standard Multi-Leg Structures

All strategies are constructed from interpolated 0DTE SPXW options at a fixed intraday entry time (10:00 ET in main text) and held to 16:00 ET settlement.

### 1. Strangle / Straddle
- **Legs**: Long OTM put + Long OTM call (strangle) or both ATM (straddle)
- **Exposure**: Long volatility, delta-neutral at initiation
- **Payoff profile**: Limited downside (premium paid), unlimited upside on large moves
- **PNL driver**: Realized variance exceeding implied variance

### 2. Iron Butterfly / Iron Condor
- **Legs**: Short ATM strangle + Long OTM wings (butterfly); wider short strikes = condor
- **Exposure**: Short volatility with bounded risk
- **Payoff profile**: Bounded upside and bounded downside
- **PNL driver**: Realized variance staying below implied; time-value capture

### 3. Risk Reversal
- **Legs**: Long OTM call + Short OTM put
- **Exposure**: Directional bullish (long delta), short skewness premium
- **Payoff profile**: Open-tailed on both sides
- **PNL driver**: Directional index move and skewness premium

### 4. Bull Call Spread
- **Legs**: Long lower-strike call + Short higher-strike call
- **Exposure**: Directional bullish, bounded
- **Payoff profile**: Bounded profit and loss
- **PNL driver**: Upward index move

### 5. Bear Put Spread
- **Legs**: Long higher-strike put + Short lower-strike put
- **Exposure**: Directional bearish, bounded
- **Payoff profile**: Bounded profit and loss
- **PNL driver**: Downward index move

### 6. Call Ratio Spread (1×2)
- **Legs**: Long 1 ATM call + Short 2 OTM calls
- **Exposure**: Moderate upside bet with asymmetric tail
- **Payoff profile**: Capped upside, open downside on large up-moves
- **PNL driver**: Moderate up move; hurt by large up moves

### 7. Put Ratio Spread (1×2)
- **Legs**: Long 1 ATM put + Short 2 OTM puts
- **Exposure**: Moderate downside bet with asymmetric tail
- **Payoff profile**: Capped upside, open downside on large down-moves
- **PNL driver**: Moderate down move; hurt by large down moves

## Moneyness Configuration

Strategies are parameterized by moneyness of each leg. For example, a strangle at (0.99, 1.01) uses a put at M=0.99 and a call at M=1.01.

For the conditional analysis, one **representative** near-ATM configuration is selected per strategy (all legs within ±1% of ATM, chosen by maximum day coverage).

## PNL Definition

```
Strategy PNL = Σ (q_l × PNL_l)   over legs l
PNL_l = Payoff_l - Mid_l
Net PNL = Strategy PNL - half_spread_cost - 0.5bp_slippage
```

All quantities are in **percentage of underlying** (spot-relative × 100).
