# Known Issues and Corrections

This file records defects found in this replication package, what they affected, and
their current status. Corrections found by external replicators are credited by name.
If you find something, please open an issue.

---

## 2026-08 — Transaction-cost unit-scale error (FIXED in code; paper being revised)

**Status:** code fixed in this commit. Paper revision in progress. Shipped derived
outputs partially refreshed (see *Artifact state* below).

**Reported by:** the team at [gex.live](https://gex.live), who reproduced the shipped
`implementable_pnl` table to four decimal places on all seven structures before
diagnosing the unit inconsistency.

### What was wrong

The panel stores two different scales:

| Variable | Scale | Built as |
|---|---|---|
| `bas`, `mid`, `tv` | **fraction** of spot | `(ask-bid)/active_underlying_price` |
| `reth_und`, `reth` | **percent** of spot | `(payoff-mid)*100` |

Cost code combined them directly, for example:

```python
pnl_mid = reth_und                      # percent of spot
pnl_ba  = pnl_mid - half_spread_cost    # percent MINUS fraction  <-- wrong
pnl_ba_fee05 = pnl_ba - 0.005           # percent (0.5bp)         <-- correct
```

The bid-ask half-spread was therefore charged at **1/100 of its true size** —
about 0.022 bp instead of 2.2 bp — while the fixed 0.5 bp fee was correctly scaled.
The same `tc = half_spread_cost + 0.005` pattern appeared in the conditional protocol,
so the conditional tables inherited it.

Independent confirmations that `bas` is a fraction of spot:

1. `reth_und == 100 * (payoff - mid)` holds exactly (max deviation 0.0).
2. `bas * spot` implies a mean dollar spread of about \$1.33, plausible for SPXW 0DTE.
   If `bas` were already percent, the implied spread would be \$0.013, below the
   minimum tick.
3. `option_strats_uncond_analysis.py` already multiplied `bas`, `mid`, `tv` by 100
   before reporting, so the codebase was internally inconsistent.

### What it affected

- `compute_implementable_pnl.py` → `0dte_implementable_pnl.tex`
- `compute_tail_risk_diagnostics.py` → `0dte_tail_risk_diagnostics.tex`
- `compute_conditional_oos_protocol.py` → `0dte_conditional_oos.tex`,
  `conditional_oos_protocol_summary.csv`, `conditional_oos_protocol_predictions.parquet`
- `compute_conditional_oos_investment_ts.py` (downstream of the predictions)

A second, related scale error was found in the same review: `turnover_proxy` mixed
`mid` (fraction) with `half_spread_cost`, and the column labeled "Turnover (bps)" was
actually in percent. Both are fixed.

### Impact

The correction is material and changes published conclusions. On the shipped
2024-05-01 panel, unconditional net Sharpe ratios:

| Structure | SR mid | SR net (before) | SR net (after) |
|---|---|---|---|
| Put Ratio Spread | +1.06 | +0.84 | **−0.61** |
| Risk Reversal | +0.65 | +0.44 | **+0.10** |
| Bear Put Spread | +0.48 | +0.30 | **−0.73** |
| Strangle/Straddle | −0.27 | −0.51 | **−0.97** |
| Iron Butterfly/Condor | −0.56 | −0.96 | **−2.67** |

No structure retains a materially positive net Sharpe ratio. The mean half-spread goes
from 0.022 bp to 2.2 bp, and entry turnover from an apparent 0.24 bp to a true 24 bp
for a two-leg strangle. Conditional results move in the same direction: on the paper's
full sample the put-ratio conditional net Sharpe goes from +0.93 to −0.75 and the
top-three basket from +0.82 to −0.82.

Note that gross columns also move, because the binary target `y = 1{pnl_net > 0}`
depends on the cost. Correcting the cost relabels the training data, so the classifier
learns a different — and weaker — signal. Anyone rescaling the cost only at the
evaluation step, without also rebuilding the target, will get a more favorable number
than the one reported here.

### Scope note on the model zoo

`compute_conditional_model_zoo.py`, `derive_binary_decision_summary.py`, and the
`anchor_zoo` path of `compute_conditional_oos_investment_ts.py` use a flat
`--net-cost 0.005` (0.5 bp). This is correctly scaled but **excludes the bid-ask
half-spread entirely**. It is a documented modeling choice rather than a unit bug, but
since the half-spread averages about 2.2 bp, those "net" figures understate realistic
cost by roughly 5x and should be read as an upper bound.

### Regression guard

`code/analysis/cost_units.py` now provides `assert_percent_of_spot_scale()`, called by
all three cost paths. It fails loudly if a cost series falls outside a plausible
percent-of-spot band, which is the signature of a missing or duplicated `*100`.

### Artifact state

- **Refreshed:** analysis code, `output/tables/*`, `tests/reference/tables/*`,
  `data/conditional_oos_protocol_summary.csv`
- **Stale:** `data/conditional_oos_protocol_predictions.parquet` and the Git LFS data
  panels. The repository's LFS budget is currently exceeded, so LFS objects can be
  neither fetched nor updated. Regenerate locally from the panels once LFS access is
  restored.
- **In progress:** the paper PDF and the AI-context documents under `docs/`.

---

## 2026-05 — Sign error in transaction costs on short days (FIXED)

**Status:** fixed.

**Reported by:** Victor Yoong ([victoryg739](https://github.com/victoryg739)) in
[issue #1](https://github.com/vilkovgr/0dte-strategies/issues/1).

Directional net PNL was computed as `sign * pnl_net`, where `pnl_net` already had
costs subtracted. On short days (`sign = -1`) this flipped the sign of the cost term,
so the half-spread and the 0.5 bp fee were *added* to PNL instead of subtracted:

```python
# wrong
dir_pnl_net = sign * pnl_net          # = -reth_und + half_spread + 0.005 when short

# correct
dir_pnl_net = sign * gross - cost
```

Costs must always reduce PNL regardless of trade direction. The fix propagates gross
and cost terms separately, and gross columns were added throughout so that signal
quality and cost drag can be read independently.
