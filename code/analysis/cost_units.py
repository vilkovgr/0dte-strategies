#!/usr/bin/env python3
"""Unit-scale guards for transaction-cost terms.

The strategy panels mix two scales, which has caused sign- and scale-errors in
published tables before:

* ``bas``, ``mid``, ``tv``  -- fractions of the underlying level
* ``reth_und``, ``reth``    -- percent of the underlying level (built as ``*100``)

Cost terms are combined with ``reth_und``, so they must be expressed in percent
of spot. In that scale 1 basis point equals 0.01. The helpers below fail loudly
when a cost series is off by the characteristic factor of 100.
"""

from __future__ import annotations

import pandas as pd

# Plausible range for a per-structure half-spread in PERCENT of spot.
# SPXW 0DTE quotes imply roughly 0.5-10bp per structure at 10:00 ET; we allow a
# wide band so the guard only trips on a scale error, not on a regime shift.
MIN_HALF_SPREAD_PCT = 0.002  # 0.2bp
MAX_HALF_SPREAD_PCT = 0.50   # 50bp


class CostScaleError(AssertionError):
    """Raised when a cost series is not on the percent-of-spot scale."""


def assert_percent_of_spot_scale(
    cost: pd.Series,
    name: str = "half_spread_cost",
    min_pct: float = MIN_HALF_SPREAD_PCT,
    max_pct: float = MAX_HALF_SPREAD_PCT,
) -> None:
    """Validate that ``cost`` is in percent of spot rather than a fraction.

    Raises CostScaleError with an actionable message if the mean falls outside
    the plausible band, which is the signature of a missing or duplicated *100.
    """
    s = pd.to_numeric(cost, errors="coerce").dropna()
    if s.empty:
        raise CostScaleError(f"{name}: no finite values to validate")

    mean = float(s.mean())
    if mean < min_pct:
        raise CostScaleError(
            f"{name} mean={mean:.6g} percent-of-spot ({mean * 100:.4f}bp) is below the "
            f"plausible floor of {min_pct:.6g} ({min_pct * 100:.4f}bp). This is the "
            "signature of a fraction-vs-percent unit error: `bas` is stored as a "
            "fraction of spot while `reth_und` is in percent, so the spread must be "
            "scaled by 100 before it is combined with reth_und."
        )
    if mean > max_pct:
        raise CostScaleError(
            f"{name} mean={mean:.6g} percent-of-spot ({mean * 100:.4f}bp) exceeds the "
            f"plausible ceiling of {max_pct:.6g} ({max_pct * 100:.4f}bp). Check for a "
            "duplicated *100 scaling."
        )
