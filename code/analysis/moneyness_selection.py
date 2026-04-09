#!/usr/bin/env python3
"""Helpers for representative moneyness selection in conditional tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RepresentativeSelectionConfig:
    max_moneyness_dev: float = 0.01


def _max_abs_dev(mnes: str) -> float:
    vals = [float(x) for x in str(mnes).split("/")]
    return float(max(abs(v - 1.0) for v in vals))


def choose_representative_moneyness(
    data: pd.DataFrame,
    *,
    strategy_col: str = "option_type",
    mnes_col: str = "mnes",
    date_col: str = "quote_date",
    cfg: RepresentativeSelectionConfig | None = None,
) -> pd.DataFrame:
    """Pick one representative moneyness config per strategy.

    Selection rule:
    1) Restrict to configs with max abs moneyness deviation <= threshold.
    2) Within each strategy, choose config with largest #days, then #rows.
    3) Break remaining ties by smaller max deviation, then lexicographic mnes.
    """

    use_cfg = cfg or RepresentativeSelectionConfig()
    if data.empty:
        return pd.DataFrame(
            columns=[strategy_col, mnes_col, "rows", "days", "max_moneyness_dev"]
        )

    work = data[[strategy_col, mnes_col, date_col]].copy()
    work[strategy_col] = work[strategy_col].astype(str)
    work[mnes_col] = work[mnes_col].astype(str)
    work[date_col] = pd.to_datetime(work[date_col])
    work["max_moneyness_dev"] = work[mnes_col].map(_max_abs_dev)
    work = work[work["max_moneyness_dev"] <= float(use_cfg.max_moneyness_dev)].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[strategy_col, mnes_col, "rows", "days", "max_moneyness_dev"]
        )

    counts = (
        work.groupby([strategy_col, mnes_col], as_index=False)
        .agg(rows=(mnes_col, "size"), days=(date_col, "nunique"))
        .merge(
            work[[strategy_col, mnes_col, "max_moneyness_dev"]]
            .drop_duplicates(subset=[strategy_col, mnes_col]),
            how="left",
            on=[strategy_col, mnes_col],
        )
        .sort_values(
            [strategy_col, "days", "rows", "max_moneyness_dev", mnes_col],
            ascending=[True, False, False, True, True],
        )
    )
    selected = counts.groupby(strategy_col, as_index=False).head(1).reset_index(drop=True)
    return selected[[strategy_col, mnes_col, "rows", "days", "max_moneyness_dev"]]


def apply_representative_filter(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    strategy_col: str = "option_type",
    mnes_col: str = "mnes",
) -> pd.DataFrame:
    """Inner-join data to selected representative strategy/moneyness keys."""

    if data.empty or selected.empty:
        return data.iloc[0:0].copy()

    keys = selected[[strategy_col, mnes_col]].copy()
    work = data.copy()
    work[strategy_col] = work[strategy_col].astype(str)
    work[mnes_col] = work[mnes_col].astype(str)
    keys[strategy_col] = keys[strategy_col].astype(str)
    keys[mnes_col] = keys[mnes_col].astype(str)
    return work.merge(keys, how="inner", on=[strategy_col, mnes_col])
