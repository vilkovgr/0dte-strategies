#!/usr/bin/env python3
"""Compute tail-risk diagnostics for 0DTE strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import get_project_root, get_data_dir, get_tables_dir


STRATEGY_LABELS = {
    "strangle": "Strangle/Straddle",
    "iron_condor": "Iron Butterfly/Condor",
    "risk_reversal": "Risk Reversal",
    "bull_call_spread": "Bull Call Spread",
    "call_ratio_spread": "Call Ratio Spread",
    "bear_put_spread": "Bear Put Spread",
    "put_ratio_spread": "Put Ratio Spread",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tail-risk diagnostics table.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to ODTE_REPO_ROOT or repo root inferred from script location.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file path. Defaults to <root>/output/tables/0dte_tail_risk_diagnostics.tex",
    )
    return parser.parse_args()


def parse_levels(mnes_str: str) -> list[float]:
    return sorted(float(x) for x in str(mnes_str).split("/"))


def mnes_int(level: float) -> int:
    return int(round(level * 1e5))


def get_legs(strategy: str, mnes_str: str) -> list[tuple[str, int, int]]:
    levels = parse_levels(mnes_str)
    if len(levels) < 2:
        return []

    l = levels[0]
    h = levels[-1]

    if strategy == "strangle":
        return [("P", mnes_int(l), 1), ("C", mnes_int(h), 1)]
    if strategy == "risk_reversal":
        return [("P", mnes_int(l), -1), ("C", mnes_int(h), 1)]
    if strategy == "bull_call_spread":
        return [("C", mnes_int(l), 1), ("C", mnes_int(h), -1)]
    if strategy == "call_ratio_spread":
        return [("C", mnes_int(l), 1), ("C", mnes_int(h), -2)]
    if strategy == "bear_put_spread":
        return [("P", mnes_int(l), -1), ("P", mnes_int(h), 1)]
    if strategy == "put_ratio_spread":
        return [("P", mnes_int(l), -2), ("P", mnes_int(h), 1)]
    if strategy == "iron_condor":
        if len(levels) == 3:
            m = levels[1]
            return [("P", mnes_int(l), 1), ("P", mnes_int(m), -1), ("C", mnes_int(m), -1), ("C", mnes_int(h), 1)]
        if len(levels) == 4:
            ml = levels[1]
            mh = levels[2]
            return [("P", mnes_int(l), 1), ("P", mnes_int(ml), -1), ("C", mnes_int(mh), -1), ("C", mnes_int(h), 1)]
    return []


def compute_daily_net_pnl(strats: pd.DataFrame, opt: pd.DataFrame) -> pd.DataFrame:
    strats = strats[strats["option_type"].isin(STRATEGY_LABELS.keys())].copy()
    strats["quote_date"] = pd.to_datetime(strats["quote_date"])
    strats["quote_time"] = strats["quote_time"].astype(str)
    strats["mnes"] = strats["mnes"].astype(str)
    strats = strats[strats["quote_time"] == "10:00:00"].copy()

    opt = opt.copy()
    opt["quote_date"] = pd.to_datetime(opt["quote_date"])
    opt["quote_time"] = opt["quote_time"].astype(str)
    opt = opt[opt["quote_time"] == "10:00:00"][["quote_date", "quote_time", "option_type", "mnes", "bas"]].copy()
    opt["option_type"] = opt["option_type"].astype(str)
    opt["mnes"] = opt["mnes"].astype(int)
    opt = opt.groupby(["quote_date", "quote_time", "option_type", "mnes"], as_index=False)["bas"].mean()
    bas_lookup = {
        (row.quote_date, row.quote_time, row.option_type, int(row.mnes)): float(row.bas)
        for row in opt.itertuples(index=False)
    }

    def calc_half_spread(row: pd.Series) -> float:
        legs = get_legs(str(row["option_type"]), str(row["mnes"]))
        if not legs:
            return np.nan
        total = 0.0
        for otype, m_int, qty in legs:
            key = (row["quote_date"], row["quote_time"], otype, m_int)
            bas = bas_lookup.get(key, np.nan)
            if pd.isna(bas):
                return np.nan
            total += abs(qty) * bas
        return 0.5 * total

    strats["half_spread_cost"] = strats.apply(calc_half_spread, axis=1)
    strats = strats.dropna(subset=["half_spread_cost"]).copy()
    strats["pnl_net"] = strats["reth_und"].astype(float) - strats["half_spread_cost"] - 0.005
    by_day = strats.groupby(["option_type", "quote_date"], as_index=False)["pnl_net"].mean()
    by_day = by_day.sort_values(["option_type", "quote_date"])
    return by_day


def max_drawdown_from_pnl(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    cum = s.cumsum()
    dd = cum - cum.cummax()
    return float(-dd.min())


def expected_shortfall(series: pd.Series, q: float = 0.01) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    cutoff = s.quantile(q)
    tail = s[s <= cutoff]
    if tail.empty:
        return np.nan
    return float(-tail.mean())


def fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def write_latex(rows: list[dict[str, str]], output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Strategy & Skewness & ES$_{1\%}$ & Max DD & Worst Day & Worst 5-Day & Loss Prob (\%) & Obs \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['skew']} & {row['es1']} & {row['maxdd']} & {row['worst_day']} & "
            f"{row['worst_week']} & {row['loss_prob']} & {row['obs']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = get_project_root(args.project_root)
    data_dir = get_data_dir(root)
    tables_dir = get_tables_dir(root)

    strats_file = data_dir / "data_structures.parquet"
    opt_file = data_dir / "data_opt.parquet"
    output_file = args.output or (tables_dir / "0dte_tail_risk_diagnostics.tex")

    strats = pd.read_parquet(strats_file)
    opt = pd.read_parquet(opt_file)
    daily = compute_daily_net_pnl(strats=strats, opt=opt)

    rows: list[dict[str, str]] = []
    for strategy, label in STRATEGY_LABELS.items():
        s = daily[daily["option_type"] == strategy]["pnl_net"].dropna()
        if s.empty:
            continue

        worst_day = float(s.min())
        worst_week = float(s.rolling(5).sum().min()) if len(s) >= 5 else np.nan
        rows.append(
            {
                "strategy": label,
                "skew": fmt(float(s.skew()), 2),
                "es1": fmt(expected_shortfall(s, q=0.01), 4),
                "maxdd": fmt(max_drawdown_from_pnl(s), 3),
                "worst_day": fmt(worst_day, 4),
                "worst_week": fmt(worst_week, 4),
                "loss_prob": fmt(float((s < 0).mean()) * 100.0, 1),
                "obs": f"{len(s):,}",
            }
        )

    write_latex(rows=rows, output_file=output_file)

    print(f"Input strategies: {strats_file}")
    print(f"Input options: {opt_file}")
    print(f"Output: {output_file}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
