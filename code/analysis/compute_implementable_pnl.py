#!/usr/bin/env python3
"""Compute implementable-PNL diagnostics for 0DTE strategies."""

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
    parser = argparse.ArgumentParser(description="Build implementable-PNL table.")
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
        help="Output .tex file path. Defaults to <root>/output/tables/0dte_implementable_pnl.tex",
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


def annualized_sharpe(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    vol = s.std()
    if vol == 0 or pd.isna(vol):
        return np.nan
    return (s.mean() / vol) * np.sqrt(252.0)


def fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def build_table(df_daily: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for strategy, label in STRATEGY_LABELS.items():
        temp = df_daily[df_daily["option_type"] == strategy].copy()
        if temp.empty:
            continue

        s_mid = temp["pnl_mid"]
        s_ba = temp["pnl_ba"]
        s_ba_fee05 = temp["pnl_ba_fee05"]

        mean_mid = s_mid.mean()
        mean_ba = s_ba.mean()
        mean_ba_fee05 = s_ba_fee05.mean()

        sr_mid = annualized_sharpe(s_mid)
        sr_ba_fee05 = annualized_sharpe(s_ba_fee05)

        avg_turn = temp["turnover_proxy"].mean()
        avg_turn_bps = avg_turn * 100.0 if pd.notna(avg_turn) else np.nan
        sr_turn = sr_ba_fee05 / avg_turn_bps if avg_turn_bps and avg_turn_bps > 0 else np.nan

        q01 = s_ba_fee05.quantile(0.01)
        tail = s_ba_fee05[s_ba_fee05 <= q01]
        es1 = -tail.mean() if not tail.empty else np.nan
        mean_over_es1 = mean_ba_fee05 / es1 if es1 and es1 > 0 else np.nan

        rows.append(
            {
                "strategy": label,
                "mean_mid": fmt(mean_mid, 4),
                "mean_ba": fmt(mean_ba, 4),
                "mean_ba_fee05": fmt(mean_ba_fee05, 4),
                "sr_mid": fmt(sr_mid, 2),
                "sr_ba_fee05": fmt(sr_ba_fee05, 2),
                "turnover": fmt(avg_turn_bps, 3),
                "sr_turn": fmt(sr_turn, 2),
                "es1": fmt(es1, 4),
                "mean_over_es1": fmt(mean_over_es1, 3),
                "obs": f"{len(temp):,}",
            }
        )
    return rows


def write_latex(rows: list[dict[str, str]], output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Strategy & Mean Mid & Mean B/A & Mean B/A+0.5bp & SR Mid & SR B/A+0.5bp & Turnover (bps) & SR/Turnover & ES$_{1\%}$ & Mean/ES$_{1\%}$ & Obs \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['mean_mid']} & {row['mean_ba']} & {row['mean_ba_fee05']} & "
            f"{row['sr_mid']} & {row['sr_ba_fee05']} & {row['turnover']} & {row['sr_turn']} & "
            f"{row['es1']} & {row['mean_over_es1']} & {row['obs']} \\\\"
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
    output_file = args.output or (tables_dir / "0dte_implementable_pnl.tex")

    strats = pd.read_parquet(strats_file)
    opt = pd.read_parquet(opt_file)

    strats = strats[strats["option_type"].isin(STRATEGY_LABELS.keys())].copy()
    strats["quote_date"] = pd.to_datetime(strats["quote_date"])
    strats["quote_time"] = strats["quote_time"].astype(str)
    strats["mnes"] = strats["mnes"].astype(str)
    strats = strats[strats["quote_time"] == "10:00:00"].copy()

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

    strats["pnl_mid"] = strats["reth_und"].astype(float)
    strats["pnl_ba"] = strats["pnl_mid"] - strats["half_spread_cost"]
    # 0.5bp slippage+fees in percent units (1bp = 0.01).
    strats["pnl_ba_fee05"] = strats["pnl_ba"] - 0.005
    # Turnover proxy: gross premium exchanged at entry, including half-spread.
    strats["turnover_proxy"] = strats["mid"].abs() + strats["half_spread_cost"]

    by_day = (
        strats.groupby(["option_type", "quote_date"], as_index=False)[
            ["pnl_mid", "pnl_ba", "pnl_ba_fee05", "turnover_proxy"]
        ]
        .mean()
    )

    rows = build_table(by_day)
    write_latex(rows, output_file)

    print(f"Input strategies: {strats_file}")
    print(f"Input options: {opt_file}")
    print(f"Output: {output_file}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
