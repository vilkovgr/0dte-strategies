#!/usr/bin/env python3
"""Build a simple ex-ante VIX-regime conditioning table for 0DTE strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    parser = argparse.ArgumentParser(description="Build VIX-regime conditioning table.")
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
        help="Output .tex file path. Defaults to <root>/output/tables/0dte_vix_regime_1000.tex",
    )
    return parser.parse_args()


def fmt(x: float | int | None, n: int = 4) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def build_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for strategy, label in STRATEGY_LABELS.items():
        temp = df[df["option_type"] == strategy]
        if temp.empty:
            continue

        low = temp[temp["vix_regime"] == "Low"]["reth_und"]
        mid = temp[temp["vix_regime"] == "Mid"]["reth_und"]
        high = temp[temp["vix_regime"] == "High"]["reth_und"]
        stat = stats.ttest_ind(high, low, equal_var=False, nan_policy="omit")

        rows.append(
            {
                "strategy": label,
                "mean_full": fmt(temp["reth_und"].mean(), 4),
                "mean_low": fmt(low.mean(), 4),
                "mean_mid": fmt(mid.mean(), 4),
                "mean_high": fmt(high.mean(), 4),
                "high_minus_low": fmt(high.mean() - low.mean(), 4),
                "tstat_hl": fmt(float(stat.statistic), 2),
                "pval_hl": fmt(float(stat.pvalue), 3),
                "obs": f"{len(temp):,}",
            }
        )
    return rows


def write_latex(rows: list[dict[str, str]], output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Strategy & Mean Full & Mean Low-VIX & Mean Mid-VIX & Mean High-VIX & High-Low & t(H-L) & p(H-L) & Obs \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['mean_full']} & {row['mean_low']} & {row['mean_mid']} & "
            f"{row['mean_high']} & {row['high_minus_low']} & {row['tstat_hl']} & {row['pval_hl']} & {row['obs']} \\\\"
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
    vix_file = data_dir / "vix.parquet"
    output_file = args.output or (tables_dir / "0dte_vix_regime_1000.tex")

    strats = pd.read_parquet(strats_file)
    vix = pd.read_parquet(vix_file)

    strats = strats[strats["option_type"].isin(STRATEGY_LABELS.keys())].copy()
    strats["quote_date"] = pd.to_datetime(strats["quote_date"])
    strats["quote_time"] = strats["quote_time"].astype(str)
    strats = strats[strats["quote_time"] == "10:00:00"].copy()

    by_day = strats.groupby(["quote_date", "option_type"], as_index=False)["reth_und"].mean()

    vix["quote_date"] = pd.to_datetime(vix["quote_date"])
    vix["quote_time"] = vix["quote_time"].astype(str)
    vix10 = vix[vix["quote_time"] == "10:00:00"].copy()
    if "root" in vix10.columns:
        vix10 = vix10[vix10["root"] == "SPXW"]
    # Prefer same-day expiry identified directly by calendar DTE. In newer
    # processed files the dts tag can be -1 even when expiration == quote_date.
    if "dte" in vix10.columns:
        vix10 = vix10[vix10["dte"] == 0]
    elif "dts" in vix10.columns:
        vix10 = vix10[vix10["dts"] == 0]
    vix10 = vix10.groupby("quote_date", as_index=False)["vix"].mean()

    merged = by_day.merge(vix10, how="inner", on="quote_date")
    q33, q67 = merged["vix"].quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    merged["vix_regime"] = np.where(
        merged["vix"] <= q33,
        "Low",
        np.where(merged["vix"] <= q67, "Mid", "High"),
    )

    rows = build_rows(merged)
    write_latex(rows, output_file=output_file)

    print(f"Input strategies: {strats_file}")
    print(f"Input vix: {vix_file}")
    print(f"Output: {output_file}")
    print(f"Rows: {len(rows)}")
    print(f"VIX terciles: q33={q33:.8f}, q67={q67:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
