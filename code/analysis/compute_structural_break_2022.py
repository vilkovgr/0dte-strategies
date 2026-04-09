#!/usr/bin/env python3
"""Compute pre/post-2022 structural-break table for 0DTE strategy PNL."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

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
    parser = argparse.ArgumentParser(description="Build pre/post-2022 structural-break table.")
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
        help="Output .tex file path. Defaults to <root>/output/tables/0dte_structbreak_post2022.tex",
    )
    return parser.parse_args()


def significance_stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def fmt_num(value: float | int | None, n: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.{n}f}"


def build_rows(df: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for strategy, label in STRATEGY_LABELS.items():
        temp = df[df["option_type"] == strategy].copy()
        if temp.empty:
            continue

        pre = temp[temp["post2022"] == 0]
        post = temp[temp["post2022"] == 1]
        if pre.empty or post.empty:
            continue

        pre_mean = pre["reth_und"].mean()
        post_mean = post["reth_und"].mean()
        pre_vol = pre["reth_und"].std()
        post_vol = post["reth_und"].std()
        vol_ratio = post_vol / pre_vol if pre_vol and pre_vol > 0 else float("nan")

        combo_dummies = pd.get_dummies(temp["mnes"], prefix="combo", drop_first=True, dtype=float)
        x = pd.concat(
            [pd.Series(1.0, index=temp.index, name="const"), temp["post2022"].astype(float), combo_dummies],
            axis=1,
        )
        model = sm.OLS(temp["reth_und"].astype(float), x)
        # Cluster by quote_date to handle within-day cross-combo dependence.
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": temp["quote_date"]})

        delta = float(fit.params["post2022"])
        t_stat = float(fit.tvalues["post2022"])
        p_val = float(fit.pvalues["post2022"])
        stars = significance_stars(p_val)

        rows.append(
            {
                "strategy": label,
                "pre_mean": fmt_num(pre_mean, 4),
                "post_mean": fmt_num(post_mean, 4),
                "delta": f"{fmt_num(delta, 4)}{stars}",
                "t_stat": fmt_num(t_stat, 2),
                "p_val": fmt_num(p_val, 3),
                "vol_ratio": fmt_num(vol_ratio, 2),
                "n_pre": f"{len(pre)}",
                "n_post": f"{len(post)}",
            }
        )

    return rows


def write_latex(rows: list[dict[str, str]], output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Strategy & Pre-2022 Mean & Post-2022 Mean & $\Delta$ Post-Pre & $t(\Delta)$ & $p$-value & Vol Ratio & N Pre & N Post \\",
        r"\midrule",
    ]

    for row in rows:
        line = (
            f"{row['strategy']} & {row['pre_mean']} & {row['post_mean']} & {row['delta']} & "
            f"{row['t_stat']} & {row['p_val']} & {row['vol_ratio']} & {row['n_pre']} & {row['n_post']} \\\\"
        )
        lines.append(line)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = get_project_root(args.project_root)
    data_dir = get_data_dir(root)
    tables_dir = get_tables_dir(root)

    input_file = data_dir / "data_structures.parquet"
    output_file = args.output or (tables_dir / "0dte_structbreak_post2022.tex")

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_file}")

    df = pd.read_parquet(input_file)
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["quote_time"] = df["quote_time"].astype(str)
    df["mnes"] = df["mnes"].astype(str)

    # Match the main baseline in the paper: 10:00 ET opening.
    df = df[df["quote_time"] == "10:00:00"].copy()
    df = df[df["option_type"].isin(STRATEGY_LABELS.keys())]

    # Structural break around expansion to daily expirations:
    # two additional expiration days launched on 2022-04-18 and 2022-05-11.
    pre_end = pd.Timestamp("2022-04-14")
    post_start = pd.Timestamp("2022-05-11")
    keep = (df["quote_date"] <= pre_end) | (df["quote_date"] >= post_start)
    df = df[keep].copy()
    df["post2022"] = (df["quote_date"] >= post_start).astype(int)

    rows = build_rows(df)
    write_latex(rows, output_file)

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
