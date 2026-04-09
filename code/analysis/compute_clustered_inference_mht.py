#!/usr/bin/env python3
"""Compute clustered-SE inference table with multiple-testing adjustment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

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

SPECS = [
    {
        "name": "RV only",
        "rhs": "rv",
        "key_var": "rv",
    },
    {
        "name": "RV + RS",
        "rhs": "rv + rsk",
        "key_var": "rsk",
    },
    {
        "name": "IV + IS + RV + RS",
        "rhs": "iv + isk + rv + rsk",
        "key_var": "rsk",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clustered-SE + MHT inference table.")
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
        help="Output .tex file path. Defaults to <root>/output/tables/0dte_inference_cluster_mht.tex",
    )
    return parser.parse_args()


def build_features(vix: pd.DataFrame, ex_post_file: Path) -> pd.DataFrame:
    vix = vix.copy()
    vix["quote_date"] = pd.to_datetime(vix["quote_date"])
    vix["quote_time"] = vix["quote_time"].astype(str)
    vix10 = vix[vix["quote_time"] == "10:00:00"].copy()
    if "root" in vix10.columns:
        vix10 = vix10[vix10["root"] == "SPXW"]
    if "dts" in vix10.columns:
        vix10 = vix10[vix10["dts"] == 0]
    vix10 = vix10.groupby("quote_date", as_index=False)[["vix", "vixup", "vixdn"]].mean()
    vix10["iv"] = vix10["vix"] * 1e5
    vix10["isk"] = (vix10["vixup"] - vix10["vixdn"]) * 1e5

    spx = pd.read_parquet(ex_post_file)
    spx["date"] = pd.to_datetime(spx["date"])
    spx["time"] = spx["time"].astype(str)
    spx10 = spx[spx["time"] == "10:00:00"][["date", "SPX_lrv", "SPX_lrvup", "SPX_lrvdn"]].copy()
    spx10 = spx10.rename(columns={"date": "quote_date"})
    spx10["rv"] = spx10["SPX_lrv"]
    spx10["rsk"] = spx10["SPX_lrvup"] - spx10["SPX_lrvdn"]
    spx10 = spx10[["quote_date", "rv", "rsk"]]

    feat = vix10[["quote_date", "iv", "isk"]].merge(spx10, how="inner", on="quote_date")
    return feat


def fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def write_latex(rows: list[dict[str, str]], output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Strategy & Spec & Coef (key var) & t (clustered) & p-value & q-value (BH) & Obs \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['spec']} & {row['coef']} & {row['tstat']} & {row['pval']} & {row['qval']} & {row['obs']} \\\\"
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
    ex_post_file = data_dir / "future_moments_SPX.parquet"
    output_file = args.output or (tables_dir / "0dte_inference_cluster_mht.tex")

    strats = pd.read_parquet(strats_file)
    vix = pd.read_parquet(vix_file)
    feat = build_features(vix=vix, ex_post_file=ex_post_file)

    strats = strats.copy()
    strats["quote_date"] = pd.to_datetime(strats["quote_date"])
    strats["quote_time"] = strats["quote_time"].astype(str)
    strats = strats[strats["quote_time"] == "10:00:00"]
    strats = strats[strats["option_type"].isin(STRATEGY_LABELS.keys())]
    strats["mnes"] = strats["mnes"].astype(str)

    data = strats.merge(feat, how="inner", on="quote_date")

    raw_results = []
    pvals = []
    for strategy, label in STRATEGY_LABELS.items():
        temp = data[data["option_type"] == strategy].copy()
        if temp.empty:
            continue
        for spec in SPECS:
            rhs = spec["rhs"]
            key_var = spec["key_var"]
            cols = ["reth_und", "quote_date", "mnes", "iv", "isk", "rv", "rsk"]
            reg = temp[cols].dropna().copy()
            if reg["quote_date"].nunique() < 30:
                continue

            formula = f"reth_und ~ {rhs} + C(mnes)"
            mod = smf.ols(formula, data=reg).fit(
                cov_type="cluster",
                cov_kwds={"groups": reg["quote_date"]},
            )

            pval = float(mod.pvalues.get(key_var, np.nan))
            raw_results.append(
                {
                    "strategy": label,
                    "spec": spec["name"],
                    "coef": float(mod.params.get(key_var, np.nan)),
                    "tstat": float(mod.tvalues.get(key_var, np.nan)),
                    "pval": pval,
                    "obs": int(mod.nobs),
                }
            )
            pvals.append(pval)

    if raw_results:
        _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for i, q in enumerate(qvals):
            raw_results[i]["qval"] = float(q)
    else:
        qvals = []

    rows = [
        {
            "strategy": r["strategy"],
            "spec": r["spec"],
            "coef": fmt(r["coef"], 3),
            "tstat": fmt(r["tstat"], 2),
            "pval": fmt(r["pval"], 3),
            "qval": fmt(r.get("qval", np.nan), 3),
            "obs": f"{r['obs']:,}",
        }
        for r in raw_results
    ]
    write_latex(rows=rows, output_file=output_file)

    print(f"Input strategies: {strats_file}")
    print(f"Input vix: {vix_file}")
    print(f"Input realized moments: {ex_post_file}")
    print(f"Output: {output_file}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
