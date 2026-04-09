#!/usr/bin/env python3
"""Build a LaTeX table comparing target choices for conditional models.

Compares the same model families across:
1) Direct return prediction (regression target)
2) Binary direction, hard mapping (full +/-1 sign)
3) Binary direction, soft mapping (probability-scaled weight)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FAMILY_MAP = {
    "Ridge": ("ridge", "ridge_logit"),
    "Elastic Net": ("elastic_net", "elastic_net_logit"),
    "Random Forest": ("rf", "rf_clf"),
    "LightGBM": ("lgbm", "lgbm_clf"),
    "XGBoost": ("xgb", "xgb_clf"),
}


def fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return pd.read_csv(path)


def build_table(
    reg_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    soft_df: pd.DataFrame,
) -> list[str]:
    reg = reg_df.set_index("model")
    hard = hard_df.set_index("model")
    soft = soft_df.set_index("model")

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model Family & SR (Return) & Mean (Return, bps) & SR (Hard) & Mean (Hard, bps) & SR (Soft) & Mean (Soft, bps) \\",
        r"\midrule",
    ]

    for label, (reg_key, bin_key) in FAMILY_MAP.items():
        r = reg.loc[reg_key] if reg_key in reg.index else None
        h = hard.loc[bin_key] if bin_key in hard.index else None
        s = soft.loc[bin_key] if bin_key in soft.index else None
        lines.append(
            f"{label} & "
            f"{fmt(None if r is None else r.get('sr_net'), 2)} & "
            f"{fmt(None if r is None else r.get('mean_net_bp'), 3)} & "
            f"{fmt(None if h is None else h.get('sr_net'), 2)} & "
            f"{fmt(None if h is None else h.get('mean_net_bp'), 3)} & "
            f"{fmt(None if s is None else s.get('sr_net'), 2)} & "
            f"{fmt(None if s is None else s.get('mean_net_bp'), 3)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LaTeX table: regression vs binary hard/soft.")
    p.add_argument("--reg-summary", type=Path, required=True)
    p.add_argument("--hard-summary", type=Path, required=True)
    p.add_argument("--soft-summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    reg_df = load_summary(args.reg_summary)
    hard_df = load_summary(args.hard_summary)
    soft_df = load_summary(args.soft_summary)

    lines = build_table(reg_df=reg_df, hard_df=hard_df, soft_df=soft_df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
