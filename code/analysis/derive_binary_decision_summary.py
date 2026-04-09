#!/usr/bin/env python3
"""Derive binary model-zoo summary tables for a chosen decision mapping.

This is used to avoid refitting the same binary probability models twice for
hard and soft portfolio mappings. Given stored OOS predictions with `p_hat`,
the script reconstructs the requested position rule and rewrites the summary
CSV and LaTeX tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from compute_conditional_model_zoo import (  # noqa: E402
    _positions_from_probability,
    summarize_predictions,
    write_latex,
    write_latex_compact,
    write_latex_tree_horserace_compact,
)


GROUP_COLS = [
    "model",
    "feature_set",
    "scaling",
    "protocol",
    "n_features",
    "feature_space",
    "ts_mode",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build binary summary tables from stored probabilities.")
    parser.add_argument("--pred-file", type=Path, required=True, help="Prediction parquet with p_hat/y/y_bin.")
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--latex-out", type=Path, required=True)
    parser.add_argument("--latex-compact-out", type=Path, required=True)
    parser.add_argument("--latex-tree-compact-out", type=Path, required=True)
    parser.add_argument("--decision-mode", choices=["hard", "soft"], required=True)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--net-cost", type=float, default=0.005)
    parser.add_argument("--representative-moneyness", action="store_true", default=True)
    parser.add_argument("--max-moneyness-dev", type=float, default=0.01)
    return parser.parse_args()


def build_summary(pred: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = {"quote_date", "y", "y_bin", "p_hat", *GROUP_COLS}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for group_key, grp in pred.groupby(GROUP_COLS, dropna=False, sort=True, observed=True):
        meta = dict(zip(GROUP_COLS, group_key))
        p_hat = grp["p_hat"].to_numpy(dtype=float)
        sign, weight = _positions_from_probability(
            p_hat,
            threshold=float(args.decision_threshold),
            decision_mode=args.decision_mode,
        )
        work = grp.loc[:, ["quote_date", "y", "y_bin", "p_hat"]].copy()
        work["sign"] = sign
        work["weight"] = weight

        smry = summarize_predictions(work, net_cost=float(args.net_cost))
        if not smry:
            continue

        task = str(grp["task"].iloc[0]) if "task" in grp.columns else "binary"
        y_mode = str(grp["y_mode"].iloc[0]) if "y_mode" in grp.columns else "binary"
        rows.append(
            {
                **meta,
                "representative_moneyness": bool(args.representative_moneyness),
                "max_moneyness_dev": float(args.max_moneyness_dev),
                "task": task,
                "y_mode": y_mode,
                "decision_mode": args.decision_mode,
                "decision_threshold": float(args.decision_threshold),
                **smry,
            }
        )

    if not rows:
        raise RuntimeError("No summary rows could be derived from prediction file.")

    return (
        pd.DataFrame(rows)
        .sort_values(["feature_set", "model", "scaling", "protocol"])
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    pred = pd.read_parquet(args.pred_file)
    summary = build_summary(pred=pred, args=args)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.latex_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_out, index=False)
    write_latex(summary=summary, output_file=args.latex_out)
    write_latex_compact(summary=summary, output_file=args.latex_compact_out)
    write_latex_tree_horserace_compact(summary=summary, output_file=args.latex_tree_compact_out)

    print(f"Predictions: {args.pred_file}")
    print(f"Decision mode: {args.decision_mode}")
    print(f"Summary: {args.summary_out}")
    print(f"LaTeX: {args.latex_out}")
    print(f"LaTeX compact: {args.latex_compact_out}")
    print(f"LaTeX tree compact: {args.latex_tree_compact_out}")
    print(f"Rows: {summary.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
