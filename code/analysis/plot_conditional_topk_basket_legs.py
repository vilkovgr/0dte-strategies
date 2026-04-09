#!/usr/bin/env python3
"""Plot leg decomposition for conditional equal-weight top-k baskets.

The output is a four-panel figure:
1) Top-k by mean net PNL, construction weights.
2) Top-k by SR net, construction weights.
3) Top-k by mean net PNL, average OOS signal-weighted legs.
4) Top-k by SR net, average OOS signal-weighted legs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import get_project_root, get_data_dir, get_figures_dir
from compute_conditional_oos_investment_ts import select_table9_protocols
from compute_conditional_oos_protocol import STRATEGY_LABELS, get_legs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top-k conditional basket leg decomposition.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to ODTE_REPO_ROOT or repo root inferred from script location.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top strategies in equal-weight basket.",
    )
    parser.add_argument(
        "--strategy-protocol-select",
        choices=["sr_net", "mean_net_bp"],
        default="sr_net",
        help="Protocol selection metric, matching conditional investment pipeline.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Input summary CSV. Defaults to <root>/data/conditional_oos_investment_summary.csv.",
    )
    parser.add_argument(
        "--rep-moneyness-file",
        type=Path,
        default=None,
        help="Representative moneyness CSV. Defaults to <root>/data/conditional_representative_moneyness.csv.",
    )
    parser.add_argument(
        "--protocol-summary-file",
        type=Path,
        default=None,
        help="Protocol summary CSV. Defaults to <root>/data/conditional_oos_protocol_summary.csv.",
    )
    parser.add_argument(
        "--protocol-pred-file",
        type=Path,
        default=None,
        help="Protocol prediction parquet. Defaults to <root>/data/conditional_oos_protocol_predictions.parquet.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <root>/output/figures/topk_conditional_basket_legs.pdf.",
    )
    parser.add_argument(
        "--output-legs-csv",
        type=Path,
        default=None,
        help="Output legs CSV path. Defaults to <root>/data/conditional_topk_basket_legs.csv.",
    )
    parser.add_argument(
        "--m-grid-min",
        type=float,
        default=0.985,
        help="Minimum terminal moneyness for payoff grid.",
    )
    parser.add_argument(
        "--m-grid-max",
        type=float,
        default=1.015,
        help="Maximum terminal moneyness for payoff grid.",
    )
    parser.add_argument(
        "--m-grid-n",
        type=int,
        default=601,
        help="Number of terminal moneyness grid points.",
    )
    return parser.parse_args()


def _payoff(m: np.ndarray, option_side: str, strike_m: float) -> np.ndarray:
    if option_side == "C":
        return np.maximum(m - strike_m, 0.0)
    if option_side == "P":
        return np.maximum(strike_m - m, 0.0)
    raise ValueError(f"Unsupported option side: {option_side}")


def _build_legs_frame(
    top_strategies: list[str],
    rep_mnes: pd.DataFrame,
    mean_sign: pd.Series,
    top_k: int,
) -> pd.DataFrame:
    rep_lookup = {
        str(row.option_type): str(row.mnes)
        for row in rep_mnes.itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for strategy in top_strategies:
        mnes_str = rep_lookup.get(strategy)
        if mnes_str is None:
            continue
        legs = get_legs(strategy=strategy, mnes_str=mnes_str)
        if not legs:
            continue
        for leg_idx, (option_side, m_int, qty) in enumerate(legs, start=1):
            strike_m = float(m_int) / 1e5
            rows.append(
                {
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                    "mnes": mnes_str,
                    "leg_id": leg_idx,
                    "option_side": option_side,
                    "strike_m": strike_m,
                    "qty": int(qty),
                    "w_static": float(qty) / float(top_k),
                    "mean_sign": float(mean_sign.get(strategy, np.nan)),
                    "w_avg_signal": float(qty) * float(mean_sign.get(strategy, np.nan)) / float(top_k),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["leg_label"] = (
        out["strategy_label"]
        + " | "
        + np.where(out["w_static"] >= 0, "Long ", "Short ")
        + out["option_side"].map({"C": "Call", "P": "Put"}).astype(str)
        + " (K="
        + out["strike_m"].round(3).astype(str)
        + ")"
    )
    return out


def _select_top_strategies(summary: pd.DataFrame, top_k: int, metric: str) -> list[str]:
    top_k = int(max(1, min(int(top_k), summary.shape[0])))
    if metric == "mean_net_bp":
        ranked = summary.sort_values(["mean_net_bp", "sr_net", "hit_rate"], ascending=[False, False, False])
    elif metric == "sr_net":
        ranked = summary.sort_values(["sr_net", "mean_net_bp", "hit_rate"], ascending=[False, False, False])
    else:
        raise ValueError(f"Unsupported ranking metric: {metric}")
    return ranked["strategy"].astype(str).head(top_k).tolist()


def _plot_mode(
    ax: plt.Axes,
    m_grid: np.ndarray,
    legs: pd.DataFrame,
    weight_col: str,
    title: str,
    subtitle: str,
) -> None:
    total = np.zeros_like(m_grid)
    for row in legs.itertuples(index=False):
        w = float(getattr(row, weight_col))
        if pd.isna(w) or abs(w) < 1e-12:
            continue
        y = w * _payoff(m_grid, option_side=str(row.option_side), strike_m=float(row.strike_m))
        total += y
        ax.plot(m_grid, y, linewidth=1.7, alpha=0.75, label=f"{row.strategy_label}: {w:+.3f} {row.option_side}@{row.strike_m:.3f}")
    ax.plot(m_grid, total, color="black", linewidth=2.8, linestyle="--", label="Combined payoff")
    ax.axhline(0.0, color="gray", linewidth=0.9, alpha=0.8)
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel(r"Terminal moneyness $S_T/S_0$")
    ax.set_ylabel("Payoff (spot-relative, no premium)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7.5)


def main() -> int:
    args = parse_args()
    root = get_project_root(args.project_root)
    data_dir = get_data_dir(root)
    figures_dir = get_figures_dir(root)

    summary_file = args.summary_file or (data_dir / "conditional_oos_investment_summary.csv")
    rep_mnes_file = args.rep_moneyness_file or (data_dir / "conditional_representative_moneyness.csv")
    protocol_summary_file = args.protocol_summary_file or (data_dir / "conditional_oos_protocol_summary.csv")
    protocol_pred_file = args.protocol_pred_file or (data_dir / "conditional_oos_protocol_predictions.parquet")
    output_figure = args.output_figure or (figures_dir / "topk_conditional_basket_legs.pdf")
    output_legs_csv = args.output_legs_csv or (data_dir / "conditional_topk_basket_legs.csv")

    summary = pd.read_csv(summary_file)
    summary = summary[~summary["strategy"].astype(str).str.startswith("basket_")].copy()
    top_k = int(max(1, min(int(args.top_k), summary.shape[0])))
    top_strategies_pnl = _select_top_strategies(summary=summary, top_k=top_k, metric="mean_net_bp")
    top_strategies_sr = _select_top_strategies(summary=summary, top_k=top_k, metric="sr_net")

    rep_mnes = pd.read_csv(rep_mnes_file)
    rep_mnes["option_type"] = rep_mnes["option_type"].astype(str)

    protocol_summary = pd.read_csv(protocol_summary_file)
    protocol_summary["option_type"] = protocol_summary["option_type"].astype(str)
    protocol_summary["protocol"] = protocol_summary["protocol"].astype(str)
    selected_protocols_all = select_table9_protocols(
        protocol_summary=protocol_summary,
        metric=str(args.strategy_protocol_select),
    )
    pred = pd.read_parquet(protocol_pred_file)
    pred["option_type"] = pred["option_type"].astype(str)
    pred["protocol"] = pred["protocol"].astype(str)
    pred_sel = pred.merge(
        selected_protocols_all[["option_type", "protocol"]],
        how="inner",
        on=["option_type", "protocol"],
    )
    mean_sign = pred_sel.groupby("option_type")["sign"].mean()

    legs_pnl = _build_legs_frame(
        top_strategies=top_strategies_pnl,
        rep_mnes=rep_mnes,
        mean_sign=mean_sign,
        top_k=top_k,
    )
    legs_sr = _build_legs_frame(
        top_strategies=top_strategies_sr,
        rep_mnes=rep_mnes,
        mean_sign=mean_sign,
        top_k=top_k,
    )
    if legs_pnl.empty or legs_sr.empty:
        raise RuntimeError("No leg decomposition built for the selected top-k strategies.")

    legs_pnl = legs_pnl.assign(rank_metric="mean_net_bp")
    legs_sr = legs_sr.assign(rank_metric="sr_net")
    legs = pd.concat([legs_pnl, legs_sr], axis=0, ignore_index=True)
    legs["top_k"] = top_k
    output_legs_csv.parent.mkdir(parents=True, exist_ok=True)
    legs.to_csv(output_legs_csv, index=False)

    m_grid = np.linspace(float(args.m_grid_min), float(args.m_grid_max), int(args.m_grid_n))
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16.5, 11.5), constrained_layout=True)
    _plot_mode(
        ax=axes[0, 0],
        m_grid=m_grid,
        legs=legs_pnl,
        weight_col="w_static",
        title=f"Panel A: Top {top_k} by Mean Net PNL (Construction)",
        subtitle=", ".join(STRATEGY_LABELS.get(s, s) for s in top_strategies_pnl),
    )
    _plot_mode(
        ax=axes[0, 1],
        m_grid=m_grid,
        legs=legs_sr,
        weight_col="w_static",
        title=f"Panel B: Top {top_k} by SR Net (Construction)",
        subtitle=", ".join(STRATEGY_LABELS.get(s, s) for s in top_strategies_sr),
    )
    _plot_mode(
        ax=axes[1, 0],
        m_grid=m_grid,
        legs=legs_pnl,
        weight_col="w_avg_signal",
        title=f"Panel C: Top {top_k} by Mean Net PNL (Average Signal-Weighted)",
        subtitle=", ".join(STRATEGY_LABELS.get(s, s) for s in top_strategies_pnl),
    )
    _plot_mode(
        ax=axes[1, 1],
        m_grid=m_grid,
        legs=legs_sr,
        weight_col="w_avg_signal",
        title=f"Panel D: Top {top_k} by SR Net (Average Signal-Weighted)",
        subtitle=", ".join(STRATEGY_LABELS.get(s, s) for s in top_strategies_sr),
    )
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure, dpi=300)
    plt.close(fig)

    print(f"Top-{top_k} by mean PNL: {top_strategies_pnl}")
    print(f"Top-{top_k} by SR: {top_strategies_sr}")
    print(f"Summary used: {summary_file}")
    print(f"Representative moneyness used: {rep_mnes_file}")
    print(f"Protocol summary used: {protocol_summary_file}")
    print(f"Protocol predictions used: {protocol_pred_file}")
    print(f"Mean signal by strategy:")
    for s in sorted(set(top_strategies_pnl + top_strategies_sr)):
        print(f"  - {s}: {float(mean_sign.get(s, np.nan)):.6f}")
    print(f"Leg decomposition CSV: {output_legs_csv}")
    print(f"Figure: {output_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
