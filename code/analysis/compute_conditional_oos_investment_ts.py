#!/usr/bin/env python3
"""Build OOS investment time-series outputs for conditional signals.

Two signal sources are supported:
1) strategy-specific protocol signals from compute_conditional_oos_protocol.py
2) a single anchor model from model-zoo outputs

Outputs:
1) A LaTeX table with strategy-level OOS investment diagnostics.
2) A cumulative net-PNL time-series figure for selected strategies and basket.
3) CSV/Parquet helper outputs under <root>/data/.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
if os.environ.get("MPLBACKEND") is None and matplotlib.get_backend().lower() == "macosx":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _paths import get_project_root, get_data_dir, get_tables_dir, get_figures_dir
from compute_conditional_oos_protocol import STRATEGY_LABELS


MODEL_LABELS = {
    "ridge": "Ridge (CV)",
    "ridge_strong": "Ridge (CV Strong)",
    "elastic_net": "Elastic Net",
    "elastic_net_sparse": "Elastic Net (Sparse)",
    "lgbm": "LightGBM",
    "lgbm_deep": "LightGBM (Deep)",
    "rf": "Random Forest",
    "rf_deep": "Random Forest (Deep)",
    "xgb": "XGBoost",
    "xgb_deep": "XGBoost (Deep)",
    "catboost": "CatBoost",
    "catboost_deep": "CatBoost (Deep)",
    "nn_linear_ridge": "Neural Net (Linear + Trainable Ridge)",
    "nn_cnn": "Neural Net (CNN)",
    "nn_relu": "Neural Net (ReLU)",
    "nn_relu_deep": "Neural Net (ReLU Deep)",
    "nn_tanh": "Neural Net (Tanh)",
    "nn_logistic": "Neural Net (Logistic)",
}

FEATURE_SET_LABELS = {
    "baseline": "Baseline",
    "gex": "GEX",
    "flow": "Flow",
    "liquidity": "Liquidity",
    "all": "All Features",
}

PROTOCOL_LABELS = {
    "expanding": "Expanding Window",
    "rolling": "Rolling Window",
}

FIG_TITLE_FONTSIZE = 15
FIG_AXISLABEL_FONTSIZE = 13
FIG_TICK_FONTSIZE = 11
FIG_LEGEND_FONTSIZE_LEFT = 10
FIG_LEGEND_FONTSIZE_RIGHT = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create OOS investment TS figure/table from zoo predictions.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--signal-source",
        choices=["table9", "anchor_zoo"],
        default="table9",
        help=(
            "Signal construction source: "
            "'table9' = strategy-specific benchmark models from conditional_oos_protocol outputs; "
            "'anchor_zoo' = one common anchor model from model-zoo outputs."
        ),
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Model-zoo summary CSV. Defaults to conditional_model_zoo_summary_latest.csv with fallback.",
    )
    parser.add_argument(
        "--pred-file",
        type=Path,
        default=None,
        help="Model-zoo prediction parquet. Defaults to conditional_model_zoo_predictions_latest.parquet with fallback.",
    )
    parser.add_argument(
        "--protocol-summary-file",
        type=Path,
        default=None,
        help=(
            "Strategy-specific protocol summary CSV from compute_conditional_oos_protocol.py. "
            "Defaults to conditional_oos_protocol_summary.csv."
        ),
    )
    parser.add_argument(
        "--protocol-pred-file",
        type=Path,
        default=None,
        help=(
            "Strategy-specific protocol predictions parquet from compute_conditional_oos_protocol.py. "
            "Defaults to conditional_oos_protocol_predictions.parquet."
        ),
    )
    parser.add_argument(
        "--strategy-protocol-select",
        choices=["sr_net", "mean_net_bp"],
        default="sr_net",
        help="Metric used to choose best protocol per strategy when --signal-source table9.",
    )
    parser.add_argument(
        "--table9-protocol",
        choices=["auto", "rolling", "expanding"],
        default="rolling",
        help=(
            "Protocol choice for strategy-specific Table-9 signals. "
            "Default is 'rolling' to match the paper's investment table; "
            "'auto' picks best protocol per strategy by --strategy-protocol-select; "
            "'rolling' or 'expanding' forces one protocol for all strategies."
        ),
    )
    parser.add_argument(
        "--net-cost",
        type=float,
        default=0.005,
        help="Per-trade cost in reth_und units (same convention as model-zoo).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of strategies shown in figure/table (plus equal-weight basket).",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        default=None,
        help="Output figure PDF path.",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=None,
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--daily-out",
        type=Path,
        default=None,
        help="Output daily strategy TS parquet path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output strategy summary CSV path.",
    )
    return parser.parse_args()


def annualized_sharpe(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    vol = s.std()
    if vol == 0 or pd.isna(vol):
        return np.nan
    return float((s.mean() / vol) * np.sqrt(252.0))


def _max_drawdown(cum: pd.Series) -> float:
    if cum.empty:
        return np.nan
    running_max = cum.cummax()
    dd = cum - running_max
    return float(dd.min())


def _fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def resolve_input_file(preferred: Path, fallback: Path) -> Path:
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither file exists: {preferred} or {fallback}")


def select_anchor_spec(summary: pd.DataFrame) -> dict[str, object]:
    ranked = summary.copy()
    ranked["protocol_pref"] = (ranked["protocol"].astype(str) == "expanding").astype(int)
    ranked = ranked.sort_values(
        ["mean_net_bp", "sr_net", "protocol_pref", "hit_rate"],
        ascending=[False, False, False, False],
    )
    top = ranked.iloc[0]
    return {
        "model": str(top["model"]),
        "feature_set": str(top["feature_set"]),
        "scaling": str(top["scaling"]),
        "protocol": str(top["protocol"]),
        "mean_net_bp": float(top["mean_net_bp"]),
        "sr_net": float(top["sr_net"]),
        "hit_rate": float(top["hit_rate"]),
    }


def select_table9_protocols(
    protocol_summary: pd.DataFrame,
    metric: str,
    fixed_protocol: str = "auto",
) -> pd.DataFrame:
    ranked = protocol_summary.copy()
    fixed_protocol = str(fixed_protocol)
    if fixed_protocol in {"rolling", "expanding"}:
        ranked = ranked[ranked["protocol"].astype(str) == fixed_protocol].copy()
        if ranked.empty:
            raise RuntimeError(
                f"No rows available for forced protocol={fixed_protocol} in protocol summary."
            )
    if metric not in {"sr_net", "mean_net_bp"}:
        raise ValueError(f"Unsupported strategy protocol selection metric: {metric}")
    if metric == "sr_net":
        ranked = ranked.sort_values(
            ["sr_net", "mean_net_bp", "hit_rate", "protocol"],
            ascending=[False, False, False, True],
        )
    else:
        ranked = ranked.sort_values(
            ["mean_net_bp", "sr_net", "hit_rate", "protocol"],
            ascending=[False, False, False, True],
        )
    return ranked.groupby("option_type", as_index=False).head(1).reset_index(drop=True)


def make_strategy_summary(
    daily: pd.DataFrame,
    sign_daily: pd.DataFrame,
    selected_strategies: list[str],
) -> pd.DataFrame:
    rows = []
    for strategy in selected_strategies:
        temp = daily[daily["option_type"] == strategy].copy().sort_values("quote_date")
        if temp.empty:
            continue
        temp_sign = sign_daily[sign_daily["option_type"] == strategy].copy()
        s = temp["dir_pnl_net"]
        cum = s.cumsum() * 100.0  # bps cumulative
        q01 = s.quantile(0.01)
        es1 = s[s <= q01].mean() if s.notna().any() else np.nan
        long_share = np.nan
        if not temp_sign.empty:
            long_share = float((temp_sign["sign"].astype(float) > 0).mean())

        rows.append(
            {
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
                "days": int(temp["quote_date"].nunique()),
                "mean_net_bp": float(s.mean() * 100.0),
                "sr_net": annualized_sharpe(s),
                "hit_rate": float((s > 0).mean()),
                "long_share": long_share,
                "es1_bp": float(es1 * 100.0) if pd.notna(es1) else np.nan,
                "worst_day_bp": float(s.min() * 100.0),
                "max_drawdown_bp": _max_drawdown(cum),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_net_bp", ascending=False).reset_index(drop=True)


def select_top_strategies(
    strategy_summary: pd.DataFrame,
    top_k: int,
    metric: str,
) -> list[str]:
    top_k = int(max(1, min(int(top_k), strategy_summary.shape[0])))
    if metric == "mean_net_bp":
        ranked = strategy_summary.sort_values(
            ["mean_net_bp", "sr_net", "hit_rate"],
            ascending=[False, False, False],
        )
    elif metric == "sr_net":
        ranked = strategy_summary.sort_values(
            ["sr_net", "mean_net_bp", "hit_rate"],
            ascending=[False, False, False],
        )
    else:
        raise ValueError(f"Unsupported ranking metric: {metric}")
    return ranked["strategy"].astype(str).head(top_k).tolist()


def make_basket_daily(daily: pd.DataFrame, selected: list[str], option_type: str) -> pd.DataFrame:
    return (
        daily[daily["option_type"].isin(selected)]
        .groupby("quote_date", as_index=False)["dir_pnl_net"]
        .mean()
        .assign(option_type=option_type)
    )


def summarize_basket(
    basket_daily: pd.DataFrame,
    basket_sign_daily: pd.DataFrame,
    strategy: str,
    strategy_label: str,
) -> pd.Series:
    series = basket_daily["dir_pnl_net"]
    cum = series.cumsum() * 100.0
    q01 = series.quantile(0.01)
    long_share = np.nan
    if not basket_sign_daily.empty:
        long_share = float(basket_sign_daily["long_frac"].mean())
    return pd.Series(
        {
            "strategy": strategy,
            "strategy_label": strategy_label,
            "days": int(basket_daily["quote_date"].nunique()),
            "mean_net_bp": float(series.mean() * 100.0),
            "sr_net": annualized_sharpe(series),
            "hit_rate": float((series > 0).mean()),
            "long_share": long_share,
            "es1_bp": float(series[series <= q01].mean() * 100.0),
            "worst_day_bp": float(series.min() * 100.0),
            "max_drawdown_bp": _max_drawdown(cum),
        }
    )


def write_table_tex(
    summary_df: pd.DataFrame,
    basket_top_row: pd.Series,
    basket_top_sr_row: pd.Series,
    basket_all_row: pd.Series,
    output_file: Path,
    top_k: int,
) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Strategy & Mean Net (bps) & SR Net & Hit Rate (\%) & Long Share (\%) & ES$_{1\%}$ (bps) & Worst Day (bps) & Max DD (bps) & Days \\",
        r"\midrule",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.strategy_label} & {_fmt(row.mean_net_bp,3)} & {_fmt(row.sr_net,2)} & "
            f"{_fmt(row.hit_rate * 100.0,1)} & {_fmt(row.long_share * 100.0,1)} & {_fmt(row.es1_bp,2)} & {_fmt(row.worst_day_bp,2)} & "
            f"{_fmt(row.max_drawdown_bp,2)} & {int(row.days)} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        f"Equal-weight basket (Top {int(top_k)} by Mean PNL) & {_fmt(basket_top_row['mean_net_bp'],3)} & {_fmt(basket_top_row['sr_net'],2)} & "
        f"{_fmt(float(basket_top_row['hit_rate']) * 100.0,1)} & {_fmt(float(basket_top_row['long_share']) * 100.0,1)} & {_fmt(basket_top_row['es1_bp'],2)} & "
        f"{_fmt(basket_top_row['worst_day_bp'],2)} & {_fmt(basket_top_row['max_drawdown_bp'],2)} & "
        f"{int(basket_top_row['days'])} \\\\"
    )
    lines.append(
        f"Equal-weight basket (Top {int(top_k)} by SR) & {_fmt(basket_top_sr_row['mean_net_bp'],3)} & {_fmt(basket_top_sr_row['sr_net'],2)} & "
        f"{_fmt(float(basket_top_sr_row['hit_rate']) * 100.0,1)} & {_fmt(float(basket_top_sr_row['long_share']) * 100.0,1)} & {_fmt(basket_top_sr_row['es1_bp'],2)} & "
        f"{_fmt(basket_top_sr_row['worst_day_bp'],2)} & {_fmt(basket_top_sr_row['max_drawdown_bp'],2)} & "
        f"{int(basket_top_sr_row['days'])} \\\\"
    )
    lines.append(
        f"Equal-weight basket (All) & {_fmt(basket_all_row['mean_net_bp'],3)} & {_fmt(basket_all_row['sr_net'],2)} & "
        f"{_fmt(float(basket_all_row['hit_rate']) * 100.0,1)} & {_fmt(float(basket_all_row['long_share']) * 100.0,1)} & {_fmt(basket_all_row['es1_bp'],2)} & "
        f"{_fmt(basket_all_row['worst_day_bp'],2)} & {_fmt(basket_all_row['max_drawdown_bp'],2)} & "
        f"{int(basket_all_row['days'])} \\\\"
    )

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_individual_panel(
    ax: plt.Axes,
    daily_all: pd.DataFrame,
    title: str,
) -> None:
    for strategy in sorted(daily_all["option_type"].unique()):
        label = STRATEGY_LABELS.get(strategy, strategy)
        temp = daily_all[daily_all["option_type"] == strategy].sort_values("quote_date")
        cum_bp = temp["dir_pnl_net"].cumsum() * 100.0
        ax.plot(temp["quote_date"], cum_bp, linewidth=1.7, alpha=0.9, label=label)

    ax.set_title(title, fontsize=FIG_TITLE_FONTSIZE)
    ax.set_xlabel("Date", fontsize=FIG_AXISLABEL_FONTSIZE)
    ax.set_ylabel("Cumulative net PNL (bps)", fontsize=FIG_AXISLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=FIG_TICK_FONTSIZE)
    ax.legend(loc="best", ncol=2, fontsize=FIG_LEGEND_FONTSIZE_LEFT)


def _plot_baskets_panel(
    ax: plt.Axes,
    basket_top_pnl_daily: pd.DataFrame,
    basket_top_sr_daily: pd.DataFrame,
    basket_all_daily: pd.DataFrame,
    top_k: int,
    title: str,
) -> None:
    basket_pnl = basket_top_pnl_daily.sort_values("quote_date")
    basket_pnl_cum_bp = basket_pnl["dir_pnl_net"].cumsum() * 100.0
    ax.plot(
        basket_pnl["quote_date"],
        basket_pnl_cum_bp,
        color="black",
        linewidth=2.8,
        label=f"EW basket (Top {int(top_k)} by Mean PNL)",
    )

    basket_sr = basket_top_sr_daily.sort_values("quote_date")
    basket_sr_cum_bp = basket_sr["dir_pnl_net"].cumsum() * 100.0
    ax.plot(
        basket_sr["quote_date"],
        basket_sr_cum_bp,
        color="#1f77b4",
        linewidth=2.5,
        label=f"EW basket (Top {int(top_k)} by SR)",
    )

    basket_all = basket_all_daily.sort_values("quote_date")
    basket_all_cum_bp = basket_all["dir_pnl_net"].cumsum() * 100.0
    ax.plot(
        basket_all["quote_date"],
        basket_all_cum_bp,
        color="#8c2d04",
        linewidth=2.4,
        linestyle="--",
        label="EW basket (All)",
    )

    ax.set_title(title, fontsize=FIG_TITLE_FONTSIZE)
    ax.set_xlabel("Date", fontsize=FIG_AXISLABEL_FONTSIZE)
    ax.set_ylabel("Cumulative net PNL (bps)", fontsize=FIG_AXISLABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=FIG_TICK_FONTSIZE)
    ax.legend(loc="best", ncol=1, fontsize=FIG_LEGEND_FONTSIZE_RIGHT)


def make_figure(
    daily_all: pd.DataFrame,
    basket_top_pnl_daily: pd.DataFrame,
    basket_top_sr_daily: pd.DataFrame,
    basket_all_daily: pd.DataFrame,
    output_file: Path,
    top_k: int,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6.8), sharey=True)
    _plot_individual_panel(
        ax=axes[0],
        daily_all=daily_all,
        title="Panel A: Individual Strategy Series",
    )
    _plot_baskets_panel(
        ax=axes[1],
        basket_top_pnl_daily=basket_top_pnl_daily,
        basket_top_sr_daily=basket_top_sr_daily,
        basket_all_daily=basket_all_daily,
        top_k=top_k,
        title="Panel B: Equal-Weight Baskets",
    )
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = get_project_root(args.project_root)
    data_dir = get_data_dir(root)
    tables_dir = get_tables_dir(root)
    figures_dir = get_figures_dir(root)

    figure_out = args.figure_out or (figures_dir / "oos_conditional_investment_ts.pdf")
    table_out = args.table_out or (tables_dir / "0dte_conditional_oos_investment.tex")
    daily_out = args.daily_out or (data_dir / "conditional_oos_investment_daily.parquet")
    summary_out = args.summary_out or (data_dir / "conditional_oos_investment_summary.csv")
    signal_source = str(args.signal_source)
    strategy_signal_meta = pd.DataFrame(columns=["option_type", "selected_signal"])
    anchor: dict[str, object] | None = None
    summary_file: Path | None = None
    pred_file: Path | None = None
    protocol_summary_file: Path | None = None
    protocol_pred_file: Path | None = None
    sign_daily: pd.DataFrame | None = None

    if signal_source == "anchor_zoo":
        summary_pref = args.summary_file or (data_dir / "conditional_model_zoo_summary_latest.csv")
        summary_fallback = data_dir / "conditional_model_zoo_summary.csv"
        pred_pref = args.pred_file or (data_dir / "conditional_model_zoo_predictions_latest.parquet")
        pred_fallback = data_dir / "conditional_model_zoo_predictions.parquet"
        summary_file = resolve_input_file(summary_pref, summary_fallback)
        pred_file = resolve_input_file(pred_pref, pred_fallback)

        summary = pd.read_csv(summary_file)
        pred = pd.read_parquet(pred_file)
        pred["quote_date"] = pd.to_datetime(pred["quote_date"])

        anchor = select_anchor_spec(summary=summary)
        mask = (
            (pred["model"].astype(str) == anchor["model"])
            & (pred["feature_set"].astype(str) == anchor["feature_set"])
            & (pred["scaling"].astype(str) == anchor["scaling"])
            & (pred["protocol"].astype(str) == anchor["protocol"])
        )
        anchor_pred = pred[mask].copy()
        if anchor_pred.empty:
            raise RuntimeError("Anchor spec produced no prediction rows.")

        anchor_pred["dir_pnl_net"] = anchor_pred["sign"].astype(float) * (
            anchor_pred["y"].astype(float) - float(args.net_cost)
        )
        daily = (
            anchor_pred.groupby(["quote_date", "option_type"], as_index=False)["dir_pnl_net"]
            .mean()
            .sort_values(["option_type", "quote_date"])
        )
        sign_daily = (
            anchor_pred.groupby(["quote_date", "option_type"], as_index=False)["sign"]
            .mean()
            .sort_values(["option_type", "quote_date"])
        )
        strategy_signal_meta = (
            daily[["option_type"]]
            .drop_duplicates()
            .assign(
                selected_model=MODEL_LABELS.get(str(anchor["model"]), str(anchor["model"])),
                selected_signal=(
                    f"{MODEL_LABELS.get(str(anchor['model']), str(anchor['model']))} | "
                    f"{FEATURE_SET_LABELS.get(str(anchor['feature_set']), str(anchor['feature_set']))} | "
                    f"{PROTOCOL_LABELS.get(str(anchor['protocol']), str(anchor['protocol']))}"
                )
            )
        )
    elif signal_source == "table9":
        protocol_summary_file = args.protocol_summary_file or (data_dir / "conditional_oos_protocol_summary.csv")
        protocol_pred_file = args.protocol_pred_file or (data_dir / "conditional_oos_protocol_predictions.parquet")
        if not protocol_summary_file.exists() or not protocol_pred_file.exists():
            raise FileNotFoundError(
                "Table-9 signal files not found. Run compute_conditional_oos_protocol.py first to create "
                f"{protocol_summary_file} and {protocol_pred_file}."
            )

        protocol_summary = pd.read_csv(protocol_summary_file)
        protocol_pred = pd.read_parquet(protocol_pred_file)
        protocol_pred["quote_date"] = pd.to_datetime(protocol_pred["quote_date"])
        protocol_summary["option_type"] = protocol_summary["option_type"].astype(str)
        protocol_summary["protocol"] = protocol_summary["protocol"].astype(str)
        protocol_pred["option_type"] = protocol_pred["option_type"].astype(str)
        protocol_pred["protocol"] = protocol_pred["protocol"].astype(str)

        selected_protocols = select_table9_protocols(
            protocol_summary=protocol_summary,
            metric=str(args.strategy_protocol_select),
            fixed_protocol=str(args.table9_protocol),
        )
        strategy_signal_meta = selected_protocols[["option_type", "protocol_label", "sr_net", "mean_net_bp"]].copy()
        strategy_signal_meta["selected_model"] = "Logistic benchmark"
        strategy_signal_meta = strategy_signal_meta.rename(
            columns={
                "protocol_label": "selected_signal",
                "sr_net": "selected_signal_sr_net",
                "mean_net_bp": "selected_signal_mean_net_bp",
            }
        )

        keys = selected_protocols[["option_type", "protocol"]].copy()
        pred_sel = protocol_pred.merge(keys, how="inner", on=["option_type", "protocol"])
        if pred_sel.empty:
            raise RuntimeError("No strategy-specific Table-9 predictions after protocol selection.")

        daily = (
            pred_sel.groupby(["quote_date", "option_type"], as_index=False)["dir_pnl_net"]
            .mean()
            .sort_values(["option_type", "quote_date"])
        )
        sign_daily = (
            pred_sel.groupby(["quote_date", "option_type"], as_index=False)["sign"]
            .mean()
            .sort_values(["option_type", "quote_date"])
        )
    else:
        raise ValueError(f"Unsupported signal source: {signal_source}")

    if sign_daily is None:
        raise RuntimeError("Sign series not initialized.")

    all_strategies = sorted(daily["option_type"].astype(str).unique())
    strategy_summary = make_strategy_summary(daily=daily, sign_daily=sign_daily, selected_strategies=all_strategies)
    if not strategy_signal_meta.empty:
        strategy_summary = strategy_summary.merge(
            strategy_signal_meta,
            how="left",
            left_on="strategy",
            right_on="option_type",
        ).drop(columns=["option_type"], errors="ignore")

    top_k = int(max(1, min(args.top_k, strategy_summary.shape[0])))
    selected_pnl = select_top_strategies(strategy_summary=strategy_summary, top_k=top_k, metric="mean_net_bp")
    selected_sr = select_top_strategies(strategy_summary=strategy_summary, top_k=top_k, metric="sr_net")

    basket_top_pnl_daily = make_basket_daily(
        daily=daily,
        selected=selected_pnl,
        option_type="basket_eqw_top_mean",
    )
    basket_top_sr_daily = make_basket_daily(
        daily=daily,
        selected=selected_sr,
        option_type="basket_eqw_top_sr",
    )
    basket_all_daily = make_basket_daily(
        daily=daily,
        selected=all_strategies,
        option_type="basket_eqw_all",
    )
    basket_top_pnl_sign_daily = (
        sign_daily[sign_daily["option_type"].isin(selected_pnl)]
        .assign(long_ind=lambda x: (x["sign"].astype(float) > 0).astype(float))
        .groupby("quote_date", as_index=False)["long_ind"]
        .mean()
        .rename(columns={"long_ind": "long_frac"})
    )
    basket_top_sr_sign_daily = (
        sign_daily[sign_daily["option_type"].isin(selected_sr)]
        .assign(long_ind=lambda x: (x["sign"].astype(float) > 0).astype(float))
        .groupby("quote_date", as_index=False)["long_ind"]
        .mean()
        .rename(columns={"long_ind": "long_frac"})
    )
    basket_all_sign_daily = (
        sign_daily[sign_daily["option_type"].isin(all_strategies)]
        .assign(long_ind=lambda x: (x["sign"].astype(float) > 0).astype(float))
        .groupby("quote_date", as_index=False)["long_ind"]
        .mean()
        .rename(columns={"long_ind": "long_frac"})
    )

    basket_top_row = summarize_basket(
        basket_daily=basket_top_pnl_daily,
        basket_sign_daily=basket_top_pnl_sign_daily,
        strategy="basket_eqw_top_mean",
        strategy_label=f"Equal-weight basket (Top {int(top_k)} by Mean PNL)",
    )
    basket_top_sr_row = summarize_basket(
        basket_daily=basket_top_sr_daily,
        basket_sign_daily=basket_top_sr_sign_daily,
        strategy="basket_eqw_top_sr",
        strategy_label=f"Equal-weight basket (Top {int(top_k)} by SR)",
    )
    basket_all_row = summarize_basket(
        basket_daily=basket_all_daily,
        basket_sign_daily=basket_all_sign_daily,
        strategy="basket_eqw_all",
        strategy_label="Equal-weight basket (All)",
    )

    summary_export = pd.concat(
        [
            strategy_summary,
            pd.DataFrame([basket_top_row.to_dict(), basket_top_sr_row.to_dict(), basket_all_row.to_dict()]),
        ],
        axis=0,
        ignore_index=True,
    )

    daily_all_out = daily.copy()
    if not strategy_signal_meta.empty:
        daily_all_out = daily_all_out.merge(strategy_signal_meta, how="left", on="option_type")
    save_daily = pd.concat(
        [
            daily_all_out.assign(series_type="strategy", signal_source=signal_source),
            basket_top_pnl_daily.assign(series_type="basket_top_mean", signal_source=signal_source),
            basket_top_sr_daily.assign(series_type="basket_top_sr", signal_source=signal_source),
            basket_all_daily.assign(series_type="basket_all", signal_source=signal_source),
        ],
        axis=0,
        ignore_index=True,
    )

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    daily_out.parent.mkdir(parents=True, exist_ok=True)
    summary_export.to_csv(summary_out, index=False)
    save_daily.to_parquet(daily_out, index=False)

    write_table_tex(
        summary_df=strategy_summary,
        basket_top_row=basket_top_row,
        basket_top_sr_row=basket_top_sr_row,
        basket_all_row=basket_all_row,
        output_file=table_out,
        top_k=top_k,
    )
    make_figure(
        daily_all=daily,
        basket_top_pnl_daily=basket_top_pnl_daily,
        basket_top_sr_daily=basket_top_sr_daily,
        basket_all_daily=basket_all_daily,
        output_file=figure_out,
        top_k=top_k,
    )

    print(f"Signal source: {signal_source}")
    print(f"Total strategies available: {daily['option_type'].nunique()}")
    if signal_source == "anchor_zoo":
        assert summary_file is not None and pred_file is not None and anchor is not None
        print(f"Zoo summary used: {summary_file}")
        print(f"Zoo predictions used: {pred_file}")
        model_label = MODEL_LABELS.get(str(anchor["model"]), str(anchor["model"]))
        feature_label = FEATURE_SET_LABELS.get(str(anchor["feature_set"]), str(anchor["feature_set"]))
        protocol_label = PROTOCOL_LABELS.get(str(anchor["protocol"]), str(anchor["protocol"]))
        print("Anchor spec: " f"{model_label} | {feature_label} | {protocol_label}")
    else:
        assert protocol_summary_file is not None and protocol_pred_file is not None
        print(f"Table-9 summary used: {protocol_summary_file}")
        print(f"Table-9 predictions used: {protocol_pred_file}")
        print(f"Per-strategy protocol selection metric: {args.strategy_protocol_select}")
        print(f"Table-9 protocol mode: {args.table9_protocol}")
        if not strategy_signal_meta.empty:
            selected_map = strategy_signal_meta[["option_type", "selected_signal"]].copy()
            print("Selected strategy-specific signals:")
            for row in selected_map.itertuples(index=False):
                print(f"  - {row.option_type}: {row.selected_signal}")
            print("Signal model class: Logistic benchmark (fixed across strategies).")
    print(f"Selected Top-{top_k} by mean PNL: {selected_pnl}")
    print(f"Selected Top-{top_k} by SR: {selected_sr}")
    print(f"Daily TS: {daily_out}")
    print(f"Summary CSV: {summary_out}")
    print(f"Table: {table_out}")
    print(f"Figure: {figure_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
