#!/usr/bin/env python3
"""Compute strict out-of-sample conditional diagnostics for 0DTE strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from _paths import get_project_root, get_data_dir, get_tables_dir
from moneyness_selection import (
    RepresentativeSelectionConfig,
    apply_representative_filter,
    choose_representative_moneyness,
)


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
    parser = argparse.ArgumentParser(description="Build strict OOS conditional-protocol table.")
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
        help="Output .tex file path. Defaults to <root>/output/tables/0dte_conditional_oos.tex",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=252,
        help="Minimum initial train window size (trading days).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=252,
        help="Rolling train window size (trading days).",
    )
    parser.add_argument(
        "--representative-moneyness",
        action="store_true",
        default=True,
        help=(
            "Use one representative moneyness configuration per strategy in conditional tests "
            "(default: on)."
        ),
    )
    parser.add_argument(
        "--all-moneyness",
        action="store_false",
        dest="representative_moneyness",
        help="Disable representative-moneyness filter and use all strategy moneyness configurations.",
    )
    parser.add_argument(
        "--max-moneyness-dev",
        type=float,
        default=0.01,
        help=(
            "Maximum absolute deviation from 1.0 across legs for candidate representative "
            "moneyness configurations."
        ),
    )
    parser.add_argument(
        "--rep-moneyness-out",
        type=Path,
        default=None,
        help="Representative strategy-moneyness CSV path. Default: <root>/data/conditional_representative_moneyness.csv",
    )
    parser.add_argument(
        "--rep-moneyness-tex-out",
        type=Path,
        default=None,
        help="Representative strategy-moneyness LaTeX table path. Default: <root>/output/tables/...",
    )
    parser.add_argument(
        "--summary-csv-out",
        type=Path,
        default=None,
        help="Numeric summary CSV path. Default: <root>/data/conditional_oos_protocol_summary.csv",
    )
    parser.add_argument(
        "--pred-out",
        type=Path,
        default=None,
        help="Strategy-protocol prediction parquet path. Default: <root>/data/conditional_oos_protocol_predictions.parquet",
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


def build_feature_frame(vix: pd.DataFrame, slopes: pd.DataFrame, ex_post_file: Path) -> pd.DataFrame:
    vix = vix.copy()
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
    vix10 = vix10.groupby("quote_date", as_index=False)[["vix", "vixup", "vixdn"]].mean()
    vix10["iv"] = vix10["vix"] * 1e5
    vix10["isk"] = (vix10["vixup"] - vix10["vixdn"]) * 1e5
    vix10 = vix10[["quote_date", "iv", "isk"]]

    slopes = slopes.copy()
    slopes["quote_date"] = pd.to_datetime(slopes["quote_date"])
    slopes["quote_time"] = slopes["quote_time"].astype(str)
    slopes10 = slopes[slopes["quote_time"] == "10:00:00"].copy()
    slopes10 = slopes10.groupby("quote_date", as_index=False)[["slope_up", "slope_dn"]].mean()

    spx = pd.read_parquet(ex_post_file)
    spx["date"] = pd.to_datetime(spx["date"])
    spx["time"] = spx["time"].astype(str)
    spx10 = spx[spx["time"] == "10:00:00"][["date", "SPX_lret", "SPX_lrv", "SPX_lrv_skew"]].copy()
    spx10 = spx10.sort_values("date")
    # Strict no-look-ahead: only lagged realized quantities are used as predictors.
    for col in ["SPX_lret", "SPX_lrv", "SPX_lrv_skew"]:
        spx10[col] = spx10[col].shift(1)
    spx10 = spx10.rename(columns={"date": "quote_date"})

    feat = vix10.merge(slopes10, how="inner", on="quote_date")
    feat = feat.merge(spx10, how="inner", on="quote_date")
    feat = feat.sort_values("quote_date")
    return feat


def calibr_slope(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    if len(y_true) < 30:
        return np.nan
    p = np.clip(np.asarray(p_hat, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=float)
    x = np.log(p / (1 - p))
    X = sm.add_constant(x)
    try:
        mod = sm.Logit(y, X).fit(disp=0)
        return float(mod.params[1])
    except Exception:
        # Fallback linear calibration slope.
        A = np.column_stack([np.ones_like(p), p])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(beta[1])


def run_protocol(
    data: pd.DataFrame,
    feature_cols: list[str],
    protocol: str,
    min_train_days: int,
    rolling_window: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    work = data.dropna(subset=feature_cols + ["y", "pnl_net"]).copy()
    work = work.sort_values("quote_date").reset_index(drop=True)
    if len(work) <= min_train_days:
        return pd.DataFrame(), {}

    X = work[feature_cols].to_numpy(dtype=float)
    y = work["y"].to_numpy(dtype=int)
    pnl = work["pnl_net"].to_numpy(dtype=float)
    dt = work["quote_date"].to_numpy()

    preds = []
    for i in range(min_train_days, len(work)):
        if protocol == "expanding":
            start = 0
        elif protocol == "rolling":
            start = max(0, i - rolling_window)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        X_train = X[start:i]
        y_train = y[start:i]
        X_test = X[i : i + 1]

        if len(np.unique(y_train)) < 2:
            p_hat = float(np.mean(y_train))
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
            model.fit(X_train_sc, y_train)
            p_hat = float(model.predict_proba(X_test_sc)[0, 1])

        sign = 1.0 if p_hat >= 0.5 else -1.0
        preds.append(
            {
                "quote_date": dt[i],
                "p_hat": p_hat,
                "y": float(y[i]),
                "sign": sign,
                "pnl_net": pnl[i],
                "dir_pnl_net": sign * pnl[i],
            }
        )

    pred_df = pd.DataFrame(preds)
    if pred_df.empty:
        return pred_df, {}

    y_bin = pred_df["y"].to_numpy(dtype=float)
    sign_true = np.where(y_bin > 0.5, 1.0, -1.0)
    hit_rate = float((pred_df["sign"].to_numpy(dtype=float) == sign_true).mean())
    brier = float(np.mean((pred_df["p_hat"].to_numpy(dtype=float) - y_bin) ** 2))
    slope = calibr_slope(y_true=y_bin, p_hat=pred_df["p_hat"].to_numpy(dtype=float))
    mean_net = float(pred_df["dir_pnl_net"].mean())
    sr_net = float(annualized_sharpe(pred_df["dir_pnl_net"]))

    summary = {
        "hit_rate": hit_rate,
        "brier": brier,
        "calib_slope": slope,
        "mean_net_bp": mean_net * 100.0,  # pnl is in % of underlying; convert to bps.
        "sr_net": sr_net,
        "obs": int(len(pred_df)),
    }
    return pred_df, summary


def fmt(x: float | int | None, n: int = 3) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{x:.{n}f}"


def write_representative_moneyness_latex(selected: pd.DataFrame, output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Strategy & Representative moneyness & Days & Max $|M-1|$ \\",
        r"\midrule",
    ]
    for row in selected.itertuples(index=False):
        strategy = STRATEGY_LABELS.get(str(row.option_type), str(row.option_type))
        lines.append(
            f"{strategy} & {str(row.mnes)} & {int(row.days)} & {fmt(float(row.max_moneyness_dev), 3)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list[dict[str, str]], output_file: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Strategy & Protocol & Hit Rate (\%) & Brier & Calib. Slope & Mean Net (bps) & SR Net & N \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['protocol']} & {row['hit_rate']} & {row['brier']} & "
            f"{row['calib_slope']} & {row['mean_net_bp']} & {row['sr_net']} & {row['obs']} \\\\"
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
    vix_file = data_dir / "vix.parquet"
    slopes_file = data_dir / "slopes.parquet"
    ex_post_file = data_dir / "future_moments_SPX.parquet"
    output_file = args.output or (tables_dir / "0dte_conditional_oos.tex")
    rep_moneyness_out = args.rep_moneyness_out or (data_dir / "conditional_representative_moneyness.csv")
    rep_moneyness_tex_out = (
        args.rep_moneyness_tex_out
        or (tables_dir / "0dte_conditional_representative_moneyness.tex")
    )
    summary_csv_out = args.summary_csv_out or (data_dir / "conditional_oos_protocol_summary.csv")
    pred_out = args.pred_out or (data_dir / "conditional_oos_protocol_predictions.parquet")

    strats = pd.read_parquet(strats_file)
    opt = pd.read_parquet(opt_file)
    vix = pd.read_parquet(vix_file)
    slopes = pd.read_parquet(slopes_file)

    selected_mnes = pd.DataFrame(columns=["option_type", "mnes", "rows", "days", "max_moneyness_dev"])
    if args.representative_moneyness:
        strat_sel_base = strats.copy()
        strat_sel_base["quote_time"] = strat_sel_base["quote_time"].astype(str)
        strat_sel_base = strat_sel_base[
            (strat_sel_base["quote_time"] == "10:00:00")
            & (strat_sel_base["option_type"].astype(str).isin(STRATEGY_LABELS.keys()))
        ][["quote_date", "option_type", "mnes"]].copy()
        selected_mnes = choose_representative_moneyness(
            strat_sel_base,
            cfg=RepresentativeSelectionConfig(max_moneyness_dev=float(args.max_moneyness_dev)),
        )
        if selected_mnes.empty:
            raise RuntimeError(
                f"No representative moneyness configurations found with max deviation <= {args.max_moneyness_dev:.4f}."
            )
        strats = apply_representative_filter(strats, selected_mnes)

    daily = compute_daily_net_pnl(strats=strats, opt=opt)
    feat = build_feature_frame(vix=vix, slopes=slopes, ex_post_file=ex_post_file)

    rows: list[dict[str, str]] = []
    rows_num: list[dict[str, object]] = []
    pred_store: list[pd.DataFrame] = []
    protocol_map = {"expanding": "Expanding (252d)", "rolling": "Rolling (252d)"}
    for strategy, label in STRATEGY_LABELS.items():
        temp = daily[daily["option_type"] == strategy][["quote_date", "pnl_net"]].copy().sort_values("quote_date")
        if temp.empty:
            continue
        temp = temp.merge(feat, how="inner", on="quote_date")
        temp = temp.sort_values("quote_date")
        temp["pnl_l1"] = temp["pnl_net"].shift(1)
        temp["pnl_mean5_l1"] = temp["pnl_net"].shift(1).rolling(5).mean()
        temp["pnl_std5_l1"] = temp["pnl_net"].shift(1).rolling(5).std()
        temp["y"] = (temp["pnl_net"] > 0).astype(int)

        feature_cols = [
            "iv",
            "isk",
            "slope_up",
            "slope_dn",
            "SPX_lret",
            "SPX_lrv",
            "SPX_lrv_skew",
            "pnl_l1",
            "pnl_mean5_l1",
            "pnl_std5_l1",
        ]

        for protocol in ["expanding", "rolling"]:
            pred_df, smry = run_protocol(
                data=temp,
                feature_cols=feature_cols,
                protocol=protocol,
                min_train_days=args.min_train_days,
                rolling_window=args.rolling_window,
            )
            if not smry:
                continue

            rows.append(
                {
                    "strategy": label,
                    "protocol": protocol_map[protocol],
                    "hit_rate": fmt(smry["hit_rate"] * 100.0, 1),
                    "brier": fmt(smry["brier"], 3),
                    "calib_slope": fmt(smry["calib_slope"], 2),
                    "mean_net_bp": fmt(smry["mean_net_bp"], 3),
                    "sr_net": fmt(smry["sr_net"], 2),
                    "obs": f"{smry['obs']:,}",
                }
            )
            rows_num.append(
                {
                    "option_type": strategy,
                    "strategy_label": label,
                    "protocol": protocol,
                    "protocol_label": protocol_map[protocol],
                    "hit_rate": float(smry["hit_rate"]),
                    "brier": float(smry["brier"]),
                    "calib_slope": float(smry["calib_slope"]),
                    "mean_net_bp": float(smry["mean_net_bp"]),
                    "sr_net": float(smry["sr_net"]),
                    "obs": int(smry["obs"]),
                    "min_train_days": int(args.min_train_days),
                    "rolling_window": int(args.rolling_window),
                    "representative_moneyness": bool(args.representative_moneyness),
                    "max_moneyness_dev": float(args.max_moneyness_dev),
                }
            )
            if not pred_df.empty:
                p = pred_df.copy()
                p["option_type"] = strategy
                p["strategy_label"] = label
                p["protocol"] = protocol
                p["protocol_label"] = protocol_map[protocol]
                pred_store.append(p)

    write_latex(rows=rows, output_file=output_file)
    if rows_num:
        summary_csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_num).to_csv(summary_csv_out, index=False)
    if pred_store:
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(pred_store, axis=0, ignore_index=True).to_parquet(pred_out, index=False)
    if not selected_mnes.empty:
        rep_moneyness_out.parent.mkdir(parents=True, exist_ok=True)
        selected_mnes.to_csv(rep_moneyness_out, index=False)
        write_representative_moneyness_latex(selected=selected_mnes, output_file=rep_moneyness_tex_out)

    print(f"Input strategies: {strats_file}")
    print(f"Input options: {opt_file}")
    print(f"Input vix/slopes: {vix_file} / {slopes_file}")
    print(f"Input realized moments: {ex_post_file}")
    print(f"Representative moneyness filter: {bool(args.representative_moneyness)}")
    if bool(args.representative_moneyness):
        print(f"Max |moneyness-1|: {float(args.max_moneyness_dev):.4f}")
        print(f"Representative map CSV: {rep_moneyness_out}")
        print(f"Representative map TeX: {rep_moneyness_tex_out}")
    print(f"Output: {output_file}")
    if rows_num:
        print(f"Summary CSV: {summary_csv_out}")
    if pred_store:
        print(f"Predictions: {pred_out}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
