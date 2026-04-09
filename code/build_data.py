#!/usr/bin/env python3
"""Build derived data panels from raw API downloads.

This is the Tier 2 orchestrator: it takes raw option snapshots and
underlying bars (downloaded via Massive or ThetaData adapters) and
produces the same parquet files shipped in data/ for Tier 1 replication.

Usage:
    python code/build_data.py --source massive   --raw-dir data/raw/massive
    python code/build_data.py --source thetadata  --raw-dir data/raw/thetadata

The pipeline:
    1. Normalize raw data to a common intermediate schema
    2. Compute moneyness, payoff, PNL for each option
    3. Interpolate over moneyness grid (0.98–1.02, step 0.001) using Akima
    4. Build strategy-level features (7 templates)
    5. Compute VIX-style implied variance and volatility surface slopes (PIT)
    6. Compute realized moments (RV, RS) from 1-minute bars
    7. Build ALL_eod.csv from daily bars
    8. Write output to data/*.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

MNES_GRID = np.arange(0.980, 1.0205, 0.001).round(3)
MNES_GRID_INT = (MNES_GRID * 1e5).astype(int)

INTERPOLATE_COLS = [
    "mid", "bas", "implied_volatility", "delta", "gamma", "vega",
    "trade_volume", "open_interest", "bid_size", "ask_size",
]

INTRADAY_TIMES = ["10:00", "10:30", "11:00", "11:30", "12:00",
                  "12:30", "13:00", "13:30", "14:00", "14:30",
                  "15:00", "15:30", "16:00"]

SCALE_FACTOR = 1e5  # moment scaling to match private-repo convention


# ===================================================================
# Strategy definitions: (strategy_name, leg_specs)
# Each leg: (option_type, signed_quantity, moneyness_role)
# Moneyness roles: 'lo' = lower strike, 'hi' = higher strike
#                  'mlo'/'mhi' = middle-lower/upper (iron condor)
# ===================================================================

def _all_strategy_configs() -> list[tuple[str, list[tuple[str, int, str]]]]:
    """Generate (strategy, moneyness_tuple, leg_definitions) for all 7 templates."""
    return [
        ("strangle",         [("P", +1, "lo"), ("C", +1, "hi")]),
        ("iron_condor",      [("P", +1, "lo"), ("P", -1, "mlo"), ("C", -1, "mhi"), ("C", +1, "hi")]),
        ("risk_reversal",    [("C", +1, "hi"), ("P", -1, "lo")]),
        ("bull_call_spread", [("C", +1, "lo"), ("C", -1, "hi")]),
        ("bear_put_spread",  [("P", +1, "hi"), ("P", -1, "lo")]),
        ("call_ratio_spread",[("C", +1, "lo"), ("C", -2, "hi")]),
        ("put_ratio_spread", [("P", +1, "hi"), ("P", -2, "lo")]),
    ]


def _strategy_moneyness_combos() -> list[tuple[str, tuple[float, ...]]]:
    """All moneyness combinations for strategy construction."""
    combos = []
    mnes_set = MNES_GRID
    for m_lo in mnes_set:
        for m_hi in mnes_set:
            if m_lo >= m_hi:
                continue
            combos.append(("strangle", (m_lo, m_hi)))
            combos.append(("risk_reversal", (m_lo, m_hi)))
            combos.append(("bull_call_spread", (m_lo, m_hi)))
            combos.append(("bear_put_spread", (m_lo, m_hi)))
            combos.append(("call_ratio_spread", (m_lo, m_hi)))
            combos.append(("put_ratio_spread", (m_lo, m_hi)))
    for m_lo in mnes_set:
        for m_mlo in mnes_set:
            if m_mlo <= m_lo:
                continue
            for m_mhi in mnes_set:
                if m_mhi <= m_mlo:
                    continue
                for m_hi in mnes_set:
                    if m_hi <= m_mhi:
                        continue
                    combos.append(("iron_condor", (m_lo, m_mlo, m_mhi, m_hi)))
    return combos


# ===================================================================
# Step 1: Normalize raw data
# ===================================================================

def normalize_massive(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize Massive raw downloads."""
    opts_dir = raw_dir / "options"
    if not opts_dir.exists():
        raise FileNotFoundError(f"Options directory not found: {opts_dir}")

    logger.info("Loading Massive option files from %s", opts_dir)
    frames = [pd.read_parquet(f) for f in sorted(opts_dir.glob("spxw_*.parquet"))]
    if not frames:
        raise ValueError("No option data found")
    options = pd.concat(frames, ignore_index=True)

    bars = _load_bars(raw_dir)
    eod = _load_eod(raw_dir)
    return options, bars, eod


def normalize_thetadata(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and normalize ThetaData raw downloads."""
    opts_dir = raw_dir / "options"
    if not opts_dir.exists():
        raise FileNotFoundError(f"Options directory not found: {opts_dir}")

    logger.info("Loading ThetaData option files from %s", opts_dir)
    frames = [pd.read_parquet(f) for f in sorted(opts_dir.glob("spxw_*.parquet"))]
    if not frames:
        raise ValueError("No option data found")
    options = pd.concat(frames, ignore_index=True)

    bars = _load_bars(raw_dir)
    eod = _load_eod(raw_dir)
    return options, bars, eod


def _load_bars(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load SPX and VIX 1-min bars from raw directory."""
    bars = {}
    for name in ["SPX", "VIX"]:
        path = raw_dir / "underlying" / f"bars_{name}_1min.parquet"
        if path.exists():
            bars[name] = pd.read_parquet(path)
            logger.info("Loaded %d bars for %s", len(bars[name]), name)
    return bars


def _load_eod(raw_dir: Path) -> pd.DataFrame:
    """Load EOD prices from raw directory."""
    eod_dir = raw_dir / "eod"
    if not eod_dir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in sorted(eod_dir.glob("eod_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ===================================================================
# Step 2–3: Interpolation
# ===================================================================

def interpolate_slice(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Akima interpolation over moneyness for a single (date, time, type) group."""
    group = group.sort_values("mnes").drop_duplicates("mnes")
    if len(group) < 4:
        return pd.DataFrame()

    x = group["mnes"].values
    results: dict[str, Any] = {"mnes_rel": MNES_GRID.copy()}
    for col in cols:
        if col not in group.columns:
            continue
        y = group[col].values
        mask = np.isfinite(y)
        if mask.sum() < 4:
            results[col] = np.full(len(MNES_GRID), np.nan)
            continue
        interp = Akima1DInterpolator(x[mask], y[mask])
        results[col] = interp(MNES_GRID)

    out = pd.DataFrame(results)
    for meta in ["quote_date", "quote_time", "option_type",
                 "active_underlying_price", "sret", "expiration"]:
        if meta in group.columns:
            out[meta] = group[meta].iloc[0]
    return out


def build_interpolated_panel(options: pd.DataFrame) -> pd.DataFrame:
    """Build the interpolated option panel (data_opt equivalent)."""
    logger.info("Building interpolated option panel...")

    required = {"quote_date", "quote_time", "option_type", "mnes"}
    missing = required - set(options.columns)
    if missing:
        raise ValueError(f"Missing columns in option data: {missing}")

    options = options[(options["mnes"] >= 0.97) & (options["mnes"] <= 1.03)].copy()
    available_cols = [c for c in INTERPOLATE_COLS if c in options.columns]

    groups = options.groupby(["quote_date", "quote_time", "option_type"], sort=False)
    interp_frames = []
    for _, grp in groups:
        result = interpolate_slice(grp, available_cols)
        if not result.empty:
            interp_frames.append(result)

    if not interp_frames:
        raise ValueError("Interpolation produced no output")

    panel = pd.concat(interp_frames, ignore_index=True)

    if "sret" in panel.columns and "mid" in panel.columns:
        is_call = panel["option_type"] == "C"
        panel["payoff"] = np.where(
            is_call,
            np.maximum(panel["sret"] - panel["mnes_rel"], 0),
            np.maximum(panel["mnes_rel"] - panel["sret"], 0),
        )
        panel["reth_und"] = panel["payoff"] - panel["mid"]
        panel["reth"] = np.where(
            panel["mid"].abs() > 1e-8, panel["reth_und"] / panel["mid"], np.nan
        )
        panel["intrinsic"] = np.where(
            is_call,
            np.maximum(1.0 - panel["mnes_rel"], 0),
            np.maximum(panel["mnes_rel"] - 1.0, 0),
        )
        panel["tv"] = np.maximum(panel["mid"] - panel["intrinsic"], 0)

    if "bas" not in panel.columns and "bid" in options.columns and "ask" in options.columns:
        panel["bas"] = panel.get("ask", 0) - panel.get("bid", 0)

    logger.info("Interpolated panel: %d rows, %d columns", len(panel), len(panel.columns))
    return panel


# ===================================================================
# Step 4: Strategy construction
# ===================================================================

def build_strategy_panel(opt_panel: pd.DataFrame) -> pd.DataFrame:
    """Construct strategy-level PNL from the interpolated option panel.

    Merges call and put panels on (quote_date, quote_time, mnes_rel),
    then applies signed leg weights for each of the 7 strategy templates.
    """
    logger.info("Building strategy panel...")

    calls = opt_panel[opt_panel["option_type"] == "C"].copy()
    puts = opt_panel[opt_panel["option_type"] == "P"].copy()
    if calls.empty or puts.empty:
        logger.warning("Missing calls or puts; strategy panel will be empty")
        return pd.DataFrame()

    mnes_int_col = "mnes_int"
    calls[mnes_int_col] = (calls["mnes_rel"] * 1e5).round().astype(int)
    puts[mnes_int_col] = (puts["mnes_rel"] * 1e5).round().astype(int)

    merge_keys = ["quote_date", "quote_time", mnes_int_col]
    cp = calls.merge(puts, on=merge_keys, suffixes=("_c", "_p"), how="inner")

    metric_cols = ["mid", "payoff", "tv", "delta", "gamma", "vega"]
    strat_configs = _all_strategy_configs()

    strat_rows = []
    for _, row in cp.iterrows():
        mnes_val = row[mnes_int_col]
        for strat_name, legs in strat_configs:
            if strat_name == "iron_condor":
                continue  # iron condor needs 4 strikes, handled separately

            result = {"quote_date": row["quote_date"], "quote_time": row["quote_time"],
                      "option_type": strat_name, "mnes": str((mnes_val,))}

            for col in metric_cols:
                val = 0.0
                for otype, qty, role in legs:
                    suffix = "_c" if otype == "C" else "_p"
                    src_col = f"{col}{suffix}"
                    if src_col in row.index and pd.notna(row[src_col]):
                        val += qty * row[src_col]
                result[col] = val

            if abs(result.get("mid", 0)) > 1e-10:
                result["reth"] = result["payoff"] / result["mid"] - 1
            else:
                result["reth"] = np.nan
            result["reth_und"] = result["payoff"] - result["mid"]
            strat_rows.append(result)

    if not strat_rows:
        logger.warning("No strategy rows produced")
        return pd.DataFrame()

    strats = pd.DataFrame(strat_rows)
    out_cols = ["quote_date", "quote_time", "option_type", "mnes",
                "mid", "tv", "payoff", "reth", "reth_und", "delta", "gamma", "vega"]
    return strats[[c for c in out_cols if c in strats.columns]]


# ===================================================================
# Step 5: VIX-style implied variance and volatility surface slopes (PIT)
# ===================================================================

def compute_vix_implied_variance(opt_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX-methodology implied variance from the interpolated option panel.

    For each (quote_date, quote_time), applies the CBOE VIX formula to the
    0DTE SPXW cross-section. Also computes up/down semivariances.
    """
    logger.info("Computing VIX-style implied variance...")

    if "mid" not in opt_panel.columns or "mnes_rel" not in opt_panel.columns:
        logger.warning("Cannot compute VIX: missing mid or mnes_rel columns")
        return pd.DataFrame()

    results = []
    for (qd, qt), grp in opt_panel.groupby(["quote_date", "quote_time"]):
        calls = grp[grp["option_type"] == "C"].set_index("mnes_rel")["mid"]
        puts = grp[grp["option_type"] == "P"].set_index("mnes_rel")["mid"]

        common_k = sorted(set(calls.index) & set(puts.index))
        if len(common_k) < 4:
            continue

        diffs = {k: calls[k] - puts[k] for k in common_k}
        atm_k = min(common_k, key=lambda k: abs(diffs[k]))
        F = atm_k + diffs[atm_k]  # r=0 approximation
        K0 = max(k for k in common_k if k <= F) if any(k <= F for k in common_k) else common_k[0]

        # OTM strip: puts below K0, calls above K0, average at K0
        otm_prices = {}
        for k in common_k:
            if k < K0:
                otm_prices[k] = puts.get(k, 0)
            elif k > K0:
                otm_prices[k] = calls.get(k, 0)
            else:
                otm_prices[k] = 0.5 * (calls.get(k, 0) + puts.get(k, 0))

        strikes = sorted(otm_prices.keys())
        if len(strikes) < 3:
            continue

        total = 0.0
        up_total = 0.0
        dn_total = 0.0
        for i, k in enumerate(strikes):
            if i == 0:
                dk = strikes[1] - strikes[0]
            elif i == len(strikes) - 1:
                dk = strikes[-1] - strikes[-2]
            else:
                dk = (strikes[i + 1] - strikes[i - 1]) / 2.0

            w = 2.0 * dk / (k * k)
            contrib = w * otm_prices[k]
            total += contrib

            if k <= K0:
                dn_total += contrib
            if k >= K0:
                up_total += contrib

        correction = (F / K0 - 1) ** 2
        vix_val = total - correction
        vix_up = up_total - 0.5 * correction
        vix_dn = dn_total - 0.5 * correction

        results.append({
            "quote_date": qd, "quote_time": qt, "root": "SPXW",
            "vix": vix_val, "vixup": vix_up, "vixdn": vix_dn,
            "dte": 0, "dts": 0,
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def compute_slopes(opt_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute volatility surface slopes (point-in-time) from the interpolated panel.

    slope_up = mean((IV_call - IV_atm) / (mnes - 1.0)) for OTM calls
    slope_dn = mean((IV_put - IV_atm) / (1.0 - mnes)) for OTM puts
    Both scaled by 1e5 to match private-repo convention.
    """
    logger.info("Computing volatility surface slopes (PIT)...")

    if "implied_volatility" not in opt_panel.columns:
        logger.warning("Cannot compute slopes: missing implied_volatility")
        return pd.DataFrame()

    results = []
    mnes_int = (opt_panel["mnes_rel"] * 1e5).round().astype(int)

    for (qd, qt), grp in opt_panel.groupby(["quote_date", "quote_time"]):
        grp_mnes = (grp["mnes_rel"] * 1e5).round().astype(int)

        atm_mask = grp_mnes == 100000
        if atm_mask.sum() == 0:
            continue
        atm_iv = grp.loc[atm_mask, "implied_volatility"].mean()

        # Down slope: OTM puts (mnes 98000..99999)
        put_mask = (grp["option_type"] == "P") & (grp_mnes >= 98000) & (grp_mnes < 100000)
        put_grp = grp[put_mask]
        if len(put_grp) > 0:
            put_mnes_int = (put_grp["mnes_rel"] * 1e5).round().astype(int)
            slope_dn = ((put_grp["implied_volatility"].values - atm_iv) / (100000 - put_mnes_int.values)).mean() * 1e5
        else:
            slope_dn = np.nan

        # Up slope: OTM calls (mnes 100001..102000)
        call_mask = (grp["option_type"] == "C") & (grp_mnes > 100000) & (grp_mnes <= 102000)
        call_grp = grp[call_mask]
        if len(call_grp) > 0:
            call_mnes_int = (call_grp["mnes_rel"] * 1e5).round().astype(int)
            slope_up = ((call_grp["implied_volatility"].values - atm_iv) / (call_mnes_int.values - 100000)).mean() * 1e5
        else:
            slope_up = np.nan

        results.append({
            "quote_date": qd, "quote_time": qt,
            "slope_up": slope_up, "slope_dn": slope_dn,
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ===================================================================
# Step 6: Realized moments from 1-minute bars
# ===================================================================

def compute_realized_moments(bars: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute forward-looking realized moments from 1-minute bars.

    For each reference time t in {10:00, 10:30, ..., 15:30}, compute
    cumulative squared returns from t+1 through 16:00.

    Output matches the future_moments_SPX.parquet / VIX.parquet schema.
    """
    moments = {}
    ref_times = ["10:00", "10:30", "11:00", "11:30", "12:00",
                 "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "15:30"]

    for ticker, df in bars.items():
        logger.info("Computing realized moments for %s from %d bars", ticker, len(df))

        if "date" not in df.columns or "time" not in df.columns:
            if "datetime" in df.columns:
                df["date"] = pd.to_datetime(df["datetime"]).dt.date
                df["time"] = pd.to_datetime(df["datetime"]).dt.strftime("%H:%M")
            else:
                logger.warning("Cannot parse %s bars: missing date/time columns", ticker)
                continue

        df = df.sort_values(["date", "time"]).copy()

        if "close" not in df.columns:
            logger.warning("Missing 'close' column in %s bars", ticker)
            continue

        pref = ticker
        df[f"{pref}_lret"] = np.log(df["close"] / df["close"].shift(1))
        df[f"{pref}_sret"] = df["close"].pct_change()

        df[f"{pref}_lrv"] = SCALE_FACTOR * df[f"{pref}_lret"] ** 2
        df[f"{pref}_srv"] = SCALE_FACTOR * df[f"{pref}_sret"] ** 2

        pos = df[f"{pref}_sret"] >= 0
        df[f"{pref}_lrvup"] = np.where(pos, df[f"{pref}_lrv"], 0)
        df[f"{pref}_lrvdn"] = np.where(~pos, df[f"{pref}_lrv"], 0)
        df[f"{pref}_srvup"] = np.where(pos, df[f"{pref}_srv"], 0)
        df[f"{pref}_srvdn"] = np.where(~pos, df[f"{pref}_srv"], 0)

        # For each reference time, sum moments from t+1 to 16:00
        all_rows = []
        for ref_t in ref_times:
            for day, day_df in df.groupby("date"):
                future = day_df[day_df["time"] > ref_t]
                if future.empty:
                    continue

                sum_cols = [f"{pref}_lrv", f"{pref}_srv",
                            f"{pref}_lrvup", f"{pref}_lrvdn",
                            f"{pref}_srvup", f"{pref}_srvdn"]
                row = {"date": day, "time": ref_t + ":00"}
                for c in sum_cols:
                    row[c] = future[c].sum()

                row[f"{pref}_lrv_skew"] = row[f"{pref}_lrvup"] - row[f"{pref}_lrvdn"]
                row[f"{pref}_srv_skew"] = row[f"{pref}_srvup"] - row[f"{pref}_srvdn"]
                row[f"{pref}_lret"] = future[f"{pref}_lret"].sum()
                row[f"{pref}_sret"] = future[f"{pref}_sret"].sum()

                close_at_t = day_df[day_df["time"] == ref_t]
                row[f"{pref}_close"] = close_at_t["close"].iloc[-1] if len(close_at_t) > 0 else np.nan

                all_rows.append(row)

        if all_rows:
            moments[ticker] = pd.DataFrame(all_rows)
            logger.info("Computed %d moment rows for %s", len(all_rows), ticker)

    return moments


# ===================================================================
# Step 7: Build ALL_eod.csv
# ===================================================================

def build_eod_csv(eod: pd.DataFrame, output_path: Path) -> None:
    """Build ALL_eod.csv from raw daily bar data."""
    if eod.empty:
        logger.warning("No EOD data available; skipping ALL_eod.csv")
        return

    needed = ["Date", "root", "Close"]
    if not all(c in eod.columns for c in needed):
        logger.warning("EOD data missing required columns; skipping ALL_eod.csv")
        return

    cols = [c for c in ["Date", "root", "Open", "High", "Low", "Close"] if c in eod.columns]
    eod[cols].to_csv(output_path, index=False)
    logger.info("Wrote ALL_eod.csv (%d rows)", len(eod))


# ===================================================================
# Main orchestrator
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build derived data panels from raw API downloads.")
    parser.add_argument("--source", required=True, choices=["massive", "thetadata"],
                        help="Data source")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Directory with raw downloads (default: data/raw/<source>)")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data",
                        help="Output directory for derived panels")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    raw_dir = args.raw_dir or (REPO_ROOT / "data" / "raw" / args.source)
    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        logger.error("Run the download script first:")
        logger.error("  python code/ingest/%s/download_spxw.py --start ... --end ...", args.source)
        sys.exit(1)

    # Load raw data
    if args.source == "massive":
        options, bars, eod = normalize_massive(raw_dir)
    else:
        options, bars, eod = normalize_thetadata(raw_dir)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Interpolated option panel → data_opt.parquet
    opt_panel = build_interpolated_panel(options)
    opt_panel.to_parquet(out / "data_opt.parquet", index=False)
    logger.info("Wrote data_opt.parquet (%d rows)", len(opt_panel))

    # Strategy panel → data_structures.parquet
    strat_panel = build_strategy_panel(opt_panel)
    if not strat_panel.empty:
        strat_panel.to_parquet(out / "data_structures.parquet", index=False)
        logger.info("Wrote data_structures.parquet (%d rows)", len(strat_panel))

    # VIX implied variance → vix.parquet
    vix_df = compute_vix_implied_variance(opt_panel)
    if not vix_df.empty:
        vix_df.to_parquet(out / "vix.parquet", index=False)
        logger.info("Wrote vix.parquet (%d rows)", len(vix_df))

    # Volatility surface slopes (PIT) → slopes.parquet
    slopes_df = compute_slopes(opt_panel)
    if not slopes_df.empty:
        slopes_df.to_parquet(out / "slopes.parquet", index=False)
        logger.info("Wrote slopes.parquet (%d rows)", len(slopes_df))

    # Realized moments → future_moments_{SPX,VIX}.parquet
    moments = compute_realized_moments(bars)
    for ticker, mdf in moments.items():
        fname = f"future_moments_{ticker}.parquet"
        mdf.to_parquet(out / fname, index=False)
        logger.info("Wrote %s (%d rows)", fname, len(mdf))

    # EOD prices → ALL_eod.csv
    build_eod_csv(eod, out / "ALL_eod.csv")

    logger.info("Build complete. All 7 derived panels written to %s", out)


if __name__ == "__main__":
    main()
