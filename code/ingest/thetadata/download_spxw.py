#!/usr/bin/env python3
"""Download SPXW 0DTE option snapshots and SPX/VIX minute bars from ThetaData.

Usage:
    python code/ingest/thetadata/download_spxw.py \
        --start 2016-09-01 --end 2026-02-01 \
        --output-dir data/raw/thetadata

Requires a running ThetaTerminal (local REST API on port 25510).
Set THETADATA_USERNAME and THETADATA_PASSWORD in .env for terminal auto-config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import ThetaDataClient  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv() -> None:
    """Load .env from repo root if python-dotenv is available."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def _monthly_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Split a date range into monthly chunks."""
    chunks = []
    current = start
    while current <= end:
        month_end = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
        chunk_end = min(month_end, end)
        chunks.append((current, chunk_end))
        current = chunk_end + dt.timedelta(days=1)
    return chunks


def download_underlying_bars(client: ThetaDataClient, output_dir: Path,
                             start: dt.date, end: dt.date) -> None:
    """Download 1-minute SPX and VIX bars in monthly chunks."""
    bars_dir = output_dir / "underlying"
    bars_dir.mkdir(parents=True, exist_ok=True)

    for symbol in ["SPX", "VIX"]:
        out_path = bars_dir / f"bars_{symbol}_1min.parquet"
        if out_path.exists():
            logger.info("Skipping %s — already exists", out_path)
            continue

        logger.info("Downloading %s 1-min bars %s → %s", symbol, start, end)
        frames = []
        for chunk_start, chunk_end in _monthly_chunks(start, end):
            logger.debug("  %s chunk %s → %s", symbol, chunk_start, chunk_end)
            df = client.get_index_bars(
                root=symbol,
                start_date=chunk_start.isoformat(),
                end_date=chunk_end.isoformat(),
                ivl_ms=60_000,
            )
            if not df.empty:
                frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            logger.info("Saved %d bars to %s", len(combined), out_path)
        else:
            logger.warning("No bars returned for %s", symbol)


def download_option_data(client: ThetaDataClient, output_dir: Path,
                         start: dt.date, end: dt.date) -> None:
    """Download option quotes and Greeks for each 0DTE expiration."""
    opts_dir = output_dir / "options"
    opts_dir.mkdir(parents=True, exist_ok=True)

    expirations = client.list_expirations(root="SPXW")
    if not expirations:
        logger.warning("No expirations returned. Is ThetaTerminal running?")
        return

    for exp_str in expirations:
        try:
            exp_date = dt.date.fromisoformat(exp_str[:10]) if "-" in exp_str else dt.datetime.strptime(exp_str[:8], "%Y%m%d").date()
        except ValueError:
            continue

        if exp_date < start or exp_date > end:
            continue

        out_path = opts_dir / f"spxw_{exp_date.isoformat()}.parquet"
        if out_path.exists():
            continue

        logger.info("Fetching option data for %s", exp_date)

        quotes = client.get_option_quotes(
            root="SPXW",
            expiration=exp_date.isoformat(),
            start_date=exp_date.isoformat(),
            end_date=exp_date.isoformat(),
            ivl_ms=1_800_000,
        )
        greeks = client.get_option_greeks(
            root="SPXW",
            expiration=exp_date.isoformat(),
            start_date=exp_date.isoformat(),
            end_date=exp_date.isoformat(),
            ivl_ms=1_800_000,
        )

        frames = []
        if not quotes.empty:
            frames.append(quotes)
        if not greeks.empty:
            frames.append(greeks)

        if frames:
            if len(frames) == 2:
                merge_cols = [c for c in quotes.columns if c in greeks.columns
                              and c not in ("bid", "ask", "bid_size", "ask_size")]
                if merge_cols:
                    combined = quotes.merge(greeks, on=merge_cols, how="outer", suffixes=("", "_greeks"))
                else:
                    combined = pd.concat(frames, ignore_index=True)
            else:
                combined = frames[0]

            combined["expiration"] = exp_date.isoformat()
            combined.to_parquet(out_path, index=False)
            logger.info("Saved %d rows for %s", len(combined), exp_date)


def download_eod_prices(client: ThetaDataClient, output_dir: Path,
                        start: dt.date, end: dt.date) -> None:
    """Download daily EOD for SPX and VIX."""
    eod_dir = output_dir / "eod"
    eod_dir.mkdir(parents=True, exist_ok=True)

    for symbol in ["SPX", "VIX"]:
        out_path = eod_dir / f"eod_{symbol}.parquet"
        if out_path.exists():
            continue
        df = client.get_eod_prices(root=symbol, start_date=start.isoformat(), end_date=end.isoformat())
        if not df.empty:
            df.to_parquet(out_path, index=False)
            logger.info("Saved %d daily rows for %s", len(df), symbol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SPXW 0DTE data from ThetaData.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "thetadata",
                        help="Output directory for raw downloads")
    parser.add_argument("--base-url", default=None, help="ThetaTerminal REST URL")
    parser.add_argument("--skip-underlying", action="store_true")
    parser.add_argument("--skip-options", action="store_true")
    parser.add_argument("--skip-eod", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    _load_dotenv()

    client = ThetaDataClient(base_url=args.base_url)
    if not client.is_alive():
        logger.info("ThetaTerminal not reachable. Waiting up to 90s...")
        if not client.wait_for_terminal(90):
            logger.error("ThetaTerminal not available. Start it manually or set --base-url.")
            sys.exit(1)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_underlying:
        download_underlying_bars(client, args.output_dir, start, end)

    if not args.skip_options:
        download_option_data(client, args.output_dir, start, end)

    if not args.skip_eod:
        download_eod_prices(client, args.output_dir, start, end)

    logger.info("Done. Raw data in %s", args.output_dir)


if __name__ == "__main__":
    main()
