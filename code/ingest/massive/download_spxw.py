#!/usr/bin/env python3
"""Download SPXW 0DTE option snapshots and SPX minute bars from Massive.

Usage:
    python code/ingest/massive/download_spxw.py \
        --start 2016-09-01 --end 2026-02-01 \
        --output-dir data/raw/massive

Requires MASSIVE_API_KEY in .env or environment.
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
from client import MassiveClient  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

SPX_INDEX_TICKER = "I:SPX"
VIX_INDEX_TICKER = "I:VIX"


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


def _trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    """Generate weekday dates between start and end (inclusive)."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def download_underlying_bars(client: MassiveClient, output_dir: Path,
                             start: dt.date, end: dt.date) -> None:
    """Download 1-minute SPX and VIX bars."""
    bars_dir = output_dir / "underlying"
    bars_dir.mkdir(parents=True, exist_ok=True)

    for ticker, name in [(SPX_INDEX_TICKER, "SPX"), (VIX_INDEX_TICKER, "VIX")]:
        out_path = bars_dir / f"bars_{name}_1min.parquet"
        if out_path.exists():
            logger.info("Skipping %s — already exists", out_path)
            continue

        logger.info("Downloading %s 1-min bars %s → %s", name, start, end)
        rows = client.list_aggregates(
            ticker=ticker,
            from_date=start.isoformat(),
            to_date=end.isoformat(),
            multiplier=1,
            timespan="minute",
        )
        if not rows:
            logger.warning("No bars returned for %s", name)
            continue

        df = pd.DataFrame(rows)
        df = df.rename(columns={
            "t": "timestamp_ms", "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap", "n": "num_trades",
        })
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["date"] = df["datetime"].dt.date
        df["time"] = df["datetime"].dt.strftime("%H:%M:%S")
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d bars to %s", len(df), out_path)


def download_option_snapshots(client: MassiveClient, output_dir: Path,
                              dates: list[dt.date]) -> None:
    """Download option quotes for each 0DTE expiration date."""
    opts_dir = output_dir / "options"
    opts_dir.mkdir(parents=True, exist_ok=True)

    for exp_date in dates:
        date_str = exp_date.isoformat()
        out_path = opts_dir / f"spxw_{date_str}.parquet"
        if out_path.exists():
            continue

        logger.info("Fetching contracts for %s", date_str)
        contracts = client.list_option_contracts(expiration_date=date_str)
        spxw = [c for c in contracts if c.get("underlying_ticker") == "I:SPX"
                and c.get("ticker", "").startswith("O:SPX")]
        if not spxw:
            logger.debug("No SPXW contracts for %s", date_str)
            continue

        all_quotes: list[dict] = []
        for i, contract in enumerate(spxw):
            ticker = contract["ticker"]
            strike = contract.get("strike_price", 0)
            cp = contract.get("contract_type", "?")

            quotes = client.list_option_quotes(ticker=ticker)
            for q in quotes:
                q["option_ticker"] = ticker
                q["strike"] = strike
                q["option_type"] = cp
                q["expiration"] = date_str
            all_quotes.extend(quotes)

            if (i + 1) % 50 == 0:
                logger.info("  %s: %d/%d contracts fetched", date_str, i + 1, len(spxw))

        if all_quotes:
            df = pd.DataFrame(all_quotes)
            df.to_parquet(out_path, index=False)
            logger.info("Saved %d quotes across %d contracts for %s",
                        len(all_quotes), len(spxw), date_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SPXW 0DTE data from Massive.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "data" / "raw" / "massive",
                        help="Output directory for raw downloads")
    parser.add_argument("--skip-underlying", action="store_true",
                        help="Skip underlying bar download")
    parser.add_argument("--skip-options", action="store_true",
                        help="Skip option quote download")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    _load_dotenv()

    client = MassiveClient()
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_underlying:
        download_underlying_bars(client, args.output_dir, start, end)

    if not args.skip_options:
        dates = _trading_days(start, end)
        download_option_snapshots(client, args.output_dir, dates)

    logger.info("Done. Raw data in %s", args.output_dir)


if __name__ == "__main__":
    main()
