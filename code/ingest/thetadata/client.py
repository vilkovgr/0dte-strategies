"""ThetaData local REST client for option and index data download.

ThetaData runs a local Java terminal (ThetaTerminal.jar) that exposes a
REST API on http://127.0.0.1:25510.  Authentication is handled by the
terminal's subscription; no API key is passed per-request.

Set THETADATA_USERNAME and THETADATA_PASSWORD in .env for the terminal
auto-start helper, or start the terminal manually before running scripts.
"""

from __future__ import annotations

import logging
import os
import time
from io import StringIO
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:25510"


class ThetaDataClient:
    """Minimal client for ThetaData's local REST terminal."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.getenv("THETADATA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"

    def is_alive(self) -> bool:
        """Check if the terminal is responding."""
        for endpoint in ["/v2/system/mdds/status", "/v2/system/status"]:
            try:
                resp = self.session.get(self._url(endpoint), timeout=5)
                if resp.ok:
                    return True
            except requests.RequestException:
                continue
        return False

    def wait_for_terminal(self, max_seconds: int = 90) -> bool:
        """Poll until the terminal API is reachable."""
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            if self.is_alive():
                return True
            time.sleep(2)
        return False

    def _get_csv(self, path: str, params: Dict[str, Any]) -> pd.DataFrame:
        """GET endpoint with use_csv=true and return parsed DataFrame."""
        params["use_csv"] = "true"
        url = self._url(path)
        frames: list[pd.DataFrame] = []
        next_page: str | None = None

        while True:
            if next_page and next_page != "null":
                resp = self.session.get(url, params={**params, "next_page": next_page}, timeout=self.timeout)
            else:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()

            text = resp.text.strip()
            if text:
                df = pd.read_csv(StringIO(text))
                if not df.empty:
                    frames.append(df)

            np_header = resp.headers.get("Next-Page") or resp.headers.get("next-page")
            if np_header and np_header != "null":
                next_page = np_header
            else:
                break

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ── Index / equity bars ──

    def get_index_bars(self, *, root: str, start_date: str, end_date: str,
                       ivl_ms: int = 60_000) -> pd.DataFrame:
        """Download intraday bars for an index (SPX, VIX, etc.).

        Tries multiple endpoint versions; returns first success.
        """
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        params = {
            "root": root,
            "start_date": sd,
            "end_date": ed,
            "ivl": ivl_ms,
        }
        endpoints = [
            "/v2/hist/index/ohlc",
            "/v2/hist/index/price",
            "/v3/index/history/ohlc",
        ]
        for ep in endpoints:
            try:
                df = self._get_csv(ep, params.copy())
                if not df.empty:
                    return df
            except requests.HTTPError as exc:
                logger.debug("Endpoint %s returned %s; trying next", ep, exc.response.status_code if exc.response else "?")
                continue
        return pd.DataFrame()

    # ── Option data ──

    def list_expirations(self, root: str = "SPXW") -> List[str]:
        """List available expiration dates for an option root."""
        try:
            df = self._get_csv("/v2/list/expirations", {"root": root})
            if not df.empty and "expiration" in df.columns:
                return df["expiration"].astype(str).tolist()
        except Exception:
            pass
        return []

    def list_strikes(self, *, root: str = "SPXW", expiration: str) -> List[float]:
        """List available strikes for an expiration."""
        try:
            df = self._get_csv("/v2/list/strikes", {"root": root, "exp": expiration.replace("-", "")})
            if not df.empty:
                col = df.columns[0]
                return df[col].tolist()
        except Exception:
            pass
        return []

    def get_option_quotes(self, *, root: str = "SPXW", expiration: str,
                          start_date: str | None = None,
                          end_date: str | None = None,
                          ivl_ms: int = 1_800_000) -> pd.DataFrame:
        """Bulk download option quotes (bid/ask) for all strikes on an expiration.

        ivl_ms=1_800_000 corresponds to 30-minute snapshots (matching Cboe bar convention).
        """
        exp = expiration.replace("-", "")
        params: Dict[str, Any] = {"root": root, "exp": exp, "ivl": ivl_ms}
        if start_date:
            params["start_date"] = start_date.replace("-", "")
        if end_date:
            params["end_date"] = end_date.replace("-", "")

        endpoints = ["/v2/bulk_hist/option/quote", "/v2/bulk_hist/option/trade_quote"]
        for ep in endpoints:
            try:
                df = self._get_csv(ep, params.copy())
                if not df.empty:
                    return df
            except requests.HTTPError:
                continue
        return pd.DataFrame()

    def get_option_greeks(self, *, root: str = "SPXW", expiration: str,
                          start_date: str | None = None,
                          end_date: str | None = None,
                          ivl_ms: int = 1_800_000) -> pd.DataFrame:
        """Bulk download option Greeks for all strikes on an expiration."""
        exp = expiration.replace("-", "")
        params: Dict[str, Any] = {"root": root, "exp": exp, "ivl": ivl_ms}
        if start_date:
            params["start_date"] = start_date.replace("-", "")
        if end_date:
            params["end_date"] = end_date.replace("-", "")

        endpoints = ["/v2/bulk_hist/option/greeks", "/v2/hist/option/greeks"]
        for ep in endpoints:
            try:
                df = self._get_csv(ep, params.copy())
                if not df.empty:
                    return df
            except requests.HTTPError:
                continue
        return pd.DataFrame()

    def get_option_ohlc(self, *, root: str = "SPXW", expiration: str,
                        start_date: str | None = None,
                        end_date: str | None = None,
                        ivl_ms: int = 1_800_000) -> pd.DataFrame:
        """Bulk download option OHLC for all strikes on an expiration."""
        exp = expiration.replace("-", "")
        params: Dict[str, Any] = {"root": root, "exp": exp, "ivl": ivl_ms}
        if start_date:
            params["start_date"] = start_date.replace("-", "")
        if end_date:
            params["end_date"] = end_date.replace("-", "")
        return self._get_csv("/v2/bulk_hist/option/ohlc", params)

    def get_eod_prices(self, *, root: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download daily (EOD) OHLCV for an index or stock."""
        sd = start_date.replace("-", "")
        ed = end_date.replace("-", "")
        params = {"root": root, "start_date": sd, "end_date": ed, "ivl": 86_400_000, "rth": "false"}

        endpoints = ["/v2/hist/index/ohlc", "/v2/hist/stock/ohlc",
                     "/v3/index/history/ohlc", "/v3/stock/history/ohlc"]
        for ep in endpoints:
            try:
                df = self._get_csv(ep, params.copy())
                if not df.empty:
                    return df
            except requests.HTTPError:
                continue
        return pd.DataFrame()
