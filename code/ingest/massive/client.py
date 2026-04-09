"""Massive.com REST client for 0DTE SPXW option data download.

Authentication: set MASSIVE_API_KEY in .env or environment.
Docs: https://massive.com/docs
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RequestConfig:
    timeout_seconds: float = 45.0
    max_retries: int = 6
    backoff_base_seconds: float = 0.75


class MassiveClient:
    """Minimal Massive REST client for SPX option research."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config: RequestConfig | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing MASSIVE_API_KEY. Set it in .env or pass api_key explicitly."
            )
        self.base_url = (
            base_url or os.getenv("MASSIVE_BASE_URL") or "https://api.massive.com"
        ).rstrip("/")
        self.config = config or RequestConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "0dte-strategies/0.1",
        })

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        return f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = self._url(path)
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
                if resp.status_code in RETRYABLE_STATUS_CODES:
                    wait = self.config.backoff_base_seconds * (2 ** attempt)
                    logger.warning("Retryable %s from %s; sleeping %.1fs", resp.status_code, url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt < self.config.max_retries:
                    wait = self.config.backoff_base_seconds * (2 ** attempt)
                    logger.warning("Request error %s; retry %d; sleeping %.1fs", exc, attempt + 1, wait)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Max retries exceeded for {url}")

    def _paginate(self, path: str, params: Dict[str, Any] | None = None,
                  results_key: str = "results") -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        next_url: str | None = path
        current_params = params.copy() if params else None
        while next_url:
            payload = self._get(next_url, params=current_params)
            items.extend(payload.get(results_key, []) or [])
            next_url = payload.get("next_url")
            current_params = None
        return items

    # ── Option contracts ──

    def list_option_contracts(self, *, expiration_date: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """List SPXW option contracts for a given expiration date (YYYY-MM-DD)."""
        return self._paginate(
            "/v3/reference/options/contracts",
            params={
                "expiration_date": expiration_date,
                "expired": "true",
                "order": "asc",
                "sort": "ticker",
                "limit": limit,
            },
        )

    def list_option_quotes(self, *, ticker: str,
                           ts_gte: int | None = None,
                           ts_lte: int | None = None,
                           limit: int = 50_000) -> List[Dict[str, Any]]:
        """Fetch NBBO quotes for an option ticker."""
        encoded = quote(ticker, safe="")
        params: Dict[str, Any] = {"order": "asc", "sort": "timestamp", "limit": limit}
        if ts_gte is not None:
            params["timestamp.gte"] = ts_gte
        if ts_lte is not None:
            params["timestamp.lte"] = ts_lte
        return self._paginate(f"/v3/quotes/{encoded}", params=params)

    # ── Option snapshots (Greeks via snapshot endpoint) ──

    def get_option_snapshot(self, *, underlying: str = "I:SPX",
                            limit: int = 250) -> List[Dict[str, Any]]:
        """Fetch latest option chain snapshot (includes Greeks and IV).

        Uses the /v3/snapshot/options/{underlyingAsset} endpoint which returns
        Greeks, IV, and quote data in a single call for all live contracts.
        """
        encoded = quote(underlying, safe="")
        return self._paginate(
            f"/v3/snapshot/options/{encoded}",
            params={"limit": limit},
        )

    def get_option_contract_details(self, *, ticker: str) -> Dict[str, Any]:
        """Fetch detailed contract info including Greeks for a single option."""
        encoded = quote(ticker, safe="")
        payload = self._get(f"/v3/reference/tickers/{encoded}")
        return payload.get("results", {}) or {}

    # ── Underlying bars ──

    def list_aggregates(self, *, ticker: str, from_date: str, to_date: str,
                        multiplier: int = 1, timespan: str = "minute",
                        limit: int = 50_000) -> List[Dict[str, Any]]:
        """Fetch OHLCV bars for an index or equity ticker."""
        encoded = quote(ticker, safe="")
        payload = self._get(
            f"/v2/aggs/ticker/{encoded}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            params={"adjusted": "true", "sort": "asc", "limit": limit},
        )
        return payload.get("results", []) or []

    def get_daily_bars(self, *, ticker: str, from_date: str, to_date: str,
                       limit: int = 50_000) -> List[Dict[str, Any]]:
        """Fetch daily (EOD) bars for SPX, VIX, etc."""
        return self.list_aggregates(
            ticker=ticker, from_date=from_date, to_date=to_date,
            multiplier=1, timespan="day", limit=limit,
        )
