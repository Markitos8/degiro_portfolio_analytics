from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from degiro_analytics.settings import EODHD_API_KEY, FULL_HISTORY_START, POLYGON_API_KEY, PRICE_DB_PATH

logger = logging.getLogger(__name__)


def _ensure_price_db(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eod_prices (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close REAL,
            source TEXT,
            updated_at TEXT,
            PRIMARY KEY (date, ticker)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS eod_meta (
            ticker TEXT PRIMARY KEY,
            first_date TEXT,
            last_date TEXT,
            source TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def _yahoo_to_eodhd_symbol(ticker: str) -> Optional[str]:
    if ticker.startswith("^") or "=X" in ticker:
        return None
    if "." in ticker:
        parts = ticker.rsplit(".", 1)
        return f"{parts[0]}.{parts[1].upper()}"
    return f"{ticker}.US"


def _fetch_eodhd(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    if not EODHD_API_KEY:
        return None, "eodhd_no_key"
    symbol = _yahoo_to_eodhd_symbol(ticker)
    if symbol is None:
        return None, "eodhd_skip_symbol"
    url = (
        f"https://eodhd.com/api/eod/{symbol}"
        f"?from={start_date}&to={end_date}&api_token={EODHD_API_KEY}&fmt=json"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            return None, "eodhd_empty"
        df = pd.DataFrame(payload)
        s = pd.Series(df["adjusted_close"].values, index=pd.to_datetime(df["date"]))
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = ticker
        return (s if not s.empty else None), "eodhd"
    except Exception as exc:
        logger.warning("EODHD failed for %s: %s", ticker, exc)
        return None, "eodhd_error"


def _fetch_polygon(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    if not POLYGON_API_KEY:
        return None, "polygon_no_key"
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if not results:
            return None, "polygon_empty"
        idx = pd.to_datetime([r["t"] for r in results], unit="ms", utc=True).tz_localize(None)
        s = pd.Series([r["c"] for r in results], index=idx)
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = ticker
        return (s if not s.empty else None), "polygon"
    except Exception as exc:
        logger.warning("Polygon failed for %s: %s", ticker, exc)
        return None, "polygon_error"


def _fetch_yahoo_chart(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    try:
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int((pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "div,splits"}
        resp = requests.get(url, params=params, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        result = (((resp.json() or {}).get("chart") or {}).get("result") or [None])[0]
        if not result:
            return None, "yahoo_chart_empty"
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        adjclose_block = (indicators.get("adjclose") or [{}])[0]
        quote_block = (indicators.get("quote") or [{}])[0]
        prices = adjclose_block.get("adjclose") or quote_block.get("close") or []
        if not timestamps or not prices:
            return None, "yahoo_chart_no_series"
        s = pd.Series(prices, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None))
        s = pd.to_numeric(s, errors="coerce").dropna()
        s.name = ticker
        return (s if not s.empty else None), "yahoo_chart_api"
    except Exception as exc:
        logger.warning("Yahoo chart failed for %s: %s", ticker, exc)
        return None, "yahoo_chart_error"


def _series_covers_request(series: pd.Series, start_date: str, end_date: str, min_ratio: float = 0.5) -> bool:
    if series is None or series.empty:
        return False
    expected = max(len(pd.bdate_range(start_date, end_date)), 1)
    return len(series) / expected >= min_ratio


def fetch_ticker_close_series(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    best_series: Optional[pd.Series] = None
    best_source = "all_providers_failed"
    for fetcher in (_fetch_yahoo_chart, _fetch_eodhd, _fetch_polygon):
        series, source = fetcher(ticker, start_date, end_date)
        if series is None or series.empty:
            continue
        if _series_covers_request(series, start_date, end_date):
            return series, source
        if best_series is None or len(series) > len(best_series):
            best_series = series
            best_source = source
    if best_series is not None and not best_series.empty:
        return best_series, best_source
    return None, best_source


def update_local_price_db(
    tickers: Iterable[str],
    end_date: str,
    db_path: Path | str | None = None,
    full_history_start: str = FULL_HISTORY_START,
) -> pd.DataFrame:
    db_path = Path(db_path) if db_path else PRICE_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    universe = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not universe:
        return pd.DataFrame(columns=["ticker", "status", "rows_fetched", "source", "error"])

    conn = _ensure_price_db(db_path)
    cur = conn.cursor()
    cur.execute(
        f"SELECT ticker, first_date, last_date FROM eod_meta WHERE ticker IN ({','.join(['?'] * len(universe))})",
        universe,
    )
    meta_map = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    records = []
    total = len(universe)
    logger.info("Updating EOD DB for %s tickers from %s to %s", total, full_history_start, end_date)
    for idx, ticker in enumerate(universe, start=1):
        started = time.perf_counter()
        first_date, last_date = meta_map.get(ticker, (None, None))
        fetch_ranges: list[tuple[str, str]] = []

        if first_date and pd.to_datetime(first_date) > pd.to_datetime(full_history_start):
            backfill_end = (pd.to_datetime(first_date) - pd.Timedelta(days=1)).date().isoformat()
            if pd.to_datetime(full_history_start) <= pd.to_datetime(backfill_end):
                fetch_ranges.append((full_history_start, backfill_end))

        if last_date:
            forward_start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).date().isoformat()
        else:
            forward_start = full_history_start
        if pd.to_datetime(forward_start) <= pd.to_datetime(end_date):
            fetch_ranges.append((forward_start, end_date))
        elif not fetch_ranges:
            records.append(
                {
                    "ticker": ticker,
                    "status": "up_to_date",
                    "rows_fetched": 0,
                    "source": "db",
                    "error": None,
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            continue

        combined = pd.Series(dtype=float)
        source = "db"
        for range_start, range_end in fetch_ranges:
            logger.info("[%s/%s] %s from %s to %s", idx, total, ticker, range_start, range_end)
            series, range_source = fetch_ticker_close_series(ticker, range_start, range_end)
            if series is None or series.empty:
                continue
            combined = series if combined.empty else combined.combine_first(series)
            source = range_source

        if combined.empty:
            records.append(
                {
                    "ticker": ticker,
                    "status": "failed",
                    "rows_fetched": 0,
                    "source": source,
                    "error": "empty_series",
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            continue

        now = dt.datetime.utcnow().isoformat()
        rows = [(d.strftime("%Y-%m-%d"), ticker, float(v), source, now) for d, v in combined.items()]
        cur.executemany(
            "INSERT OR REPLACE INTO eod_prices (date, ticker, close, source, updated_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        new_first = combined.index.min().strftime("%Y-%m-%d")
        new_last = combined.index.max().strftime("%Y-%m-%d")
        if first_date:
            new_first = min(first_date, new_first)
        if last_date:
            new_last = max(last_date, new_last)
        cur.execute(
            "INSERT OR REPLACE INTO eod_meta (ticker, first_date, last_date, source, updated_at) VALUES (?, ?, ?, ?, ?)",
            (ticker, new_first, new_last, source, now),
        )
        conn.commit()
        records.append(
            {
                "ticker": ticker,
                "status": "updated",
                "rows_fetched": int(len(combined)),
                "source": source,
                "error": None,
                "elapsed_s": time.perf_counter() - started,
            }
        )
    conn.close()
    return pd.DataFrame(records).sort_values(["status", "ticker"])


def load_prices_from_db(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    db_path = Path(db_path) if db_path else PRICE_DB_PATH
    universe = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not universe or not db_path.exists():
        return pd.DataFrame()
    conn = _ensure_price_db(db_path)
    query = (
        "SELECT date, ticker, close FROM eod_prices "
        f"WHERE ticker IN ({','.join(['?'] * len(universe))}) AND date BETWEEN ? AND ?"
    )
    df = pd.read_sql_query(query, conn, params=universe + [start_date, end_date])
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=universe)
    df["date"] = pd.to_datetime(df["date"])
    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    px = px.reindex(columns=universe)
    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    px = px.reindex(idx).ffill()
    px.index.name = "date"
    return px


def fetch_fx_rates(ticker_to_currency: dict[str, str], idx: pd.DatetimeIndex) -> dict[str, pd.Series]:
    fx_rates: dict[str, pd.Series] = {}
    for ccy in set(ticker_to_currency.values()):
        if ccy == "EUR":
            fx_rates[ccy] = pd.Series(1.0, index=idx)
            continue
        fx_ticker = f"{ccy}EUR=X"
        series, _ = fetch_ticker_close_series(fx_ticker, idx.min().date().isoformat(), idx.max().date().isoformat())
        if series is None or series.empty:
            fx_rates[ccy] = pd.Series(np.nan, index=idx).ffill().fillna(1.0)
        else:
            fx_rates[ccy] = series.reindex(idx).ffill()
    return fx_rates
