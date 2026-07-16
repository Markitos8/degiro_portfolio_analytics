from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from degiro_analytics.settings import EODHD_API_KEY, FULL_HISTORY_START, POLYGON_API_KEY, PRICE_DB_PATH

logger = logging.getLogger(__name__)

# Yahoo suffix -> EODHD exchange code
_YAHOO_SUFFIX_TO_EODHD = {
    "AS": "AS",
    "PA": "PA",
    "MC": "MC",
    "BR": "BR",
    "MI": "MI",
    "SW": "SW",
    "L": "LSE",
    "DE": "XETRA",
    "F": "F",
    "HK": "HK",
    "T": "TSE",
    "TO": "TO",
    "AX": "AU",
    "NS": "NSE",
    "BO": "BSE",
    "OL": "OL",
    "ST": "ST",
    "HE": "HE",
    "CO": "CO",
    "VI": "VI",
    "LS": "LS",
    "WA": "WA",
}

# Yahoo index tickers -> EODHD INDEX symbols
_YAHOO_INDEX_TO_EODHD = {
    "^GSPC": "GSPC.INDX",
    "^DJI": "DJI.INDX",
    "^IXIC": "IXIC.INDX",
    "^RUT": "RUT.INDX",
    "^STOXX50E": "STOXX50E.INDX",
    "^GDAXI": "GDAXI.INDX",
    "^N225": "N225.INDX",
    "^IBEX": "IBEX.INDX",
    "^FTSE": "FTSE.INDX",
    "^SP500TR": "SP500TR.INDX",
    "^VIX": "VIX.INDX",
    "^FCHI": "FCHI.INDX",
    "^AEX": "AEX.INDX",
}

# Yahoo index tickers -> Polygon index tickers (US-focused coverage)
_YAHOO_INDEX_TO_POLYGON = {
    "^GSPC": "I:SPX",
    "^DJI": "I:DJI",
    "^IXIC": "I:COMP",
    "^RUT": "I:RUT",
    "^VIX": "I:VIX",
}

_YAHOO_CHART_HOSTS = (
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
)


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
    """Map Yahoo-style tickers to EODHD symbols (equities, indices, FX, crypto)."""
    t = str(ticker).strip()
    if not t:
        return None

    if t in _YAHOO_INDEX_TO_EODHD:
        return _YAHOO_INDEX_TO_EODHD[t]

    if t.startswith("^"):
        # Generic index fallback: ^FOO -> FOO.INDX
        return f"{t[1:]}.INDX"

    if t.endswith("=X"):
        pair = t[:-2].upper()
        if len(pair) == 6:
            return f"{pair}.FOREX"
        return None

    if "-" in t and t.upper().endswith(("-USD", "-EUR", "-GBP")):
        return f"{t.upper()}.CC"

    if "." in t:
        root, suffix = t.rsplit(".", 1)
        exch = _YAHOO_SUFFIX_TO_EODHD.get(suffix.upper(), suffix.upper())
        return f"{root}.{exch}"

    return f"{t}.US"


def _yahoo_to_polygon_ticker(ticker: str) -> Optional[str]:
    """Map Yahoo-style tickers to Polygon tickers. Skip unsupported international symbols."""
    t = str(ticker).strip()
    if not t:
        return None

    if t in _YAHOO_INDEX_TO_POLYGON:
        return _YAHOO_INDEX_TO_POLYGON[t]

    if t.startswith("^"):
        # Unknown international indices are not reliably available on Polygon stocks API
        return None

    if t.endswith("=X"):
        pair = t[:-2].upper()
        if len(pair) == 6:
            return f"C:{pair}"
        return None

    if "." in t:
        # Polygon free/stocks aggregates are US-centric; skip Yahoo exchange suffixes
        return None

    if "-" in t:
        return None

    return t.upper()


def _series_from_closes(ticker: str, dates, closes) -> Optional[pd.Series]:
    # Use numpy values so a Series `closes` is not reindexed against the new DatetimeIndex.
    values = pd.to_numeric(pd.Series(closes).to_numpy(), errors="coerce")
    idx = pd.to_datetime(pd.Series(dates).to_numpy())
    s = pd.Series(values, index=idx, name=ticker).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s if not s.empty else None


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
        if resp.status_code in {401, 403}:
            return None, f"eodhd_http_{resp.status_code}"
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("error"):
            return None, f"eodhd_api_error:{payload.get('error')}"
        if not isinstance(payload, list) or not payload:
            return None, "eodhd_empty"
        df = pd.DataFrame(payload)
        close_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
        if "date" not in df.columns or close_col not in df.columns:
            return None, "eodhd_bad_payload"
        s = _series_from_closes(ticker, df["date"], df[close_col])
        if s is None:
            return None, "eodhd_empty"
        return s, "eodhd"
    except Exception as exc:
        logger.warning("EODHD failed for %s (%s): %s", ticker, symbol, exc)
        return None, "eodhd_error"


def _fetch_polygon(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    if not POLYGON_API_KEY:
        return None, "polygon_no_key"
    poly_ticker = _yahoo_to_polygon_ticker(ticker)
    if poly_ticker is None:
        return None, "polygon_skip_symbol"
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/1/day/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code in {401, 403}:
            return None, f"polygon_http_{resp.status_code}"
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") == "ERROR":
            return None, f"polygon_api_error:{body.get('error')}"
        results = body.get("results") or []
        if not results:
            return None, "polygon_empty"
        idx = pd.to_datetime([r["t"] for r in results], unit="ms", utc=True).tz_localize(None)
        s = _series_from_closes(ticker, idx, [r["c"] for r in results])
        if s is None:
            return None, "polygon_empty"
        return s, "polygon"
    except Exception as exc:
        logger.warning("Polygon failed for %s (%s): %s", ticker, poly_ticker, exc)
        return None, "polygon_error"


def _fetch_yahoo_chart(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int((pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp())
    params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "div,splits"}
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = "yahoo_chart_error"

    for host in _YAHOO_CHART_HOSTS:
        url = f"{host}/v8/finance/chart/{ticker}"
        try:
            resp = requests.get(url, params=params, timeout=12, headers=headers)
            resp.raise_for_status()
            result = (((resp.json() or {}).get("chart") or {}).get("result") or [None])[0]
            if not result:
                last_error = "yahoo_chart_empty"
                continue
            timestamps = result.get("timestamp") or []
            indicators = result.get("indicators") or {}
            adjclose_block = (indicators.get("adjclose") or [{}])[0]
            quote_block = (indicators.get("quote") or [{}])[0]
            prices = adjclose_block.get("adjclose") or quote_block.get("close") or []
            if not timestamps or not prices:
                last_error = "yahoo_chart_no_series"
                continue
            idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None)
            s = _series_from_closes(ticker, idx, prices)
            if s is None:
                last_error = "yahoo_chart_no_series"
                continue
            return s, "yahoo_chart_api"
        except Exception as exc:
            last_error = "yahoo_chart_error"
            logger.warning("Yahoo chart failed for %s via %s: %s", ticker, host, exc)
    return None, last_error


def _fetch_yfinance(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance_missing"

    try:
        # yfinance end is exclusive
        end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(
            ticker,
            start=start_date,
            end=end_exclusive,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            return None, "yfinance_empty"
        if isinstance(df.columns, pd.MultiIndex):
            closes = df["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
        else:
            closes = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        s = _series_from_closes(ticker, closes.index, closes.values)
        return (s, "yfinance") if s is not None else (None, "yfinance_empty")
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        return None, "yfinance_error"


def _series_covers_request(series: pd.Series, start_date: str, end_date: str, min_ratio: float = 0.5) -> bool:
    if series is None or series.empty:
        return False
    expected = max(len(pd.bdate_range(start_date, end_date)), 1)
    return len(series) / expected >= min_ratio


ProviderFetcher = Callable[[str, str, str], Tuple[Optional[pd.Series], str]]

# Preference order matches project docs: paid vendors first, free Yahoo fallbacks last.
_PROVIDER_FETCHERS: tuple[tuple[str, ProviderFetcher], ...] = (
    ("eodhd", _fetch_eodhd),
    ("polygon", _fetch_polygon),
    ("yahoo_chart", _fetch_yahoo_chart),
    ("yfinance", _fetch_yfinance),
)


def fetch_ticker_close_series(ticker: str, start_date: str, end_date: str) -> Tuple[Optional[pd.Series], str]:
    best_series: Optional[pd.Series] = None
    best_source = "all_providers_failed"
    for _name, fetcher in _PROVIDER_FETCHERS:
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


def probe_eod_providers(
    tickers: Optional[Iterable[str]] = None,
    start_date: str = "2024-01-02",
    end_date: str = "2024-03-28",
) -> pd.DataFrame:
    """Hit every configured EOD provider for a small ticker set and report status."""
    sample = list(tickers) if tickers is not None else ["AAPL", "IWDA.AS", "^GSPC", "EURUSD=X"]
    rows = []
    for ticker in sample:
        for name, fetcher in _PROVIDER_FETCHERS:
            t0 = time.perf_counter()
            series, source = fetcher(ticker, start_date, end_date)
            elapsed = time.perf_counter() - t0
            ok = series is not None and not series.empty
            rows.append(
                {
                    "provider": name,
                    "ticker": ticker,
                    "ok": ok,
                    "status": source if not ok else "ok",
                    "rows": int(len(series)) if ok else 0,
                    "first_date": series.index.min().strftime("%Y-%m-%d") if ok else None,
                    "last_date": series.index.max().strftime("%Y-%m-%d") if ok else None,
                    "last_close": float(series.iloc[-1]) if ok else None,
                    "mapped_symbol": (
                        _yahoo_to_eodhd_symbol(ticker)
                        if name == "eodhd"
                        else _yahoo_to_polygon_ticker(ticker)
                        if name == "polygon"
                        else ticker
                    ),
                    "elapsed_s": round(elapsed, 3),
                    "has_api_key": (
                        bool(EODHD_API_KEY)
                        if name == "eodhd"
                        else bool(POLYGON_API_KEY)
                        if name == "polygon"
                        else True
                    ),
                }
            )
    return pd.DataFrame(rows)


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
            # Keep existing DB coverage when incremental/backfill windows are empty
            # (e.g. exchange holiday on the single missing day).
            if first_date and last_date:
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
            else:
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
