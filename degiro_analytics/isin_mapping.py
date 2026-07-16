from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from degiro_analytics.settings import ISIN_MAP_PATH, MARKINES_ISIN_MAP_JSON, MARKINES_ISIN_MAP_PICKLE

MANUAL_ISIN_OVERRIDES = {
    "FR0011675362": "N1N.F",
    "AU000000BKY0": "BKY.MC",
    "ES0105079000": "GRE.MC",
    "FR0011742329": "MPHYF",  # McPhy; ALMCP.PA no longer quotes on Yahoo
}


def is_valid_isin(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().upper()
    return bool(re.fullmatch(r"[A-Z0-9]{12}", text))


def _load_mapping_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        if path.suffix.lower() in {".p", ".pickle", ".pkl"}:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            return {}
        if isinstance(data, dict):
            return {str(k).strip().upper(): str(v).strip() for k, v in data.items() if k and v}
    except Exception:
        return {}
    return {}


def load_base_isin_ticker_map() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    merged.update(MANUAL_ISIN_OVERRIDES)
    for path in (MARKINES_ISIN_MAP_PICKLE, MARKINES_ISIN_MAP_JSON, ISIN_MAP_PATH):
        merged.update(_load_mapping_file(Path(path)))
    return merged


def get_ticker_from_isin_yahoo(isin: str, retries: int = 3) -> Optional[str]:
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={isin}"
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            response.raise_for_status()
            quotes = response.json().get("quotes", [])
            if quotes:
                symbol = quotes[0].get("symbol")
                if symbol:
                    return str(symbol)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


def build_isin_ticker_map(
    isins: Iterable[str],
    map_path: Path | str | None = None,
    lookup_missing: bool = True,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    requested = sorted({str(x).strip().upper() for x in isins if is_valid_isin(x)})
    cache = load_base_isin_ticker_map()
    rows: List[dict] = []

    for isin in requested:
        if isin in cache:
            rows.append({"ISIN": isin, "ticker": cache[isin], "status": "mapped", "source": "cache"})
            continue
        if not lookup_missing:
            rows.append({"ISIN": isin, "ticker": None, "status": "unmapped", "source": None})
            continue
        ticker = get_ticker_from_isin_yahoo(isin)
        if ticker:
            cache[isin] = ticker
            rows.append({"ISIN": isin, "ticker": ticker, "status": "mapped", "source": "yahoo_search"})
        else:
            rows.append({"ISIN": isin, "ticker": None, "status": "unmapped", "source": None})

    out_path = Path(map_path) if map_path else ISIN_MAP_PATH
    if cache:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            pickle.dump(cache, fh)

    report = pd.DataFrame(rows).sort_values(["status", "ISIN"])
    resolved = {row["ISIN"]: row["ticker"] for _, row in report.iterrows() if row["status"] == "mapped"}
    return resolved, report


def tickers_for_isins(isins: Iterable[str], isin_map: Dict[str, str]) -> List[str]:
    tickers = []
    for isin in isins:
        if not is_valid_isin(isin):
            continue
        ticker = isin_map.get(str(isin).strip().upper())
        if ticker:
            tickers.append(ticker)
    return sorted(set(tickers))
