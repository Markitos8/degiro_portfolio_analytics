from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

for env_path in (ROOT / ".env", ROOT.parent / "EOM" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=False)

CONFIG_PATH = ROOT / "config" / "config.json"
CACHE_DIR = ROOT / "data" / "cache" / "degiro_connector_data"
PRICE_DB_PATH = ROOT / "data" / "sqlite" / "eod_prices.sqlite"
EXPORT_DIR = ROOT / "data" / "exports"
CHART_DIR = ROOT / "output" / "charts"
ISIN_MAP_PATH = ROOT / "data" / "cache" / "isin_ticker_map.pickle"
MARKINES_ISIN_MAP_PICKLE = ROOT.parent / "mark_ines" / "medium" / "isin_ticker_map.p"
MARKINES_ISIN_MAP_JSON = ROOT.parent / "mark_ines" / "medium" / "degiro_connector_data" / "isin_ticker_map.json"
DEGIRO_EXCEL_DIR = CACHE_DIR / "excel"

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "").strip()

DEFAULT_BENCHMARKS = {
    "S&P500": "^GSPC",
    "EuroStoxx50": "^STOXX50E",
    "MSCI World": "IWDA.AS",
    "IBEX35": "^IBEX",
}

PERIODS_PER_YEAR = 252
FULL_HISTORY_START = "1990-01-01"
