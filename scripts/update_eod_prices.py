#!/usr/bin/env python
"""Download / backfill EOD prices into SQLite without running full analytics."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degiro_analytics.data_quality import load_sqlite_inventory
from degiro_analytics.degiro_client import extract_traded_isins, fetch_degiro_reports
from degiro_analytics.isin_mapping import build_isin_ticker_map, tickers_for_isins
from degiro_analytics.market_data import update_local_price_db
from degiro_analytics.settings import CACHE_DIR, DEFAULT_BENCHMARKS, PRICE_DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill EOD prices into SQLite")
    parser.add_argument("--from-date", default="2010-01-01", help="First EOD date to store (YYYY-MM-DD)")
    parser.add_argument("--to-date", default=None, help="Last EOD date (default: today)")
    parser.add_argument("--refresh-degiro", action="store_true", help="Re-download DeGiro reports first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    reports = fetch_degiro_reports(cache_dir=CACHE_DIR, use_cache=not args.refresh_degiro)
    traded_universe = extract_traded_isins(reports["account"], reports["positions"], reports.get("transactions"))
    traded_isins = traded_universe["ISIN"].tolist() if not traded_universe.empty else []
    isin_map, mapping_report = build_isin_ticker_map(traded_isins)
    portfolio_tickers = tickers_for_isins(traded_isins, isin_map)
    all_tickers = sorted(set(portfolio_tickers + list(DEFAULT_BENCHMARKS.values())))

    end_date = args.to_date or __import__("datetime").date.today().isoformat()
    print(f"ISINs in universe: {len(traded_isins)}")
    print(f"Mapped tickers: {len(portfolio_tickers)}")
    print(f"Unmapped ISINs: {int((mapping_report['status'] == 'unmapped').sum())}")
    if (mapping_report["status"] == "unmapped").any():
        print(mapping_report[mapping_report["status"] == "unmapped"][["ISIN"]].to_string(index=False))
    print(f"Tickers: {', '.join(all_tickers)}")
    print(f"Downloading EOD prices from {args.from_date} to {end_date} ...")

    report = update_local_price_db(
        tickers=all_tickers,
        end_date=end_date,
        db_path=PRICE_DB_PATH,
        full_history_start=args.from_date,
    )
    inventory = load_sqlite_inventory(PRICE_DB_PATH)

    print("\n=== Update result ===")
    print(report.to_string(index=False))
    if not inventory.empty:
        print("\n=== SQLite inventory ===")
        print(inventory[["ticker", "first_date", "last_date", "row_count", "source"]].to_string(index=False))
    print(f"\nDatabase: {PRICE_DB_PATH}")


if __name__ == "__main__":
    main()
