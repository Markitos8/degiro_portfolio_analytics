#!/usr/bin/env python
"""Probe every EOD provider and optional public market-data sources."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EOD data source connectivity")
    parser.add_argument("--from-date", default="2024-01-02")
    parser.add_argument("--to-date", default="2024-03-28")
    parser.add_argument(
        "--write-db",
        action="store_true",
        help="Also download benchmark EOD into the local SQLite price DB",
    )
    parser.add_argument(
        "--allow-missing-keys",
        action="store_true",
        help="Do not fail when EODHD/Polygon API keys are absent",
    )
    parser.add_argument(
        "--use-eodhd-demo",
        action="store_true",
        help="Use EODHD demo token for US-symbol smoke tests if no key is set",
    )
    args = parser.parse_args()

    if args.use_eodhd_demo and not os.getenv("EODHD_API_KEY", "").strip():
        os.environ["EODHD_API_KEY"] = "demo"
        print("Using EODHD demo API token for smoke test (US symbols / limited coverage).")

    # Import after optional demo key injection so settings pick it up.
    import pandas as pd
    import requests

    import degiro_analytics.market_data as market_data
    import degiro_analytics.settings as settings
    from degiro_analytics.market_data import (
        _yahoo_to_eodhd_symbol,
        _yahoo_to_polygon_ticker,
        probe_eod_providers,
        update_local_price_db,
    )
    from degiro_analytics.metrics import fetch_fama_french_factors
    from degiro_analytics.settings import DEFAULT_BENCHMARKS, PRICE_DB_PATH

    eodhd_key = (settings.EODHD_API_KEY or os.getenv("EODHD_API_KEY", "")).strip()
    polygon_key = (settings.POLYGON_API_KEY or os.getenv("POLYGON_API_KEY", "")).strip()
    # Keep market_data module in sync if settings were loaded before env mutation.
    market_data.EODHD_API_KEY = eodhd_key
    market_data.POLYGON_API_KEY = polygon_key
    settings.EODHD_API_KEY = eodhd_key
    settings.POLYGON_API_KEY = polygon_key

    print("API key status:")
    print(f"  EODHD_API_KEY: {'set (' + eodhd_key + ')' if eodhd_key else 'MISSING'}")
    print(f"  POLYGON_API_KEY: {'set' if polygon_key else 'MISSING'}")
    print()

    tickers = ["AAPL", "IWDA.AS", "^GSPC", "EURUSD=X", "VWCE.DE"]
    print("Symbol mapping samples:")
    for t in tickers:
        print(f"  {t:12} eodhd={_yahoo_to_eodhd_symbol(t)}  polygon={_yahoo_to_polygon_ticker(t)}")
    print()

    probe = probe_eod_providers(tickers=tickers, start_date=args.from_date, end_date=args.to_date)
    print("=== Provider probe ===")
    print(probe.to_string(index=False))
    print()

    # MSCI
    msci_status = {"provider": "msci", "ok": False, "status": "msci_error", "rows": 0}
    try:
        url = (
            "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph"
            "?currency_symbol=USD&index_variant=STRD&start_date=20240101&end_date=20240329"
            "&data_frequency=DAILY&index_codes=990100"
        )
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        levels = (((resp.json() or {}).get("indexes") or {}).get("INDEX_LEVELS") or [])
        msci_status = {
            "provider": "msci",
            "ok": len(levels) > 0,
            "status": "ok" if levels else "msci_empty",
            "rows": len(levels),
        }
    except Exception as exc:
        msci_status["status"] = f"msci_error:{exc}"

    # Fama-French
    ff_status = {"provider": "famafrench", "ok": False, "status": "famafrench_error", "rows": 0}
    try:
        ff = fetch_fama_french_factors("3f")
        ff_status = {
            "provider": "famafrench",
            "ok": ff is not None and not ff.empty,
            "status": "ok" if ff is not None and not ff.empty else "famafrench_empty",
            "rows": int(len(ff)) if ff is not None and not ff.empty else 0,
        }
    except Exception as exc:
        ff_status["status"] = f"famafrench_error:{exc}"

    extras = pd.DataFrame([msci_status, ff_status])
    print("=== Public benchmark / factor sources ===")
    print(extras[["provider", "ok", "status", "rows"]].to_string(index=False))
    print()

    free_ok = probe[probe["provider"].isin(["yahoo_chart", "yfinance"])].groupby("ticker")["ok"].any()
    free_failures = free_ok[~free_ok].index.tolist()

    paid_failures: list[str] = []
    if eodhd_key:
        eodhd_aapl = probe[(probe["provider"] == "eodhd") & (probe["ticker"] == "AAPL")]
        if eodhd_aapl.empty or not bool(eodhd_aapl.iloc[0]["ok"]):
            paid_failures.append("eodhd:AAPL")
        if eodhd_key != "demo":
            for _, row in probe[probe["provider"] == "eodhd"].iterrows():
                if not row["ok"] and row["status"] != "eodhd_skip_symbol":
                    paid_failures.append(f"eodhd:{row['ticker']}")

    if polygon_key:
        for _, row in probe[probe["provider"] == "polygon"].iterrows():
            if row["status"] == "polygon_skip_symbol":
                continue
            if not row["ok"]:
                paid_failures.append(f"polygon:{row['ticker']}")

    db_failures: list[str] = []
    if args.write_db:
        bench = list(DEFAULT_BENCHMARKS.values()) + ["AAPL"]
        report = update_local_price_db(
            tickers=bench,
            end_date=args.to_date,
            db_path=PRICE_DB_PATH,
            full_history_start=args.from_date,
        )
        print("=== SQLite write smoke test ===")
        print(report.to_string(index=False))
        print(f"DB: {PRICE_DB_PATH}")
        print()
        db_failures = report.loc[report["status"] == "failed", "ticker"].tolist()

    missing_keys = []
    if not eodhd_key:
        missing_keys.append("EODHD_API_KEY")
    if not polygon_key:
        missing_keys.append("POLYGON_API_KEY")

    print("=== Summary ===")
    print(f"Free Yahoo coverage for sample tickers: {'PASS' if not free_failures else 'FAIL'}")
    if free_failures:
        print(f"  Missing: {', '.join(free_failures)}")
    print(f"MSCI EOD: {'PASS' if msci_status['ok'] else 'FAIL'} ({msci_status['status']})")
    print(f"Fama-French: {'PASS' if ff_status['ok'] else 'FAIL'} ({ff_status['status']})")
    if paid_failures:
        print(f"Paid provider failures: {', '.join(sorted(set(paid_failures)))}")
    if missing_keys:
        print(f"Missing API keys: {', '.join(missing_keys)}")
    if db_failures:
        print(f"SQLite write failures: {', '.join(db_failures)}")

    fatal = bool(free_failures) or (not msci_status["ok"]) or (not ff_status["ok"]) or bool(paid_failures) or bool(db_failures)
    if missing_keys and not args.allow_missing_keys:
        # Demo mode satisfies EODHD key presence; Polygon still required unless allowed.
        if not (args.use_eodhd_demo and missing_keys == ["POLYGON_API_KEY"] and args.allow_missing_keys):
            if missing_keys:
                fatal = True

    if missing_keys and args.allow_missing_keys and not paid_failures:
        # Keys absent is acceptable when explicitly allowed.
        pass

    # Clarify: with --allow-missing-keys, missing keys alone are non-fatal.
    if args.allow_missing_keys:
        fatal = bool(free_failures) or (not msci_status["ok"]) or (not ff_status["ok"]) or bool(paid_failures) or bool(db_failures)

    print("RESULT:", "PASS" if not fatal else "FAIL")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
