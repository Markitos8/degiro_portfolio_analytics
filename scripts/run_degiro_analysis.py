#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degiro_analytics.pipeline import run_portfolio_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="DeGiro portfolio analytics pipeline")
    parser.add_argument("--start-date", default=None, help="Analysis start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Analysis end date YYYY-MM-DD")
    parser.add_argument("--config-path", default=None, help="Path to DeGiro config.json")
    parser.add_argument("--export-dir", default=str(ROOT / "data" / "exports"))
    parser.add_argument("--chart-dir", default=str(ROOT / "output" / "charts"))
    parser.add_argument("--refresh-degiro", action="store_true", help="Re-download DeGiro reports from API")
    parser.add_argument("--refresh-prices", action="store_true", help="Refresh EOD price database")
    parser.add_argument(
        "--price-start-date",
        default="2010-01-01",
        help="Earliest EOD date to store in SQLite (default: 2010-01-01). Used with --refresh-prices.",
    )
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--ff-model", default="5f", choices=["3f", "5f"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    results = run_portfolio_analysis(
        start_date=args.start_date,
        end_date=args.end_date,
        config_path=args.config_path,
        export_dir=args.export_dir,
        chart_dir=args.chart_dir,
        refresh_degiro=args.refresh_degiro,
        refresh_prices=args.refresh_prices,
        price_history_start=args.price_start_date,
        generate_charts=not args.no_charts,
        ff_model=args.ff_model,
    )

    metrics = results["portfolio_metrics"]
    print("\n=== Portfolio Metrics ===")
    for k, v in metrics.items():
        if k in {"Sharpe", "Sortino", "Calmar", "Win Rate"}:
            print(f"{k:16s}: {v: .2f}")
        elif k == "Volatility":
            print(f"{k:16s}: {v: .2f}")
        elif k in {"Total Return", "CAGR", "Max Drawdown"}:
            print(f"{k:16s}: {v * 100:,.2f}%")
        else:
            print(f"{k:16s}: {v}")

    dividend_summary = results.get("dividend_summary")
    if isinstance(dividend_summary, pd.DataFrame) and not dividend_summary.empty:
        div_total = dividend_summary["gross_eur"].sum()
        print(f"\nTotal dividends (EUR): {div_total:,.2f}")

    data_quality = results.get("data_quality", {})
    if isinstance(data_quality, dict):
        report_path = results.get("data_quality_report_path")
        if report_path:
            print(f"\nData quality report: {report_path}")
        summary = data_quality.get("combined_summary")
        if isinstance(summary, pd.Series):
            print("\n=== Data Quality Summary ===")
            for k, v in summary.items():
                if isinstance(v, float) and "coverage" in k.lower():
                    print(f"{k}: {v:.1%}")
                else:
                    print(f"{k}: {v}")
        issues = data_quality.get("issues")
        if isinstance(issues, pd.DataFrame) and not issues.empty:
            print(f"\nTickers with issues: {', '.join(issues['ticker'].astype(str).tolist())}")

    reconciliation = results.get("position_reconciliation")
    if isinstance(reconciliation, pd.DataFrame) and not reconciliation.empty:
        mismatches = reconciliation[reconciliation["status"] != "match"]
        print("\n=== Position Reconciliation ===")
        print(f"Matches: {int((reconciliation['status'] == 'match').sum())} / {len(reconciliation)}")
        if not mismatches.empty:
            print(mismatches[["ISIN", "ticker", "degiro_quantity", "calculated_quantity", "difference", "status"]].head(10).to_string(index=False))

    unmapped = results.get("isin_mapping_report")
    if isinstance(unmapped, pd.DataFrame) and not unmapped.empty:
        bad = unmapped[unmapped["status"] == "unmapped"]
        if not bad.empty:
            print(f"\nUnmapped ISINs: {', '.join(bad['ISIN'].tolist())}")

    print(f"\nDeGiro Excel reports: {results.get('degiro_excel_dir', '')}")
    print(f"Exports: {args.export_dir}")
    print(f"Charts:  {args.chart_dir}")


if __name__ == "__main__":
    main()
