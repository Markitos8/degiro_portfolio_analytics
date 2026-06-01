from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from degiro_analytics.charts import generate_all_charts
from degiro_analytics.data_quality import build_full_data_quality_report, write_data_quality_report
from degiro_analytics.degiro_client import extract_traded_isins, fetch_degiro_reports
from degiro_analytics.isin_mapping import build_isin_ticker_map, tickers_for_isins
from degiro_analytics.market_data import load_prices_from_db, update_local_price_db
from degiro_analytics.metrics import (
    build_benchmark_comparison,
    compute_asset_metrics,
    compute_performance_metrics,
    compute_rolling_metrics,
    fetch_fama_french_factors,
    run_factor_regression,
)
from degiro_analytics.optimization import compute_efficient_frontier
from degiro_analytics.portfolio_core import (
    build_cash_series,
    build_data_quality_report,
    build_dividends_table,
    build_portfolio_equity_curve,
    build_positions_from_trades,
    build_trades_table,
    compute_asset_contribution,
    reconcile_positions_with_degiro,
    summarize_dividends,
)
from degiro_analytics.regime import detect_markov_regimes
from degiro_analytics.settings import (
    CACHE_DIR,
    CHART_DIR,
    DEFAULT_BENCHMARKS,
    DEGIRO_EXCEL_DIR,
    EXPORT_DIR,
    FULL_HISTORY_START,
    PRICE_DB_PATH,
)

logger = logging.getLogger(__name__)


def _export_results(results: Dict[str, object], export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, obj) -> None:
        if isinstance(obj, pd.Series):
            obj.to_frame(name).to_csv(export_dir / f"{name}.csv")
        elif isinstance(obj, pd.DataFrame):
            obj.to_csv(export_dir / f"{name}.csv")

    flat_keys = [
        "portfolio",
        "equity_by_ticker",
        "portfolio_metrics",
        "asset_metrics",
        "dividend_summary",
        "current_weights",
        "rolling_metrics",
        "factor_portfolio",
        "db_update",
        "traded_universe",
        "isin_mapping_report",
        "position_reconciliation",
    ]
    for key in flat_keys:
        val = results.get(key)
        if isinstance(val, (pd.DataFrame, pd.Series)):
            _write(key, val)

    nested = {
        "contribution": results.get("contribution", {}),
        "benchmark": results.get("benchmark", {}),
        "optimization": results.get("optimization", {}),
        "regimes": results.get("regimes", {}),
        "data_quality": results.get("data_quality", {}),
    }
    for prefix, block in nested.items():
        if not isinstance(block, dict):
            continue
        for sub_key, sub_val in block.items():
            if isinstance(sub_val, (pd.DataFrame, pd.Series)):
                _write(f"{prefix}_{sub_key}", sub_val)


def run_portfolio_analysis(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config_path: Optional[str] = None,
    cache_dir: Path | str | None = None,
    benchmark_tickers: Optional[Dict[str, str]] = None,
    export_dir: Path | str | None = None,
    chart_dir: Path | str | None = None,
    price_db_path: Path | str | None = None,
    ff_model: str = "5f",
    refresh_degiro: bool = False,
    refresh_prices: bool = True,
    price_history_start: Optional[str] = None,
    generate_charts: bool = True,
) -> Dict[str, object]:
    step_times: Dict[str, float] = {}
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    export_dir = Path(export_dir) if export_dir else EXPORT_DIR
    chart_dir = Path(chart_dir) if chart_dir else CHART_DIR
    price_db_path = Path(price_db_path) if price_db_path else PRICE_DB_PATH

    t0 = time.perf_counter()
    reports = fetch_degiro_reports(
        from_date=pd.to_datetime(start_date).date() if start_date else None,
        to_date=pd.to_datetime(end_date).date() if end_date else None,
        config_path=config_path,
        cache_dir=cache_dir,
        use_cache=not refresh_degiro,
    )
    step_times["fetch_degiro"] = time.perf_counter() - t0

    account = reports["account"]
    traded_universe = extract_traded_isins(account, reports["positions"], reports.get("transactions"))
    traded_isins = traded_universe["ISIN"].tolist() if not traded_universe.empty else []
    isin_map, isin_mapping_report = build_isin_ticker_map(traded_isins)
    trades = build_trades_table(account, isin_map)
    positions = build_positions_from_trades(trades, end_date=pd.to_datetime(end_date) if end_date else None)
    dividends = build_dividends_table(account, isin_map)
    dividend_summary = summarize_dividends(dividends)
    position_reconciliation = reconcile_positions_with_degiro(positions, reports["positions"], isin_map)

    if positions.empty and traded_universe.empty:
        raise ValueError("No positions were built from DeGiro trades.")

    start = start_date or (positions.index.min().date().isoformat() if not positions.empty else "2010-01-01")
    end = end_date or (positions.index.max().date().isoformat() if not positions.empty else pd.Timestamp.today().date().isoformat())
    benchmark_tickers = benchmark_tickers or DEFAULT_BENCHMARKS
    portfolio_tickers = tickers_for_isins(traded_isins, isin_map)
    if not portfolio_tickers and not positions.empty:
        portfolio_tickers = positions.columns.tolist()
    all_price_tickers = sorted(set(portfolio_tickers + list(benchmark_tickers.values())))

    t1 = time.perf_counter()
    if refresh_prices:
        db_update = update_local_price_db(
            tickers=all_price_tickers,
            end_date=end,
            db_path=price_db_path,
            full_history_start=price_history_start or FULL_HISTORY_START,
        )
    else:
        db_update = pd.DataFrame({"status": ["skipped_refresh"]})
    step_times["update_prices"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    all_prices = load_prices_from_db(all_price_tickers, start, end, db_path=price_db_path)
    prices = all_prices.reindex(columns=portfolio_tickers).reindex(positions.index).ffill() if not positions.empty else all_prices.reindex(columns=portfolio_tickers)
    benchmark_prices = all_prices.reindex(columns=list(benchmark_tickers.values()))
    step_times["load_prices"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    cash = build_cash_series(account).reindex(positions.index).ffill()
    portfolio, equity_by_ticker = build_portfolio_equity_curve(positions, prices, cash_series=cash)
    step_times["build_equity"] = time.perf_counter() - t3

    portfolio_metrics = compute_performance_metrics(portfolio["Returns"])
    contribution = compute_asset_contribution(equity_by_ticker, portfolio["Total"])
    asset_metrics = compute_asset_metrics(equity_by_ticker)
    rolling_metrics = compute_rolling_metrics(portfolio["Returns"])
    current_weights = contribution["weights"].iloc[-1].sort_values(ascending=False)

    t4 = time.perf_counter()
    benchmark = build_benchmark_comparison(
        portfolio_returns=portfolio["Returns"],
        benchmark_tickers=benchmark_tickers,
        benchmark_prices=benchmark_prices,
    )
    step_times["benchmark"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    asset_returns = equity_by_ticker.pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    optimization = compute_efficient_frontier(asset_returns)
    step_times["optimization"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    regimes = detect_markov_regimes(portfolio["Returns"], n_regimes=2)
    step_times["regimes"] = time.perf_counter() - t6

    t7 = time.perf_counter()
    try:
        ff = fetch_fama_french_factors(model=ff_model)
        factor_portfolio = run_factor_regression(portfolio["Returns"], ff, model=ff_model)
    except Exception as exc:
        logger.warning("Fama-French unavailable: %s", exc)
        ff = pd.DataFrame()
        factor_portfolio = pd.Series(dtype=float)
    step_times["factors"] = time.perf_counter() - t7

    asset_quality = build_data_quality_report(
        prices,
        positions.index if not positions.empty else pd.bdate_range(start, end),
    )
    benchmark_quality = build_data_quality_report(
        benchmark_prices.rename(columns={v: k for k, v in benchmark_tickers.items()}),
        positions.index if not positions.empty else pd.bdate_range(start, end),
    )
    data_quality = build_full_data_quality_report(
        account=account,
        isin_map=isin_map,
        trades=trades,
        positions=positions if not positions.empty else pd.DataFrame(index=pd.DatetimeIndex([])),
        expected_tickers=all_price_tickers,
        analysis_start=start,
        analysis_end=end,
        db_path=price_db_path,
        db_update=db_update,
        asset_price_quality=asset_quality["summary"],
        asset_price_by_ticker=asset_quality["by_ticker"],
        benchmark_price_quality=benchmark_quality["by_ticker"],
        benchmark_tickers=benchmark_tickers,
        isin_mapping_report=isin_mapping_report,
        traded_universe=traded_universe,
        position_reconciliation=position_reconciliation,
    )

    output: Dict[str, object] = {
        "reports": reports,
        "traded_universe": traded_universe,
        "isin_ticker_map": isin_map,
        "isin_mapping_report": isin_mapping_report,
        "position_reconciliation": position_reconciliation,
        "degiro_excel_dir": str(DEGIRO_EXCEL_DIR),
        "trades": trades,
        "positions": positions,
        "prices": prices,
        "portfolio": portfolio,
        "equity_by_ticker": equity_by_ticker,
        "portfolio_metrics": portfolio_metrics,
        "asset_metrics": asset_metrics,
        "contribution": contribution,
        "dividends": dividends,
        "dividend_summary": dividend_summary,
        "current_weights": current_weights,
        "rolling_metrics": rolling_metrics,
        "benchmark": benchmark,
        "optimization": optimization,
        "regimes": regimes,
        "fama_french_factors": ff,
        "factor_portfolio": factor_portfolio,
        "data_quality": data_quality,
        "run_timing_seconds": pd.Series(step_times),
        "db_update": db_update,
    }

    t8 = time.perf_counter()
    _export_results(output, export_dir)
    report_path = write_data_quality_report(data_quality, export_dir)
    if generate_charts:
        generate_all_charts(output, chart_dir)
    step_times["export"] = time.perf_counter() - t8
    output["run_timing_seconds"] = pd.Series(step_times)
    output["data_quality_report_path"] = str(report_path)

    with open(export_dir / "portfolio_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(portfolio_metrics.to_dict(), fh, indent=2, default=str)

    logger.info("Analysis complete. Charts: %s | Exports: %s", chart_dir, export_dir)
    return output
