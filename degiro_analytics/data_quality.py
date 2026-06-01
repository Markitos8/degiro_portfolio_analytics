from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from degiro_analytics.market_data import _ensure_price_db


def load_sqlite_inventory(db_path: Path | str) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame(
            columns=[
                "ticker",
                "first_date",
                "last_date",
                "source",
                "updated_at",
                "row_count",
                "in_db",
            ]
        )

    conn = _ensure_price_db(db_path)
    meta = pd.read_sql_query("SELECT * FROM eod_meta ORDER BY ticker", conn)
    counts = pd.read_sql_query(
        "SELECT ticker, COUNT(*) AS row_count FROM eod_prices GROUP BY ticker",
        conn,
    )
    conn.close()

    if meta.empty:
        return pd.DataFrame(columns=["ticker", "first_date", "last_date", "source", "updated_at", "row_count", "in_db"])

    inventory = meta.merge(counts, on="ticker", how="left")
    inventory["row_count"] = inventory["row_count"].fillna(0).astype(int)
    inventory["in_db"] = True
    return inventory


def load_raw_prices_from_db(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    db_path: Path | str,
) -> pd.DataFrame:
    db_path = Path(db_path)
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
    return df.pivot(index="date", columns="ticker", values="close").sort_index().reindex(columns=universe)


def build_isin_mapping_report(
    account: pd.DataFrame,
    isin_map: Dict[str, str],
    trades: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    trade_mask = account["Description"].astype(str).str.contains("Compra|Venta", na=False)
    trade_rows = account.loc[trade_mask].copy()
    trade_rows["ISIN"] = trade_rows["ISIN"].astype(str).str.strip()
    trade_rows = trade_rows[trade_rows["ISIN"].notna() & (trade_rows["ISIN"] != "") & (trade_rows["ISIN"] != "nan")]

    isin_reference = (
        trade_rows.groupby("ISIN")
        .agg(
            product=("Producto", "first"),
            trade_count=("Description", "size"),
            first_trade=("Fecha", "min"),
            last_trade=("Fecha", "max"),
        )
        .reset_index()
    )
    isin_reference["mapped_ticker"] = isin_reference["ISIN"].map(isin_map)
    isin_reference["mapping_status"] = np.where(
        isin_reference["mapped_ticker"].notna(),
        "mapped",
        "missing_ticker",
    )

    unmapped_isins = isin_reference[isin_reference["mapping_status"] == "missing_ticker"].copy()
    mapped_isins = isin_reference[isin_reference["mapping_status"] == "mapped"].copy()

    excluded_trades = trade_rows[~trade_rows["ISIN"].isin(mapped_isins["ISIN"])].copy()
    if not excluded_trades.empty:
        excluded_trades = excluded_trades[
            ["Datetime", "Fecha", "Producto", "ISIN", "Description", "Value", "Curncy"]
        ].sort_values("Datetime", ascending=False)

    duplicate_tickers = (
        mapped_isins.groupby("mapped_ticker")["ISIN"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="isins")
    )
    duplicate_tickers = duplicate_tickers[duplicate_tickers["isins"].str.contains(",")]

    mapping_summary = pd.Series(
        {
            "trade_isins_total": int(isin_reference.shape[0]),
            "mapped_isins": int(mapped_isins.shape[0]),
            "unmapped_isins": int(unmapped_isins.shape[0]),
            "excluded_trades": int(excluded_trades.shape[0]) if not excluded_trades.empty else 0,
            "portfolio_tickers": int(trades["Ticker"].nunique()) if not trades.empty else 0,
            "duplicate_ticker_mappings": int(duplicate_tickers.shape[0]),
        }
    )

    return {
        "isin_reference": isin_reference.sort_values(["mapping_status", "ISIN"]),
        "unmapped_isins": unmapped_isins.sort_values("ISIN"),
        "duplicate_ticker_mappings": duplicate_tickers.sort_values("mapped_ticker"),
        "excluded_trades": excluded_trades,
        "mapping_summary": mapping_summary,
    }


def _holding_window(positions: pd.DataFrame, ticker: str) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if ticker not in positions.columns:
        return None, None
    held = positions[ticker].fillna(0) != 0
    if not held.any():
        return None, None
    idx = held[held].index
    return idx.min(), idx.max()


def build_sqlite_completeness_report(
    expected_tickers: Iterable[str],
    positions: pd.DataFrame,
    analysis_start: str,
    analysis_end: str,
    db_path: Path | str,
    db_update: pd.DataFrame,
    inventory: Optional[pd.DataFrame] = None,
    ticker_roles: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame | pd.Series | str]:
    expected = sorted({str(t).strip() for t in expected_tickers if str(t).strip()})
    inventory = inventory if inventory is not None else load_sqlite_inventory(db_path)
    raw_prices = load_raw_prices_from_db(expected, analysis_start, analysis_end, db_path)

    inv_by_ticker = inventory.set_index("ticker") if not inventory.empty else pd.DataFrame()
    rows = []
    for ticker in expected:
        role = (ticker_roles or {}).get(ticker, "portfolio")
        meta = inv_by_ticker.loc[ticker] if ticker in inv_by_ticker.index else None
        in_db = meta is not None
        row_count = int(meta["row_count"]) if in_db else 0
        db_first = meta["first_date"] if in_db else None
        db_last = meta["last_date"] if in_db else None
        source = meta["source"] if in_db else None

        hold_start, hold_end = _holding_window(positions, ticker) if role == "portfolio" else (
            pd.to_datetime(analysis_start),
            pd.to_datetime(analysis_end),
        )
        window_start = hold_start or pd.to_datetime(analysis_start)
        window_end = hold_end or pd.to_datetime(analysis_end)
        window_index = pd.bdate_range(window_start, window_end)

        series = raw_prices[ticker] if ticker in raw_prices.columns else pd.Series(dtype=float)
        aligned = series.reindex(window_index)
        available = int(aligned.notna().sum())
        expected_points = int(len(window_index))
        missing_points = expected_points - available
        coverage = available / expected_points if expected_points > 0 else np.nan

        stale_days = np.nan
        if in_db and db_last:
            stale_days = (pd.to_datetime(analysis_end) - pd.to_datetime(db_last)).days

        update_row = db_update[db_update["ticker"] == ticker].iloc[0] if (
            not db_update.empty and "ticker" in db_update.columns and (db_update["ticker"] == ticker).any()
        ) else None
        update_status = update_row["status"] if update_row is not None else ("missing" if not in_db else "unknown")
        update_error = update_row["error"] if update_row is not None else ("not_in_sqlite" if not in_db else None)

        if not in_db:
            status = "missing_from_sqlite"
        elif coverage < 0.5:
            status = "critical_gaps"
        elif coverage < 0.95:
            status = "partial_coverage"
        elif stale_days is not np.nan and stale_days > 5:
            status = "stale"
        else:
            status = "ok"

        rows.append(
            {
                "ticker": ticker,
                "role": role,
                "in_sqlite": in_db,
                "db_row_count": row_count,
                "db_first_date": db_first,
                "db_last_date": db_last,
                "db_source": source,
                "holding_start": window_start,
                "holding_end": window_end,
                "expected_points": expected_points,
                "available_points": available,
                "missing_points": missing_points,
                "coverage_ratio": coverage,
                "stale_days_vs_analysis_end": stale_days,
                "update_status": update_status,
                "update_error": update_error,
                "status": status,
            }
        )

    by_ticker = pd.DataFrame(rows).sort_values(["status", "coverage_ratio", "ticker"])
    issues = by_ticker[by_ticker["status"] != "ok"].copy()

    summary = pd.Series(
        {
            "sqlite_path": str(db_path),
            "sqlite_exists": Path(db_path).exists(),
            "tickers_in_sqlite_total": int(inventory.shape[0]) if not inventory.empty else 0,
            "expected_tickers": len(expected),
            "missing_from_sqlite": int((~by_ticker["in_sqlite"]).sum()),
            "critical_gaps": int((by_ticker["status"] == "critical_gaps").sum()),
            "partial_coverage": int((by_ticker["status"] == "partial_coverage").sum()),
            "stale_series": int((by_ticker["status"] == "stale").sum()),
            "ok_series": int((by_ticker["status"] == "ok").sum()),
            "mean_coverage": by_ticker["coverage_ratio"].mean(),
            "failed_updates": int(db_update["status"].eq("failed").sum()) if (
                not db_update.empty and "status" in db_update.columns
            ) else 0,
        }
    )

    return {
        "inventory": inventory,
        "by_ticker": by_ticker,
        "issues": issues,
        "summary": summary,
        "report_text": format_data_quality_report_text(summary, issues),
    }


def format_data_quality_report_text(summary: pd.Series, issues: pd.DataFrame) -> str:
    lines = [
        "DATA QUALITY REPORT",
        "===================",
        f"SQLite: {summary.get('sqlite_path', '')}",
        f"SQLite exists: {summary.get('sqlite_exists', False)}",
        f"Tickers stored in SQLite: {int(summary.get('tickers_in_sqlite_total', 0))}",
        f"Tickers required for analysis: {int(summary.get('expected_tickers', 0))}",
        f"Missing from SQLite: {int(summary.get('missing_from_sqlite', 0))}",
        f"Critical coverage gaps (<50%): {int(summary.get('critical_gaps', 0))}",
        f"Partial coverage (<95%): {int(summary.get('partial_coverage', 0))}",
        f"Stale series (>5d vs analysis end): {int(summary.get('stale_series', 0))}",
        f"Failed price updates: {int(summary.get('failed_updates', 0))}",
        f"Mean coverage during holding window: {summary.get('mean_coverage', np.nan):.1%}",
        "",
    ]

    if issues.empty:
        lines.append("No material data issues detected.")
        return "\n".join(lines)

    lines.append("Issues detected:")
    for _, row in issues.iterrows():
        lines.append(
            f"- {row['ticker']} [{row['role']}]: {row['status']} | "
            f"coverage={row['coverage_ratio']:.1%}, "
            f"db_rows={row['db_row_count']}, "
            f"db_range={row['db_first_date']}..{row['db_last_date']}, "
            f"update={row['update_status']}"
        )
        if pd.notna(row.get("update_error")) and row.get("update_error"):
            lines.append(f"  error: {row['update_error']}")
    return "\n".join(lines)


def build_full_data_quality_report(
    account: pd.DataFrame,
    isin_map: Dict[str, str],
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    expected_tickers: Iterable[str],
    analysis_start: str,
    analysis_end: str,
    db_path: Path | str,
    db_update: pd.DataFrame,
    asset_price_quality: pd.Series,
    asset_price_by_ticker: pd.DataFrame,
    benchmark_price_quality: pd.DataFrame,
    benchmark_tickers: Dict[str, str],
    isin_mapping_report: Optional[pd.DataFrame] = None,
    traded_universe: Optional[pd.DataFrame] = None,
    position_reconciliation: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    mapping = build_isin_mapping_report(account, isin_map, trades)
    portfolio_tickers = [t for t in expected_tickers if t not in benchmark_tickers.values()]
    ticker_roles = {t: "portfolio" for t in portfolio_tickers}
    if not positions.empty:
        ticker_roles.update({t: "portfolio" for t in positions.columns})
    ticker_roles.update({v: "benchmark" for v in benchmark_tickers.values()})

    sqlite_report = build_sqlite_completeness_report(
        expected_tickers=expected_tickers,
        positions=positions,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
        db_path=db_path,
        db_update=db_update,
        ticker_roles=ticker_roles,
    )

    combined_summary = pd.concat(
        [
            sqlite_report["summary"].rename(lambda x: f"sqlite_{x}"),
            mapping["mapping_summary"].rename(lambda x: f"mapping_{x}"),
            asset_price_quality.rename(lambda x: f"asset_{x}") if isinstance(asset_price_quality, pd.Series) else pd.Series(dtype=float),
        ]
    )

    report_text = sqlite_report["report_text"]
    report_text += "\n\nISIN / TICKER MAPPING\n"
    report_text += "=====================\n"
    report_text += f"Trade ISINs total: {int(mapping['mapping_summary'].get('trade_isins_total', 0))}\n"
    report_text += f"Mapped: {int(mapping['mapping_summary'].get('mapped_isins', 0))}\n"
    report_text += f"Unmapped: {int(mapping['mapping_summary'].get('unmapped_isins', 0))}\n"
    report_text += f"Excluded trades (no ticker): {int(mapping['mapping_summary'].get('excluded_trades', 0))}\n"

    unmapped = mapping["unmapped_isins"]
    if not unmapped.empty:
        report_text += "\nUnmapped ISINs:\n"
        for _, row in unmapped.iterrows():
            report_text += f"- {row['ISIN']} | {row['product']} | trades={row['trade_count']}\n"

    dupes = mapping["duplicate_ticker_mappings"]
    if not dupes.empty:
        report_text += "\nPossible mapping collisions (multiple ISINs -> same ticker):\n"
        for _, row in dupes.iterrows():
            report_text += f"- {row['mapped_ticker']} <- {row['isins']}\n"

    if isin_mapping_report is not None and not isin_mapping_report.empty:
        unmapped_map = isin_mapping_report[isin_mapping_report["status"] == "unmapped"]
        if not unmapped_map.empty:
            report_text += "\nUnmapped ISINs (mapping report):\n"
            for _, row in unmapped_map.iterrows():
                report_text += f"- {row['ISIN']}\n"

    if traded_universe is not None and not traded_universe.empty:
        report_text += f"\nTraded / held ISIN universe: {len(traded_universe)} ISINs\n"

    if position_reconciliation is not None and not position_reconciliation.empty:
        mismatches = position_reconciliation[position_reconciliation["status"] != "match"]
        report_text += "\nPOSITION RECONCILIATION (DeGiro vs calculated)\n"
        report_text += "============================================\n"
        report_text += f"Rows compared: {len(position_reconciliation)}\n"
        report_text += f"Matches: {int((position_reconciliation['status'] == 'match').sum())}\n"
        report_text += f"Mismatches: {int((position_reconciliation['status'] == 'mismatch').sum())}\n"
        report_text += f"Missing calculated: {int((position_reconciliation['status'] == 'missing_calculated').sum())}\n"
        if not mismatches.empty:
            report_text += "\nLargest differences:\n"
            for _, row in mismatches.head(15).iterrows():
                report_text += (
                    f"- {row['ISIN']} ({row['ticker']}): degiro={row['degiro_quantity']}, "
                    f"calc={row['calculated_quantity']}, diff={row['difference']}, status={row['status']}\n"
                )

    account_trade_rows = int(account["Description"].astype(str).str.contains("Compra|Venta", na=False).sum()) if "Description" in account.columns else 0
    if account_trade_rows < 100:
        report_text += (
            "\nWARNING: Cached account report looks truncated "
            f"({account_trade_rows} trade rows). Run with --refresh-degiro to rebuild full history.\n"
        )

    return {
        **mapping,
        **sqlite_report,
        "asset_price_quality": asset_price_by_ticker,
        "asset_price_quality_summary": asset_price_quality,
        "benchmark_price_quality": benchmark_price_quality,
        "combined_summary": combined_summary,
        "report_text": report_text,
        "db_update": db_update,
        "isin_mapping_report": isin_mapping_report if isin_mapping_report is not None else pd.DataFrame(),
        "traded_universe": traded_universe if traded_universe is not None else pd.DataFrame(),
        "position_reconciliation": position_reconciliation if position_reconciliation is not None else pd.DataFrame(),
    }


def write_data_quality_report(data_quality: Dict[str, object], export_dir: Path | str) -> Path:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    report_path = export_dir / "data_quality_report.txt"
    report_path.write_text(str(data_quality.get("report_text", "")), encoding="utf-8")
    return report_path
