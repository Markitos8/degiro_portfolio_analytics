from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from degiro_analytics.market_data import fetch_fx_rates
from degiro_analytics.settings import PERIODS_PER_YEAR


def _extract_quantity(desc: str) -> int:
    text = str(desc)
    split_match = re.search(r"STOCK SPLIT:\s*(\d+)", text)
    if split_match:
        return int(split_match.group(1))
    trade_match = re.search(r"(?:Compra|Venta)\s+(\d+)", text)
    return int(trade_match.group(1)) if trade_match else 0


def _extract_operation(desc: str) -> str:
    text = str(desc)
    if "STOCK SPLIT" in text:
        return "Split"
    return "Compra" if "Compra" in text else "Venta"


def _extract_split_signed_quantity(desc: str, value: float) -> int:
    match = re.search(r"STOCK SPLIT:\s*(\d+)", str(desc))
    if not match:
        return 0
    qty = int(match.group(1))
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value) or numeric_value == 0:
        return 0
    # DeGiro books split as two lines: negative Value adds new shares, positive Value removes old.
    return qty if numeric_value < 0 else -qty


def _extract_trade_price(desc: str) -> float:
    match = re.search(r"@([\d.,]+)", str(desc))
    if not match:
        return np.nan
    return float(match.group(1).replace(".", "").replace(",", "."))


def extract_stock_splits(account_report: pd.DataFrame) -> pd.DataFrame:
    mask = account_report["Description"].astype(str).str.contains("STOCK SPLIT", na=False)
    splits = account_report.loc[mask].copy()
    if splits.empty:
        return pd.DataFrame(
            columns=["Datetime", "Fecha", "Producto", "ISIN", "Description", "Value", "signed_quantity"]
        )
    splits["Value"] = pd.to_numeric(splits["Value"], errors="coerce")
    splits["signed_quantity"] = splits.apply(
        lambda row: _extract_split_signed_quantity(row["Description"], row["Value"]),
        axis=1,
    )
    return splits.sort_values("Datetime")


def extract_stock_splits_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "transaction_type_id" not in transactions.columns:
        return pd.DataFrame()
    splits = transactions.loc[transactions["transaction_type_id"] == 101].copy()
    if splits.empty:
        return splits
    splits["signed_quantity"] = pd.to_numeric(splits["quantity"], errors="coerce")
    return splits.sort_values("date")


def build_trades_table(account_report: pd.DataFrame, isin_ticker_map: Dict[str, str]) -> pd.DataFrame:
    desc = account_report["Description"].astype(str)
    trades = account_report[desc.str.contains("Compra|Venta|STOCK SPLIT", na=False)].copy()
    trades["Quantity"] = trades["Description"].apply(_extract_quantity)
    trades["Operation"] = trades["Description"].apply(_extract_operation)
    trades["Price"] = trades["Description"].apply(_extract_trade_price)
    trades["Value"] = pd.to_numeric(trades["Value"], errors="coerce")
    split_mask = trades["Operation"] == "Split"
    trades["Signed Quantity"] = np.where(
        split_mask,
        trades.apply(lambda row: _extract_split_signed_quantity(row["Description"], row["Value"]), axis=1),
        np.where(trades["Operation"] == "Compra", trades["Quantity"], -trades["Quantity"]),
    )
    trades["ISIN"] = trades["ISIN"].astype(str).str.strip().str.upper()
    trades["Ticker"] = trades["ISIN"].map(isin_ticker_map)
    trades["FX"] = pd.to_numeric(trades["FX"], errors="coerce")
    trades = trades[trades["Ticker"].notna()].copy()
    return trades.sort_values("Datetime")


def build_positions_from_trades(trades: pd.DataFrame, end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    position_changes = trades.groupby(["Fecha", "Ticker"])["Signed Quantity"].sum().unstack(fill_value=0)
    positions = position_changes.sort_index().cumsum()
    end_date = pd.to_datetime(end_date) if end_date is not None else pd.Timestamp.today().normalize()
    full_range = pd.date_range(start=positions.index.min(), end=end_date, freq="D")
    positions = positions.reindex(full_range).ffill().fillna(0)
    positions.index.name = "Date"
    return positions


def build_dividends_table(account_report: pd.DataFrame, isin_ticker_map: Dict[str, str]) -> pd.DataFrame:
    mask = account_report["Description"].astype(str).str.contains("Dividendo|dividend", case=False, na=False)
    divs = account_report.loc[mask].copy()
    if divs.empty:
        return pd.DataFrame(columns=["Datetime", "Fecha", "ISIN", "Ticker", "Producto", "Description", "Value", "Curncy"])
    divs["ISIN"] = divs["ISIN"].astype(str).str.strip().str.upper()
    divs["Ticker"] = divs["ISIN"].map(isin_ticker_map)
    divs["Value"] = pd.to_numeric(divs["Value"], errors="coerce")
    divs["FX"] = pd.to_numeric(divs.get("FX"), errors="coerce")
    divs["Value_EUR"] = divs["Value"]
    eur_mask = divs["Curncy"].astype(str) != "EUR"
    if eur_mask.any():
        divs.loc[eur_mask, "Value_EUR"] = divs.loc[eur_mask, "Value"] * divs.loc[eur_mask, "FX"].fillna(1.0)
    return divs.sort_values("Datetime")


def summarize_dividends(dividends: pd.DataFrame) -> pd.DataFrame:
    if dividends.empty:
        return pd.DataFrame(columns=["gross_eur", "count"])
    gross = dividends[dividends["Description"].astype(str).str.contains("^Dividendo", case=False, na=False)]
    by_ticker = gross.groupby("Ticker")["Value_EUR"].sum().rename("gross_eur").to_frame()
    by_ticker["count"] = gross.groupby("Ticker").size()
    by_ticker = by_ticker.sort_values("gross_eur", ascending=False)
    return by_ticker


def build_cash_series(account_report: pd.DataFrame) -> pd.Series:
    cash = account_report.groupby("Fecha")["Cash balance"].first()
    cash.index = pd.to_datetime(cash.index)
    return pd.to_numeric(cash, errors="coerce")


def build_portfolio_equity_curve(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    cash_series: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if positions.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices = prices.reindex(positions.index).ffill()
    ticker_to_currency = {}
    for ticker in positions.columns:
        try:
            fast_info = getattr(yf.Ticker(ticker), "fast_info", {}) or {}
            ticker_to_currency[ticker] = fast_info.get("currency", "EUR")
        except Exception:
            ticker_to_currency[ticker] = "EUR"
    fx_rates = fetch_fx_rates(ticker_to_currency, positions.index)

    equity_by_ticker = pd.DataFrame(index=positions.index)
    for ticker in positions.columns:
        fx = fx_rates.get(ticker_to_currency[ticker], pd.Series(1.0, index=positions.index))
        if ticker in prices.columns:
            price_series = pd.to_numeric(prices[ticker], errors="coerce").ffill().fillna(0.0)
            equity_by_ticker[ticker] = positions[ticker] * price_series * fx

    total = equity_by_ticker.sum(axis=1).add(cash_series.reindex(positions.index).ffill().fillna(0), fill_value=0)
    portfolio = pd.DataFrame({"Total": total})
    portfolio["Returns"] = portfolio["Total"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    portfolio["Drawdown"] = portfolio["Total"] / portfolio["Total"].cummax() - 1
    return portfolio, equity_by_ticker


def compute_asset_contribution(equity_by_ticker: pd.DataFrame, portfolio_total: pd.Series) -> Dict[str, pd.DataFrame]:
    if equity_by_ticker.empty:
        return {"weights": pd.DataFrame(), "daily_contribution": pd.DataFrame(), "summary": pd.DataFrame()}

    total = portfolio_total.reindex(equity_by_ticker.index).replace(0, np.nan)
    weights = equity_by_ticker.div(total, axis=0).fillna(0)
    asset_returns = equity_by_ticker.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    daily_contribution = weights.shift(1).fillna(0) * asset_returns
    summary = pd.DataFrame(
        {
            "Average Weight": weights.mean(),
            "Total Contribution": daily_contribution.sum(),
            "Asset Return": (1 + asset_returns).prod() - 1,
            "Asset Max Drawdown": (equity_by_ticker / equity_by_ticker.cummax() - 1).min(),
        }
    ).sort_values("Total Contribution", ascending=False)
    return {"weights": weights, "daily_contribution": daily_contribution, "summary": summary}


def build_data_quality_report(price_data: pd.DataFrame, expected_index: pd.DatetimeIndex) -> Dict[str, pd.DataFrame | pd.Series]:
    if price_data is None or price_data.empty:
        empty = pd.DataFrame(
            columns=["ticker", "expected_points", "available_points", "missing_points", "coverage_ratio"]
        )
        return {"by_ticker": empty, "summary": pd.Series(dtype=float)}

    aligned = price_data.reindex(expected_index)
    rows = []
    for ticker in aligned.columns:
        s = pd.to_numeric(aligned[ticker], errors="coerce")
        available = int(s.notna().sum())
        expected = int(len(expected_index))
        rows.append(
            {
                "ticker": ticker,
                "expected_points": expected,
                "available_points": available,
                "missing_points": expected - available,
                "coverage_ratio": available / expected if expected > 0 else np.nan,
                "first_valid": s.first_valid_index(),
                "last_valid": s.last_valid_index(),
            }
        )
    by_ticker = pd.DataFrame(rows).sort_values(["coverage_ratio", "missing_points"], ascending=[True, False])
    summary = pd.Series(
        {
            "tickers": by_ticker.shape[0],
            "mean_coverage": by_ticker["coverage_ratio"].mean(),
            "tickers_below_95pct": int((by_ticker["coverage_ratio"] < 0.95).sum()),
            "total_missing_points": int(by_ticker["missing_points"].sum()),
        }
    )
    return {"by_ticker": by_ticker, "summary": summary}


def reconcile_positions_with_degiro(
    calculated_positions: pd.DataFrame,
    degiro_positions: pd.DataFrame,
    isin_map: Dict[str, str],
) -> pd.DataFrame:
    if calculated_positions.empty or degiro_positions.empty:
        return pd.DataFrame()

    latest = calculated_positions.iloc[-1]
    snapshot = degiro_positions.copy()
    snapshot = snapshot[snapshot.get("Symbol/ISIN").notna()]
    snapshot = snapshot[~snapshot["Producto"].astype(str).str.contains("CASH", case=False, na=False)]
    rows = []
    for _, row in snapshot.iterrows():
        isin = str(row["Symbol/ISIN"]).strip().upper()
        ticker = isin_map.get(isin)
        degiro_qty = pd.to_numeric(row.get("Cantidad"), errors="coerce")
        calc_qty = pd.to_numeric(latest.get(ticker), errors="coerce") if ticker in latest.index else np.nan
        diff = calc_qty - degiro_qty if pd.notna(calc_qty) and pd.notna(degiro_qty) else np.nan
        if pd.isna(diff):
            status = "missing_calculated" if pd.isna(calc_qty) else "missing_degiro"
        elif np.isclose(diff, 0, atol=1e-6):
            status = "match"
        elif abs(diff) <= 1:
            status = "minor_diff"
        else:
            status = "mismatch"
        rows.append(
            {
                "ISIN": isin,
                "ticker": ticker,
                "product": row.get("Producto"),
                "degiro_quantity": degiro_qty,
                "calculated_quantity": calc_qty,
                "difference": diff,
                "abs_difference": abs(diff) if pd.notna(diff) else np.nan,
                "status": status,
            }
        )
    report = pd.DataFrame(rows).sort_values(["status", "abs_difference"], ascending=[True, False])
    return report
