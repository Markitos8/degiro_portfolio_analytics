from __future__ import annotations

import datetime as dt
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from degiro_connector.trading.api import API as TradingAPI
from degiro_connector.trading.models.account import Format, ReportRequest
from degiro_connector.trading.models.credentials import build_credentials
from degiro_connector.trading.models.transaction import HistoryRequest

from degiro_analytics.isin_mapping import is_valid_isin
from degiro_analytics.settings import CACHE_DIR, CONFIG_PATH, DEGIRO_EXCEL_DIR


def connect_degiro(config_path: Optional[Path | str] = None) -> TradingAPI:
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.extend([CONFIG_PATH, Path.cwd() / "config" / "config.json"])
    resolved = next((p for p in candidates if p.exists()), None)
    if resolved is None:
        raise FileNotFoundError(
            "No DeGiro config found. Copy mark_ines/medium/config/config.json to config/config.json"
        )
    credentials = build_credentials(location=str(resolved))
    api = TradingAPI(credentials=credentials)
    api.connect()
    return api


def _excel_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64tz_dtype(safe[col]):
            safe[col] = safe[col].dt.tz_localize(None)
        elif safe[col].dtype == "object":
            safe[col] = safe[col].apply(
                lambda v: v.tz_localize(None) if isinstance(v, pd.Timestamp) and v.tz is not None else v
            )
    return safe


def _save_reports_to_excel(reports: Dict[str, pd.DataFrame], excel_dir: Path) -> Path:
    excel_dir.mkdir(parents=True, exist_ok=True)
    workbook = excel_dir / "degiro_reports.xlsx"
    safe_reports = {name: _excel_safe_frame(frame) for name, frame in reports.items()}
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in safe_reports.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    for name, frame in safe_reports.items():
        frame.to_excel(excel_dir / f"{name}.xlsx", index=False)
    return workbook


def extract_traded_isins(
    account: pd.DataFrame,
    positions: pd.DataFrame,
    transactions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: list[dict] = []

    if "ISIN" in account.columns:
        trade_mask = account["Description"].astype(str).str.contains("Compra|Venta", na=False)
        traded = account.loc[trade_mask].copy()
        traded["ISIN"] = traded["ISIN"].astype(str).str.strip().str.upper()
        for isin, group in traded.groupby("ISIN"):
            if not is_valid_isin(isin):
                continue
            rows.append(
                {
                    "ISIN": isin,
                    "product": group["Producto"].iloc[0],
                    "source": "account_trades",
                    "trade_count": int(len(group)),
                    "first_seen": group["Fecha"].min(),
                    "last_seen": group["Fecha"].max(),
                }
            )

    if "Symbol/ISIN" in positions.columns:
        pos = positions.copy()
        pos = pos[pos["Symbol/ISIN"].notna()]
        pos = pos[~pos["Producto"].astype(str).str.contains("CASH", case=False, na=False)]
        for _, row in pos.iterrows():
            isin = str(row["Symbol/ISIN"]).strip().upper()
            if not is_valid_isin(isin):
                continue
            rows.append(
                {
                    "ISIN": isin,
                    "product": row.get("Producto"),
                    "source": "positions_snapshot",
                    "trade_count": pd.NA,
                    "first_seen": pd.NaT,
                    "last_seen": pd.NaT,
                    "current_quantity": pd.to_numeric(row.get("Cantidad"), errors="coerce"),
                }
            )

    if rows:
        universe = pd.DataFrame(rows)
        agg = (
            universe.groupby("ISIN", as_index=False)
            .agg(
                product=("product", "first"),
                sources=("source", lambda s: ", ".join(sorted(set(s)))),
                trade_count=("trade_count", "max"),
                first_seen=("first_seen", "min"),
                last_seen=("last_seen", "max"),
                current_quantity=("current_quantity", "max"),
            )
            .sort_values("ISIN")
        )
        return agg

    return pd.DataFrame(columns=["ISIN", "product", "sources", "trade_count", "first_seen", "last_seen", "current_quantity"])


def fetch_degiro_reports(
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
    config_path: Optional[str] = None,
    cache_dir: Optional[Path | str] = None,
    use_cache: bool = True,
    save_excel: bool = True,
) -> Dict[str, pd.DataFrame]:
    cache_path = Path(cache_dir) if cache_dir else CACHE_DIR
    account_file = cache_path / "account_report.csv"
    positions_file = cache_path / "positions_report.csv"
    transactions_file = cache_path / "transactions_history.csv"

    if use_cache and account_file.exists() and positions_file.exists() and transactions_file.exists():
        reports = {
            "account": pd.read_csv(account_file, parse_dates=["Datetime", "Fecha", "Fecha valor"]),
            "positions": pd.read_csv(positions_file),
            "transactions": pd.read_csv(transactions_file),
        }
        if save_excel:
            _save_reports_to_excel(reports, DEGIRO_EXCEL_DIR)
        return reports

    from_date = from_date or dt.date(dt.date.today().year - 30, 1, 1)
    to_date = to_date or dt.date.today()
    api = connect_degiro(config_path=config_path)

    account_report = api.get_account_report(
        report_request=ReportRequest(
            country="FR",
            lang="fr",
            format=Format.CSV,
            from_date=from_date,
            to_date=to_date,
        ),
        raw=False,
    )
    positions_report = api.get_position_report(
        report_request=ReportRequest(
            country="FR",
            lang="fr",
            from_date=from_date,
            to_date=to_date,
        ),
        raw=False,
    )
    transactions_history = api.get_transactions_history(
        transaction_request=HistoryRequest(
            format=Format.CSV,
            from_date=from_date,
            to_date=to_date,
        ),
        raw=False,
    )

    account = pd.read_csv(
        StringIO(account_report.model_dump()["content"]),
        delimiter=",",
        quotechar='"',
        decimal=",",
    )
    account.columns = [
        "Fecha",
        "Hora",
        "Fecha valor",
        "Producto",
        "ISIN",
        "Description",
        "FX",
        "Curncy",
        "Value",
        "Curncy_2",
        "Cash balance",
        "ID Orden",
    ]
    account.insert(
        0,
        "Datetime",
        pd.to_datetime(account["Fecha"] + " " + account["Hora"], format="%d-%m-%Y %H:%M"),
    )
    account["Fecha"] = pd.to_datetime(account["Fecha"], format="%d-%m-%Y")
    account["Fecha valor"] = pd.to_datetime(account["Fecha valor"], format="%d-%m-%Y", errors="coerce")

    positions = pd.read_csv(
        StringIO(positions_report.model_dump()["content"]),
        delimiter=",",
        quotechar='"',
        decimal=",",
    )
    transactions = pd.DataFrame(transactions_history.model_dump()["data"])

    cache_path.mkdir(parents=True, exist_ok=True)
    reports = {"account": account, "positions": positions, "transactions": transactions}
    account.to_csv(account_file, index=False)
    positions.to_csv(positions_file, index=False)
    transactions.to_csv(transactions_file, index=False)
    if save_excel:
        _save_reports_to_excel(reports, DEGIRO_EXCEL_DIR)
    return reports
