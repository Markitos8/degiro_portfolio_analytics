from __future__ import annotations

import re
import zipfile
from io import BytesIO, StringIO
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from degiro_analytics.settings import PERIODS_PER_YEAR


def _sanitize_returns(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> pd.Series:
    returns = _sanitize_returns(returns)
    if returns.empty:
        return pd.Series(dtype=float)
    total_return = (1 + returns).prod() - 1
    years = len(returns) / periods_per_year
    if years <= 0:
        cagr = np.nan
    elif (1 + total_return) <= 0:
        cagr = -1.0
    else:
        cagr = (1 + total_return) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year) / vol if vol and not np.isnan(vol) else np.nan
    sortino_downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = (returns.mean() * periods_per_year) / sortino_downside if sortino_downside else np.nan
    dd = ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min()
    calmar = cagr / abs(dd) if dd and dd < 0 else np.nan
    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": dd,
            "Calmar": calmar,
            "Win Rate": (returns > 0).mean(),
        }
    )


def compute_asset_metrics(equity_by_ticker: pd.DataFrame, periods_per_year: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    if equity_by_ticker.empty:
        return pd.DataFrame()
    metrics = {}
    for ticker in equity_by_ticker.columns:
        ret = _sanitize_returns(equity_by_ticker[ticker].pct_change())
        if ret.empty:
            continue
        metrics[ticker] = compute_performance_metrics(ret, periods_per_year=periods_per_year)
    return pd.DataFrame(metrics).T.sort_values("Total Return", ascending=False) if metrics else pd.DataFrame()


def build_benchmark_comparison(
    portfolio_returns: pd.Series,
    benchmark_tickers: Dict[str, str],
    benchmark_prices: pd.DataFrame,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> Dict[str, pd.DataFrame]:
    if portfolio_returns.empty or not benchmark_tickers or benchmark_prices.empty:
        return {"prices": pd.DataFrame(), "returns": pd.DataFrame(), "metrics": pd.DataFrame(), "equity": pd.DataFrame()}

    rename_map = {v: k for k, v in benchmark_tickers.items() if v in benchmark_prices.columns}
    bench_prices = benchmark_prices.rename(columns=rename_map)
    bench_returns = bench_prices.reindex(portfolio_returns.index).ffill().pct_change()
    bench_returns = bench_returns.replace([np.inf, -np.inf], np.nan)

    perf = {}
    for col in bench_returns.columns:
        metrics = compute_performance_metrics(bench_returns[col].dropna(), periods_per_year=periods_per_year)
        aligned = pd.concat(
            [portfolio_returns.rename("portfolio"), bench_returns[col].rename("benchmark")],
            axis=1,
        ).dropna()
        if aligned.empty:
            metrics["Correlation"] = np.nan
            metrics["Beta_to_Benchmark"] = np.nan
        else:
            metrics["Correlation"] = aligned["portfolio"].corr(aligned["benchmark"])
            denom = aligned["benchmark"].var()
            metrics["Beta_to_Benchmark"] = aligned.cov().loc["portfolio", "benchmark"] / denom if denom else np.nan
        perf[col] = metrics
    benchmark_metrics = pd.DataFrame(perf).T
    portfolio_equity = (1 + portfolio_returns.fillna(0)).cumprod()
    benchmark_equity = (1 + bench_returns.fillna(0)).cumprod()
    equity = benchmark_equity.copy()
    equity.insert(0, "Portfolio", portfolio_equity.reindex(benchmark_equity.index).ffill())
    return {
        "prices": bench_prices,
        "returns": bench_returns,
        "metrics": benchmark_metrics,
        "equity": equity,
    }


def fetch_fama_french_factors(model: str = "3f") -> pd.DataFrame:
    if model == "3f":
        zip_name = "F-F_Research_Data_Factors_daily_CSV.zip"
    elif model == "5f":
        zip_name = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    else:
        raise ValueError("model must be '3f' or '5f'")

    try:
        from pandas_datareader import data as pdr

        ds_name = zip_name.replace("_CSV.zip", "")
        ff = pdr.DataReader(ds_name, "famafrench")[0].copy()
        ff.index = pd.to_datetime(ff.index)
        return ff.rename(columns={"Mkt-RF": "MKT_RF"}) / 100.0
    except Exception:
        pass

    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{zip_name}"
    resp = requests.get(url, timeout=40)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        csv_candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        raw = zf.read(csv_candidates[0]).decode("latin1", errors="ignore")
    lines = raw.splitlines()
    header_idx = start_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bMkt-RF\b", line):
            header_idx = i
        if re.match(r"^\d{8},", line):
            start_idx = i
            break
    data_lines = [line for line in lines[start_idx:] if re.match(r"^\d{8},", line)]
    ff = pd.read_csv(StringIO("\n".join([lines[header_idx]] + data_lines)), index_col=0)
    ff.index = pd.to_datetime(ff.index.astype(str), format="%Y%m%d")
    return ff.rename(columns={"Mkt-RF": "MKT_RF"}) / 100.0


def run_factor_regression(target_returns: pd.Series, ff_factors: pd.DataFrame, model: str = "3f") -> pd.Series:
    if ff_factors is None or ff_factors.empty or "RF" not in ff_factors.columns:
        return pd.Series(dtype=float)
    clean_target = _sanitize_returns(target_returns)
    if clean_target.empty:
        return pd.Series(dtype=float)
    df = pd.concat([clean_target.rename("asset"), ff_factors], axis=1, sort=False).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    y = df["asset"] - df["RF"]
    x_cols = ["MKT_RF", "SMB", "HML"] if model == "3f" else ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    x = sm.add_constant(df[x_cols])
    if x.shape[0] < len(x_cols) + 2:
        return pd.Series(dtype=float)
    fit = sm.OLS(y, x).fit()
    out = fit.params.rename(
        {
            "const": "Alpha",
            "MKT_RF": "Beta_MKT",
            "SMB": "Beta_SMB",
            "HML": "Beta_HML",
            "RMW": "Beta_RMW",
            "CMA": "Beta_CMA",
        }
    )
    out["R2"] = fit.rsquared
    out["N_obs"] = int(fit.nobs)
    return out


def compute_rolling_metrics(returns: pd.Series, window: int = 63) -> pd.DataFrame:
    returns = _sanitize_returns(returns)
    if returns.empty:
        return pd.DataFrame()
    ann_factor = PERIODS_PER_YEAR
    rolling_vol = returns.rolling(window).std() * np.sqrt(ann_factor)
    rolling_mean = returns.rolling(window).mean() * ann_factor
    rolling_sharpe = rolling_mean / rolling_vol
    return pd.DataFrame(
        {
            "rolling_vol": rolling_vol,
            "rolling_return": rolling_mean,
            "rolling_sharpe": rolling_sharpe,
        }
    )
