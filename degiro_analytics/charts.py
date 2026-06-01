from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _save(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


def plot_equity_curve(portfolio: pd.DataFrame, out_dir: Path) -> go.Figure:
    fig = px.line(portfolio, x=portfolio.index, y="Total", title="Portfolio Equity Curve")
    fig.update_layout(yaxis_title="EUR", xaxis_title="Date", template="plotly_white")
    _save(fig, out_dir / "01_equity_curve.html")
    return fig


def plot_drawdown(portfolio: pd.DataFrame, out_dir: Path) -> go.Figure:
    fig = px.area(
        portfolio,
        x=portfolio.index,
        y="Drawdown",
        title="Portfolio Drawdown",
    )
    fig.update_layout(yaxis_tickformat=".1%", template="plotly_white")
    _save(fig, out_dir / "02_drawdown.html")
    return fig


def plot_returns_distribution(returns: pd.Series, out_dir: Path) -> go.Figure:
    fig = px.histogram(returns, nbins=60, title="Daily Returns Distribution")
    fig.update_layout(xaxis_tickformat=".2%", template="plotly_white")
    _save(fig, out_dir / "03_returns_hist.html")
    return fig


def plot_rolling_metrics(rolling_df: pd.DataFrame, out_dir: Path) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Rolling Volatility", "Rolling Sharpe"))
    fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df["rolling_vol"], name="Vol"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_df.index, y=rolling_df["rolling_sharpe"], name="Sharpe"), row=2, col=1)
    fig.update_layout(title="Rolling Risk Metrics (63d)", template="plotly_white", height=600)
    _save(fig, out_dir / "04_rolling_metrics.html")
    return fig


def plot_weights(weights: pd.DataFrame, out_dir: Path) -> go.Figure:
    fig = px.area(weights, title="Portfolio Weights Over Time")
    fig.update_layout(yaxis_tickformat=".0%", template="plotly_white")
    _save(fig, out_dir / "05_weights.html")
    return fig


def plot_contribution(summary: pd.DataFrame, out_dir: Path) -> go.Figure:
    if summary.empty:
        return go.Figure()
    fig = px.bar(
        summary.reset_index().rename(columns={"index": "Ticker"}),
        x="Ticker",
        y="Total Contribution",
        title="Return Contribution by Asset",
    )
    fig.update_layout(template="plotly_white")
    _save(fig, out_dir / "06_contribution.html")
    return fig


def plot_dividends(dividends: pd.DataFrame, dividend_summary: pd.DataFrame, out_dir: Path) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cumulative Dividends (EUR)", "Dividends by Ticker"))
    if not dividends.empty:
        gross = dividends[dividends["Description"].astype(str).str.contains("^Dividendo", case=False, na=False)]
        cum = gross.groupby("Fecha")["Value_EUR"].sum().cumsum()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Cumulative"), row=1, col=1)
    if not dividend_summary.empty:
        fig.add_trace(
            go.Bar(x=dividend_summary.index, y=dividend_summary["gross_eur"], name="By Ticker"),
            row=1,
            col=2,
        )
    fig.update_layout(title="Dividend Analysis", template="plotly_white")
    _save(fig, out_dir / "07_dividends.html")
    return fig


def plot_benchmark_equity(equity: pd.DataFrame, out_dir: Path) -> go.Figure:
    if equity.empty:
        return go.Figure()
    norm = equity / equity.iloc[0]
    fig = px.line(norm, title="Portfolio vs Benchmarks (Rebased)")
    fig.update_layout(template="plotly_white")
    _save(fig, out_dir / "08_benchmark_equity.html")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame, out_dir: Path) -> go.Figure:
    if corr.empty:
        return go.Figure()
    fig = px.imshow(corr, text_auto=".2f", title="Asset Return Correlation", color_continuous_scale="RdBu_r")
    fig.update_layout(template="plotly_white")
    _save(fig, out_dir / "09_correlation.html")
    return fig


def plot_efficient_frontier(frontier: pd.DataFrame, stats: pd.DataFrame, out_dir: Path) -> go.Figure:
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier["volatility"],
                y=frontier["return"],
                mode="lines",
                name="Efficient Frontier",
            )
        )
    if not stats.empty:
        fig.add_trace(
            go.Scatter(
                x=stats["volatility"],
                y=stats["return"],
                mode="markers+text",
                text=stats.index,
                textposition="top center",
                name="Optimized Portfolios",
                marker=dict(size=12),
            )
        )
    fig.update_layout(
        title="Markowitz Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        template="plotly_white",
    )
    _save(fig, out_dir / "10_efficient_frontier.html")
    return fig


def plot_optimized_weights(portfolios: pd.DataFrame, current_weights: pd.Series, out_dir: Path) -> go.Figure:
    if portfolios.empty:
        return go.Figure()
    df = portfolios.copy()
    df["Current"] = current_weights.reindex(df.index).fillna(0)
    fig = px.bar(df, barmode="group", title="Current vs Optimized Weights")
    fig.update_layout(yaxis_tickformat=".0%", template="plotly_white")
    _save(fig, out_dir / "11_optimized_weights.html")
    return fig


def plot_regime_probabilities(regime_probs: pd.DataFrame, out_dir: Path) -> go.Figure:
    if regime_probs.empty:
        return go.Figure()
    fig = px.area(regime_probs, title="Markov Regime Probabilities")
    fig.update_layout(yaxis_tickformat=".0%", template="plotly_white")
    _save(fig, out_dir / "12_regime_probabilities.html")
    return fig


def plot_regime_returns(returns: pd.Series, regime_labels: pd.Series, out_dir: Path) -> go.Figure:
    if returns.empty or regime_labels.empty:
        return go.Figure()
    df = pd.DataFrame({"returns": returns, "regime": regime_labels}).dropna()
    fig = px.box(df, x="regime", y="returns", title="Return Distribution by Regime")
    fig.update_layout(yaxis_tickformat=".2%", template="plotly_white")
    _save(fig, out_dir / "13_regime_returns.html")
    return fig


def plot_sqlite_completeness(issues: pd.DataFrame, by_ticker: pd.DataFrame, out_dir: Path) -> go.Figure:
    if by_ticker.empty:
        return go.Figure()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coverage During Holding Window", "SQLite Status"),
    )
    fig.add_trace(
        go.Bar(
            x=by_ticker["ticker"],
            y=by_ticker["coverage_ratio"],
            marker_color=by_ticker["status"].map(
                {
                    "ok": "#2ca02c",
                    "stale": "#ff7f0e",
                    "partial_coverage": "#ffbb78",
                    "critical_gaps": "#d62728",
                    "missing_from_sqlite": "#9467bd",
                }
            ).fillna("#7f7f7f"),
            name="Coverage",
        ),
        row=1,
        col=1,
    )
    status_counts = by_ticker["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig.add_trace(
        go.Bar(x=status_counts["status"], y=status_counts["count"], name="Status Count"),
        row=1,
        col=2,
    )
    fig.update_layout(title="SQLite Data Completeness", template="plotly_white", showlegend=False)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    _save(fig, out_dir / "16_sqlite_completeness.html")
    return fig


def plot_mapping_issues(unmapped: pd.DataFrame, out_dir: Path) -> go.Figure:
    if unmapped.empty:
        return go.Figure()
    fig = px.bar(
        unmapped,
        x="ISIN",
        y="trade_count",
        hover_data=["product"],
        title="Unmapped ISINs (Trades Excluded From Analysis)",
    )
    fig.update_layout(template="plotly_white")
    _save(fig, out_dir / "17_mapping_issues.html")
    return fig


def plot_data_quality(quality: pd.DataFrame, out_dir: Path) -> go.Figure:
    if quality.empty:
        return go.Figure()
    fig = px.bar(quality, x="ticker", y="coverage_ratio", title="EOD Price Coverage by Ticker")
    fig.update_layout(yaxis_tickformat=".0%", template="plotly_white")
    _save(fig, out_dir / "14_data_quality.html")
    return fig


def plot_factor_exposures(factor_df: pd.DataFrame, out_dir: Path) -> go.Figure:
    if factor_df.empty:
        return go.Figure()
    cols = [c for c in factor_df.columns if c.startswith("Beta_")]
    fig = px.bar(factor_df[cols].T, title="Fama-French Factor Exposures")
    fig.update_layout(template="plotly_white")
    _save(fig, out_dir / "15_factor_exposures.html")
    return fig


def generate_all_charts(results: Dict[str, object], chart_dir: Path) -> Dict[str, go.Figure]:
    chart_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    portfolio = results.get("portfolio")
    if isinstance(portfolio, pd.DataFrame) and not portfolio.empty:
        figures["equity"] = plot_equity_curve(portfolio, chart_dir)
        figures["drawdown"] = plot_drawdown(portfolio, chart_dir)
        figures["returns_hist"] = plot_returns_distribution(portfolio["Returns"], chart_dir)
    rolling = results.get("rolling_metrics")
    if isinstance(rolling, pd.DataFrame):
        figures["rolling"] = plot_rolling_metrics(rolling, chart_dir)
    contribution = results.get("contribution", {})
    if isinstance(contribution, dict):
        weights = contribution.get("weights")
        summary = contribution.get("summary")
        if isinstance(weights, pd.DataFrame):
            figures["weights"] = plot_weights(weights, chart_dir)
        if isinstance(summary, pd.DataFrame):
            figures["contribution"] = plot_contribution(summary, chart_dir)
    dividends = results.get("dividends")
    dividend_summary = results.get("dividend_summary")
    if isinstance(dividends, pd.DataFrame) and isinstance(dividend_summary, pd.DataFrame):
        figures["dividends"] = plot_dividends(dividends, dividend_summary, chart_dir)
    benchmark = results.get("benchmark", {})
    if isinstance(benchmark, dict):
        equity = benchmark.get("equity")
        if isinstance(equity, pd.DataFrame):
            figures["benchmark"] = plot_benchmark_equity(equity, chart_dir)
    optimization = results.get("optimization", {})
    if isinstance(optimization, dict):
        corr = optimization.get("corr")
        frontier = optimization.get("frontier")
        stats = optimization.get("portfolio_stats")
        portfolios = optimization.get("portfolios")
        current = results.get("current_weights")
        if isinstance(corr, pd.DataFrame):
            figures["corr"] = plot_correlation_heatmap(corr, chart_dir)
        if isinstance(frontier, pd.DataFrame) and isinstance(stats, pd.DataFrame):
            figures["frontier"] = plot_efficient_frontier(frontier, stats, chart_dir)
        if isinstance(portfolios, pd.DataFrame) and isinstance(current, pd.Series):
            figures["opt_weights"] = plot_optimized_weights(portfolios, current, chart_dir)
    regimes = results.get("regimes", {})
    if isinstance(regimes, dict):
        probs = regimes.get("regime_probabilities")
        labels = regimes.get("regime_labels")
        if isinstance(probs, pd.DataFrame):
            figures["regime_probs"] = plot_regime_probabilities(probs, chart_dir)
        if isinstance(labels, pd.Series) and isinstance(portfolio, pd.DataFrame):
            figures["regime_returns"] = plot_regime_returns(portfolio["Returns"], labels, chart_dir)
    quality = results.get("data_quality", {})
    if isinstance(quality, dict):
        by_ticker = quality.get("by_ticker")
        if isinstance(by_ticker, pd.DataFrame):
            portfolio_only = by_ticker[by_ticker["role"] == "portfolio"]
            figures["quality"] = plot_data_quality(portfolio_only, chart_dir)
        issues = quality.get("issues")
        if isinstance(by_ticker, pd.DataFrame) and isinstance(issues, pd.DataFrame):
            figures["sqlite"] = plot_sqlite_completeness(issues, by_ticker, chart_dir)
        unmapped = quality.get("unmapped_isins")
        if isinstance(unmapped, pd.DataFrame):
            figures["mapping"] = plot_mapping_issues(unmapped, chart_dir)
    factor = results.get("factor_portfolio")
    if isinstance(factor, pd.Series) and not factor.empty:
        figures["factors"] = plot_factor_exposures(factor.to_frame("Portfolio").T, chart_dir)
    return figures
