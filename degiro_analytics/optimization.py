from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from degiro_analytics.settings import PERIODS_PER_YEAR


def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[float, float, float]:
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = ret / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def objective(w):
        return w @ cov @ w

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else x0


def _max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    n = len(mu)
    x0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 0 else 0

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else x0


def hrp_weights(cov: np.ndarray, corr: np.ndarray) -> np.ndarray:
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = leaves_list(link)
    ordered_cov = cov[np.ix_(sort_ix, sort_ix)]
    n = len(sort_ix)
    weights = np.ones(n)

    def _cluster_var(cov_slice: np.ndarray) -> float:
        ivp = 1 / np.diag(cov_slice)
        ivp = ivp / ivp.sum()
        return float(ivp @ cov_slice @ ivp)

    clusters = [list(range(n))]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]
        cov_left = ordered_cov[np.ix_(left, left)]
        cov_right = ordered_cov[np.ix_(right, right)]
        var_left = _cluster_var(cov_left)
        var_right = _cluster_var(cov_right)
        alpha = 1 - var_left / (var_left + var_right) if (var_left + var_right) > 0 else 0.5
        weights[left] *= alpha
        weights[right] *= 1 - alpha
        if len(left) > 1:
            clusters.append(left)
        if len(right) > 1:
            clusters.append(right)

    final = np.zeros(n)
    final[sort_ix] = weights / weights.sum()
    return final


def compute_efficient_frontier(
    asset_returns: pd.DataFrame,
    n_portfolios: int = 200,
    rf: float = 0.0,
) -> Dict[str, object]:
    rets = asset_returns.dropna(how="all").fillna(0)
    rets = rets.loc[:, rets.std() > 1e-12]
    if rets.shape[1] < 2:
        return {"frontier": pd.DataFrame(), "portfolios": pd.DataFrame(), "portfolio_stats": pd.DataFrame()}

    mu = rets.mean().values * PERIODS_PER_YEAR
    cov = rets.cov().values * PERIODS_PER_YEAR
    corr = rets.corr().fillna(0).values
    np.fill_diagonal(corr, 1.0)
    labels = list(rets.columns)

    min_w = _min_variance_weights(cov)
    max_w = _max_sharpe_weights(mu, cov, rf=rf)
    hrp_w = hrp_weights(cov, corr)

    frontier_rows = []
    target_returns = np.linspace(mu.min(), mu.max(), n_portfolios)
    n = len(labels)
    for target in target_returns:
        x0 = np.ones(n) / n

        def objective(w):
            return w @ cov @ w

        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = tuple((0, 1) for _ in range(n))
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            ret, vol, sharpe = _portfolio_stats(res.x, mu, cov)
            frontier_rows.append({"return": ret, "volatility": vol, "sharpe": sharpe})

    portfolios = pd.DataFrame(
        {
            "Min Variance": min_w,
            "Max Sharpe": max_w,
            "HRP": hrp_w,
        },
        index=labels,
    )
    stats = {}
    for col in portfolios.columns:
        w = portfolios[col].values
        ret, vol, sharpe = _portfolio_stats(w, mu, cov)
        stats[col] = {"return": ret, "volatility": vol, "sharpe": sharpe}

    return {
        "frontier": pd.DataFrame(frontier_rows),
        "portfolios": portfolios,
        "portfolio_stats": pd.DataFrame(stats).T,
        "mu": pd.Series(mu, index=labels),
        "cov": pd.DataFrame(cov, index=labels, columns=labels),
        "corr": pd.DataFrame(corr, index=labels, columns=labels),
    }
