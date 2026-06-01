from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from degiro_analytics.settings import PERIODS_PER_YEAR


def detect_markov_regimes(
    returns: pd.Series,
    n_regimes: int = 2,
    switching_variance: bool = True,
) -> Dict[str, pd.DataFrame | pd.Series]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(60, n_regimes * 20):
        return {
            "regime_probabilities": pd.DataFrame(),
            "regime_labels": pd.Series(dtype=int),
            "summary": pd.DataFrame(),
            "transition_matrix": pd.DataFrame(),
        }

    try:
        model = MarkovRegression(
            clean.values,
            k_regimes=n_regimes,
            trend="c",
            switching_variance=switching_variance,
        )
        result = model.fit(disp=False, maxiter=200)
        probs = pd.DataFrame(
            result.smoothed_marginal_probabilities,
            index=clean.index,
            columns=[f"regime_{i}" for i in range(n_regimes)],
        )
        labels = probs.idxmax(axis=1).str.replace("regime_", "", regex=False).astype(int)
        summary_rows = []
        for i in range(n_regimes):
            mask = labels == i
            regime_returns = clean[mask]
            summary_rows.append(
                {
                    "regime": i,
                    "observations": int(mask.sum()),
                    "mean_daily": regime_returns.mean(),
                    "vol_daily": regime_returns.std(),
                    "mean_ann": regime_returns.mean() * PERIODS_PER_YEAR,
                    "vol_ann": regime_returns.std() * np.sqrt(PERIODS_PER_YEAR),
                }
            )
        summary = pd.DataFrame(summary_rows)
        transition_arr = np.squeeze(np.asarray(result.regime_transition))
        if transition_arr.ndim == 1:
            transition_arr = transition_arr.reshape(n_regimes, n_regimes)
        transition = pd.DataFrame(
            transition_arr,
            index=[f"from_{i}" for i in range(n_regimes)],
            columns=[f"to_{j}" for j in range(n_regimes)],
        )
        return {
            "regime_probabilities": probs,
            "regime_labels": labels.rename("regime"),
            "summary": summary,
            "transition_matrix": transition,
            "model_aic": pd.Series({"AIC": result.aic, "BIC": result.bic}),
        }
    except Exception as exc:
        return {
            "regime_probabilities": pd.DataFrame(),
            "regime_labels": pd.Series(dtype=int),
            "summary": pd.DataFrame({"error": [str(exc)]}),
            "transition_matrix": pd.DataFrame(),
        }
