"""
Shared statistical helper functions.

These are used across pages (results analysis, balance plots) and CLI-style
diagnostics to reduce duplication and keep calculations consistent.
"""

from __future__ import annotations

from typing import Optional, Sequence, Callable, Any, Tuple, Dict, List

import numpy as np
import pandas as pd


def smd(x1: pd.Series | np.ndarray, x2: pd.Series | np.ndarray) -> float:
    """
    Standardized Mean Difference (absolute).

    Returns NaN if either sample has <2 observations or pooled variance is zero.
    """
    if len(x1) < 2 or len(x2) < 2:
        return np.nan

    # Support both Series and ndarray inputs
    x1s = pd.Series(x1).astype(float)
    x2s = pd.Series(x2).astype(float)

    pooled = np.sqrt((x1s.var(ddof=1) + x2s.var(ddof=1)) / 2)
    return abs(x1s.mean() - x2s.mean()) / pooled if pooled else np.nan


def cuped_adjust(pre: pd.Series, post: pd.Series) -> Optional[pd.Series]:
    """
    CUPED adjustment.

    Returns adjusted post series, or None if var(pre)==0 (no adjustment possible).
    """
    X = pre.values
    Y = post.values

    var_x = np.var(X, ddof=1)
    if var_x == 0:
        return None

    theta = np.cov(Y, X, ddof=1)[0, 1] / var_x
    return post - theta * (pre - X.mean())


def cuped_adjust_multi(
    pre_df: pd.DataFrame,
    post: pd.Series,
) -> Optional[Tuple[pd.Series, float, Dict[str, float]]]:
    """
    Multivariate CUPED adjustment.

    Fits a pooled OLS of `post` on the columns of `pre_df` (centered) and
    returns the residual-adjusted post values:

        Y_adj = Y - sum_i theta_i * (X_i - mean(X_i))

    Args:
        pre_df: DataFrame of pre-experiment covariate columns (n_samples, k).
        post:   Series of post-experiment values aligned with pre_df.index.

    Returns:
        (adjusted_post, r_squared, theta_dict) where:
          - adjusted_post is a pd.Series with the same index as `post`. Rows
            where any covariate or the outcome is NaN are kept as NaN.
          - r_squared is the model's coefficient of determination on the
            non-NaN rows (0.0 if total variance is 0).
          - theta_dict maps pre column name -> coefficient.

        Returns None if the design matrix is rank-deficient or has no rows
        with complete data.
    """
    if not isinstance(pre_df, pd.DataFrame) or pre_df.shape[1] == 0:
        return None

    aligned = pre_df.copy()
    aligned["__y__"] = post

    mask = aligned.notna().all(axis=1)
    if mask.sum() < pre_df.shape[1] + 1:
        return None

    X = aligned.loc[mask, pre_df.columns].to_numpy(dtype=float)  # (n, k)
    Y = aligned.loc[mask, "__y__"].to_numpy(dtype=float)         # (n,)

    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    Y_mean = Y.mean()
    Y_centered = Y - Y_mean

    # Solve centered OLS: minimise || Y_c - X_c @ theta ||
    try:
        theta, *_ = np.linalg.lstsq(X_centered, Y_centered, rcond=None)
    except np.linalg.LinAlgError:
        return None

    Y_hat = X_centered @ theta
    ss_res = float(np.sum((Y_centered - Y_hat) ** 2))
    ss_tot = float(np.sum(Y_centered ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    full_X = pre_df.to_numpy(dtype=float) - X_mean  # broadcasts; NaNs stay NaN
    correction = full_X @ theta
    adjusted = post.astype(float) - pd.Series(correction, index=pre_df.index)

    theta_dict = {col: float(theta[i]) for i, col in enumerate(pre_df.columns)}
    return adjusted, float(r2), theta_dict


def pairwise_matrix(groups: Sequence[Any], fn: Callable[[Any, Any], float]) -> pd.DataFrame:
    """Create a full pairwise matrix (diagonal NaN) for a function over group pairs."""
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat

