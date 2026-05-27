"""
Plotly visualisations for the Heterogeneity Analysis tab.

Outputs:
- CATE distribution histogram (with ATE line and optional CI band)
- Top-K feature importance horizontal bar
- Qini / cumulative-gain curve on the holdout
- Subgroup mean-CATE bar (per categorical level or per quantile bin)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def cate_histogram(
    cate: np.ndarray,
    ate: float,
    ci: Optional[Tuple[float, float]] = None,
    title: str = "Predicted Treatment Effect (CATE) distribution",
) -> go.Figure:
    """Histogram of per-user CATE with the ATE marked and optional 95% CI band."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cate, nbinsx=60, marker_color="#4a90e2", opacity=0.85,
        name="Per-user CATE", showlegend=False,
    ))
    # ATE vertical line
    fig.add_vline(
        x=ate, line_color="#d62728", line_width=2, line_dash="dash",
        annotation_text=f"ATE = {ate:+.3f}", annotation_position="top right",
    )
    if ci is not None:
        lo, hi = ci
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#d62728", opacity=0.12, line_width=0,
                      annotation_text=f"95% CI [{lo:+.3f}, {hi:+.3f}]",
                      annotation_position="bottom right")
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color="#1f77b4")),
        xaxis_title="Estimated treatment effect (CATE)",
        yaxis_title="Number of users",
        template="plotly_white",
        height=380,
        bargap=0.02,
    )
    return fig


def feature_importance_bar(
    importance: pd.Series,
    top_k: int = 15,
    title: str = "Feature importance",
) -> go.Figure:
    """Horizontal bar of top-K feature importances (already sorted descending)."""
    top = importance.head(top_k).iloc[::-1]  # reverse so largest is at the top
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color="#1f77b4",
        text=[f"{v:.3f}" for v in top.values],
        textposition="outside", cliponaxis=False,
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color="#1f77b4")),
        xaxis_title="Importance",
        yaxis_title="",
        template="plotly_white",
        height=max(320, 30 * len(top) + 100),
        margin=dict(l=180),
    )
    return fig


def qini_plot(
    qini_df: pd.DataFrame,
    auuc: float,
    title: str = "Qini curve (holdout)",
) -> go.Figure:
    """Cumulative incremental outcome vs fraction-targeted, plus random baseline."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qini_df["frac"], y=qini_df["qini"],
        mode="lines", name="Model (rank by CATE)",
        line=dict(color="#1f77b4", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=qini_df["frac"], y=qini_df["random"],
        mode="lines", name="Random targeting",
        line=dict(color="#7f7f7f", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text=f"{title}  ·  AUUC vs random = {auuc:+.2f}",
                   x=0.5, xanchor="center", font=dict(size=18, color="#1f77b4")),
        xaxis=dict(title="Fraction of population targeted", tickformat=".0%", range=[0, 1]),
        yaxis_title="Cumulative incremental outcome",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def subgroup_bar(
    subgroup_df: pd.DataFrame,
    label: str,
    title: str,
) -> go.Figure:
    """
    Horizontal bar of mean CATE per subgroup. ``subgroup_df`` has columns
    ``mean_cate`` and ``n`` with the subgroup label as index.
    """
    df = subgroup_df.copy()
    # Show in order of mean_cate so the chart is sorted
    df = df.sort_values("mean_cate")
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in df["mean_cate"]]
    text = [f"{v:+.3f} (n={int(n)})" for v, n in zip(df["mean_cate"], df["n"])]
    fig = go.Figure(go.Bar(
        x=df["mean_cate"], y=df.index.astype(str),
        orientation="h",
        marker_color=colors,
        text=text, textposition="outside", cliponaxis=False,
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16, color="#1f77b4")),
        xaxis_title=f"Mean CATE by {label}",
        yaxis_title="",
        template="plotly_white",
        height=max(280, 30 * len(df) + 100),
        margin=dict(l=160),
    )
    return fig
