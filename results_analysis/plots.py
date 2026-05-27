"""
Plotly visualizations for treatment effect analysis
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Tuple
from scipy.stats import ttest_ind
from utils.stats import smd as _smd, cuped_adjust as _cuped_adjust, cuped_adjust_multi as _cuped_adjust_multi

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def _vspace_and_height(n_rows: int, per_row_px: int = 250, gap_px: int = 90) -> Tuple[float, int]:
    """
    Return (vertical_spacing_fraction, total_height_px) that keep each row
    and each gap at consistent pixel sizes regardless of n_rows.

    plotly's `vertical_spacing` is a fraction of total figure height. Holding
    it constant while the figure grows squeezes rows; conversely, picking it
    as a pixel value and back-computing the fraction keeps row sizes stable.
    """
    if n_rows <= 1:
        return 0.12, per_row_px
    total = n_rows * per_row_px + (n_rows - 1) * gap_px
    cap = 1.0 / (n_rows - 1)
    vspace = min(gap_px / total, 0.95 * cap)
    return vspace, total


def _vspace(n_rows: int) -> float:
    """Back-compat shim: returns just the spacing fraction."""
    return _vspace_and_height(n_rows)[0]


def _pairwise_matrix(groups, fn):
    """Create pairwise matrix"""
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                mat.loc[g1, g2] = fn(g1, g2)
    return mat


def _precompute_group_data(df: pd.DataFrame, group_column: str, columns: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Pre-compute group data arrays for efficient access.
    This avoids repeated DataFrame filtering operations.
    
    Args:
        df: DataFrame with group assignments
        group_column: Name of the column containing group assignments
        columns: List of column names to pre-compute
        
    Returns:
        dict: {group_name: {column_name: numpy_array}}
    """
    groups = sorted(df[group_column].unique())
    group_data = {}
    
    # Pre-compute group indices once
    group_indices = {}
    for g in groups:
        mask = df[group_column] == g
        group_indices[g] = df.index[mask]
    
    # Pre-extract data for each group and column
    for g in groups:
        group_data[g] = {}
        for col in columns:
            if col in df.columns:
                # Use pre-computed indices for fast access
                group_data[g][col] = df.loc[group_indices[g], col].dropna().values
    
    return group_data


def _create_basic_analysis_plotly_impl(
    df: pd.DataFrame,
    value_columns: List[str],
    group_column: str,
    title: str = "Treatment Effect Analysis",
    alternative: str = "two-sided"
) -> go.Figure:
    """
    Create Plotly visualization for basic treatment effect analysis.
    Shows means, uplifts, SMD, and p-values for each metric.
    """
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)
    
    if n_groups < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 groups for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Filter valid columns
    valid_columns = [col for col in value_columns if col in df.columns]
    
    if len(valid_columns) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid metric columns selected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Create subplots: 4 columns per metric (means, uplifts, SMD, p-values)
    n_rows = len(valid_columns)
    vspace, fig_height = _vspace_and_height(n_rows)
    fig = make_subplots(
        rows=n_rows, cols=4,
        subplot_titles=[f"{col}: {subtitle}" for col in valid_columns
                       for subtitle in ["Mean by Group", "Uplift (%)", "SMD", "p-value"]],
        vertical_spacing=vspace,
        horizontal_spacing=0.08
    )
    
    row_idx = 1
    
    # Pre-compute group data for all columns (Issue #1 fix)
    group_data_cache = _precompute_group_data(df, group_column, valid_columns)
    
    for col in valid_columns:
        # Calculate statistics for all groups using cached data
        group_means = {}
        group_data = {}
        
        for g in groups:
            if g in group_data_cache and col in group_data_cache[g]:
                data_array = group_data_cache[g][col]
                if len(data_array) >= 2:
                    group_means[g] = np.mean(data_array)
                    group_data[g] = data_array
        
        if len(group_means) < 2:
            row_idx += 1
            continue
        
        # Plot 1: Bar chart - Means
        fig.add_trace(
            go.Bar(
                x=list(group_means.keys()),
                y=list(group_means.values()),
                name=f"{col} Mean",
                marker_color='steelblue',
                text=list(group_means.values()),
                texttemplate='%{text:.3f}',
                textposition='outside',
                cliponaxis=False,
                showlegend=False
            ),
            row=row_idx, col=1
        )
        
        # Plot 2: Heatmap - Uplift %
        def uplift(g1, g2):
            if g1 in group_means and g2 in group_means and group_means[g1] != 0:
                return ((group_means[g2] - group_means[g1]) / group_means[g1] * 100)
            return np.nan
        
        uplift_mat = _pairwise_matrix(groups, uplift)
        
        fig.add_trace(
            go.Heatmap(
                z=uplift_mat.values,
                x=uplift_mat.columns,
                y=uplift_mat.index,
                colorscale='RdBu_r',
                text=uplift_mat.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10},
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=2
        )
        
        # Plot 3: Heatmap - SMD
        def smd_fn(g1, g2):
            if g1 in group_data and g2 in group_data:
                return _smd(group_data[g1], group_data[g2])
            return np.nan
        
        smd_mat = _pairwise_matrix(groups, smd_fn)
        
        fig.add_trace(
            go.Heatmap(
                z=smd_mat.values,
                x=smd_mat.columns,
                y=smd_mat.index,
                colorscale='Reds',
                text=smd_mat.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=3
        )
        
        # Plot 4: Heatmap - p-values
        def pval(g1, g2):
            if g1 in group_data and g2 in group_data:
                x1, x2 = group_data[g1], group_data[g2]
                if len(x1) < 2 or len(x2) < 2:
                    return np.nan
                try:
                    _, p = ttest_ind(x1, x2, equal_var=False, alternative=alternative)
                    return p
                except:
                    return np.nan
            return np.nan

        pval_mat = _pairwise_matrix(groups, pval)
        pval_mat_clipped = pval_mat.clip(0, 1)
        
        fig.add_trace(
            go.Heatmap(
                z=pval_mat_clipped.values,
                x=pval_mat_clipped.columns,
                y=pval_mat_clipped.index,
                colorscale='Viridis',
                text=pval_mat.values,
                texttemplate='%{text:.4f}',
                textfont={"size": 10},
                zmin=0,
                zmax=1,
                showscale=False,
                showlegend=False
            ),
            row=row_idx, col=4
        )
        
        row_idx += 1
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#1f77b4')
        ),
        height=fig_height,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes
    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Group", row=i, col=1)
        fig.update_yaxes(title_text="Mean", row=i, col=1)
        for j in [2, 3, 4]:
            fig.update_xaxes(title_text="Group", row=i, col=j)
            fig.update_yaxes(title_text="Group", row=i, col=j)
    
    return fig


# Cached version for Streamlit (Issue #4 fix)
if STREAMLIT_AVAILABLE:
    @st.cache_data(show_spinner=False)
    def _create_basic_analysis_plotly_cached(
        df: pd.DataFrame,
        value_cols: tuple,
        group_col: str,
        plot_title: str,
        alternative: str
    ) -> go.Figure:
        """Cached version that Streamlit can use."""
        return _create_basic_analysis_plotly_impl(df, list(value_cols), group_col, plot_title, alternative)


def create_basic_analysis_plotly(
    df: pd.DataFrame,
    value_columns: List[str],
    group_column: str,
    title: str = "Treatment Effect Analysis",
    alternative: str = "two-sided",
    _use_cache: bool = True
) -> go.Figure:
    """
    Create Plotly visualization for basic treatment effect analysis.
    Shows means, uplifts, SMD, and p-values for each metric.
    
    Args:
        df: DataFrame with group assignments and metrics
        value_columns: List of numeric columns to analyze
        group_column: Name of the column containing group assignments
        title: Overall title for the plot
        _use_cache: Whether to use Streamlit caching (default: True)
        
    Returns:
        Plotly Figure object
    """
    # Use caching if Streamlit is available and caching is enabled
    if STREAMLIT_AVAILABLE and _use_cache:
        return _create_basic_analysis_plotly_cached(
            df,
            tuple(value_columns),
            group_column,
            title,
            alternative
        )
    else:
        # No caching, call directly
        return _create_basic_analysis_plotly_impl(df, value_columns, group_column, title, alternative)


def _create_cuped_analysis_plotly_impl(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    pre_suffixes: Tuple[str, ...] = ("_pre",),
    post_suffix: str = "_post",
    title: str = "CUPED-Adjusted Treatment Effect Analysis",
    alternative: str = "two-sided"
) -> go.Figure:
    """
    Create Plotly visualization for CUPED-adjusted analysis.

    Single-pre (len(pre_suffixes) == 1): 4-column layout
        Mean | Uplift % | SMD | p-value
    Multi-pre (len(pre_suffixes) > 1): 5-column layout
        Mean | Uplift % | SMD | p-value | Coefficients (R²)
    """
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)

    if n_groups < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 groups for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig

    multi_pre = len(pre_suffixes) > 1
    n_cols = 5 if multi_pre else 4

    # First pass: compute adjusted values, R², theta, group stats per metric
    metric_data: List[Dict] = []
    for metric in base_metrics:
        pre_cols = [metric + s for s in pre_suffixes]
        post_col = metric + post_suffix
        if post_col not in df.columns or not all(c in df.columns for c in pre_cols):
            continue

        if multi_pre:
            result = _cuped_adjust_multi(df[pre_cols], df[post_col])
            if result is None:
                continue
            adj, r2, theta = result
        else:
            adj = _cuped_adjust(df[pre_cols[0]], df[post_col])
            if adj is None:
                continue
            r2, theta = None, None

        group_indices = {g: df.index[df[group_column] == g] for g in groups}
        cuped_means: Dict = {}
        cuped_data: Dict = {}
        for g in groups:
            data_array = adj.loc[group_indices[g]].dropna().values
            if len(data_array) >= 2:
                cuped_means[g] = float(np.mean(data_array))
                cuped_data[g] = data_array

        if len(cuped_means) < 2:
            continue

        metric_data.append({
            'metric': metric, 'r2': r2, 'theta': theta,
            'cuped_means': cuped_means, 'cuped_data': cuped_data,
        })

    if not metric_data:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid metrics found. Expected columns with pre suffixes {list(pre_suffixes)} and post '{post_suffix}'",
            xref="paper", yref="paper", x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title=title, height=400)
        return fig

    # Build per-cell subplot titles (R² embedded in the coefficient column when multi-pre)
    subtitles: List[str] = []
    for md in metric_data:
        cols = ["CUPED Mean", "Uplift (%)", "CUPED SMD", "CUPED p-value"]
        if multi_pre:
            r2 = md['r2']
            r2_str = f"R²={r2:.3f}" if r2 is not None else "R²=?"
            cols.append(f"Coefficients ({r2_str})")
        subtitles.extend([f"{md['metric']}: {sub}" for sub in cols])

    n_rows = len(metric_data)
    vspace, fig_height = _vspace_and_height(n_rows)
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subtitles,
        vertical_spacing=vspace,
        horizontal_spacing=0.07,
    )

    for row_idx, md in enumerate(metric_data, start=1):
        metric = md['metric']
        cuped_means = md['cuped_means']
        cuped_data = md['cuped_data']

        # Col 1: bar of CUPED means
        fig.add_trace(
            go.Bar(
                x=list(cuped_means.keys()),
                y=list(cuped_means.values()),
                name=f"{metric} CUPED Mean",
                marker_color='green',
                text=list(cuped_means.values()),
                texttemplate='%{text:.3f}',
                textposition='outside',
                cliponaxis=False,
                showlegend=False,
            ),
            row=row_idx, col=1,
        )

        # Col 2: Uplift % heatmap
        def cuped_uplift(g1, g2, cm=cuped_means):
            if g1 in cm and g2 in cm and cm[g1] != 0:
                return ((cm[g2] - cm[g1]) / cm[g1] * 100)
            return np.nan

        uplift_mat = _pairwise_matrix(groups, cuped_uplift)
        fig.add_trace(
            go.Heatmap(
                z=uplift_mat.values, x=uplift_mat.columns, y=uplift_mat.index,
                colorscale='RdBu_r', text=uplift_mat.values, texttemplate='%{text:.1f}%',
                textfont={"size": 10}, showscale=False, showlegend=False,
            ),
            row=row_idx, col=2,
        )

        # Col 3: SMD heatmap
        def cuped_smd(g1, g2, cd=cuped_data):
            if g1 in cd and g2 in cd:
                return _smd(cd[g1], cd[g2])
            return np.nan

        smd_mat = _pairwise_matrix(groups, cuped_smd)
        fig.add_trace(
            go.Heatmap(
                z=smd_mat.values, x=smd_mat.columns, y=smd_mat.index,
                colorscale='Reds', text=smd_mat.values, texttemplate='%{text:.3f}',
                textfont={"size": 10}, showscale=False, showlegend=False,
            ),
            row=row_idx, col=3,
        )

        # Col 4: p-value heatmap
        def cuped_pval(g1, g2, cd=cuped_data, alt=alternative):
            if g1 in cd and g2 in cd:
                y1, y2 = cd[g1], cd[g2]
                if len(y1) < 2 or len(y2) < 2:
                    return np.nan
                try:
                    _, p = ttest_ind(y1, y2, equal_var=False, alternative=alt)
                    return p
                except:
                    return np.nan
            return np.nan

        pval_mat = _pairwise_matrix(groups, cuped_pval)
        pval_mat_clipped = pval_mat.clip(0, 1)
        fig.add_trace(
            go.Heatmap(
                z=pval_mat_clipped.values, x=pval_mat_clipped.columns, y=pval_mat_clipped.index,
                colorscale='Viridis', text=pval_mat.values, texttemplate='%{text:.4f}',
                textfont={"size": 10}, zmin=0, zmax=1, showscale=False, showlegend=False,
            ),
            row=row_idx, col=4,
        )

        # Col 5: horizontal coefficient bar (multi-pre only)
        if multi_pre and md['theta'] is not None:
            theta = md['theta']
            # Sort by |θ| desc; reversed y-axis below places largest |θ| at the top.
            sorted_items = sorted(theta.items(), key=lambda kv: abs(kv[1]), reverse=True)
            labels = [k for k, _ in sorted_items]
            values = [v for _, v in sorted_items]
            colors = ['#1f77b4' if v >= 0 else '#d62728' for v in values]
            fig.add_trace(
                go.Bar(
                    x=values, y=labels, orientation='h',
                    marker_color=colors,
                    text=values, texttemplate='%{text:.3f}',
                    textposition='outside', cliponaxis=False, showlegend=False,
                    name=f"{metric} θ",
                ),
                row=row_idx, col=5,
            )
            fig.update_yaxes(autorange='reversed', row=row_idx, col=5)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20, color='#1f77b4')),
        height=fig_height,
        showlegend=False,
        template='plotly_white',
    )

    for i in range(1, n_rows + 1):
        fig.update_xaxes(title_text="Group", row=i, col=1)
        fig.update_yaxes(title_text="CUPED Mean", row=i, col=1)
        for j in [2, 3, 4]:
            fig.update_xaxes(title_text="Group", row=i, col=j)
            fig.update_yaxes(title_text="Group", row=i, col=j)
        if multi_pre:
            fig.update_xaxes(title_text="θ (coefficient)", row=i, col=5)
            fig.update_yaxes(title_text="Pre-period suffix", row=i, col=5)

    return fig


# Cached version for Streamlit (Issue #4 fix)
if STREAMLIT_AVAILABLE:
    @st.cache_data(show_spinner=False)
    def _create_cuped_analysis_plotly_cached(
        df: pd.DataFrame,
        base_metrics: tuple,
        group_col: str,
        pre_suffixes: tuple,
        post_suffix: str,
        plot_title: str,
        alternative: str
    ) -> go.Figure:
        """Cached version that Streamlit can use."""
        return _create_cuped_analysis_plotly_impl(
            df, list(base_metrics), group_col, pre_suffixes, post_suffix, plot_title, alternative
        )


def create_cuped_analysis_plotly(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    pre_suffixes: Tuple[str, ...] = ("_pre",),
    post_suffix: str = "_post",
    title: str = "CUPED-Adjusted Treatment Effect Analysis",
    alternative: str = "two-sided",
    _use_cache: bool = True
) -> go.Figure:
    """
    Create Plotly visualization for CUPED-adjusted analysis.

    Args:
        df: DataFrame with group assignments and pre/post metrics.
        base_metrics: List of base metric names (without suffixes).
        group_column: Name of the column containing group assignments.
        pre_suffixes: Tuple of pre-period suffixes. If length > 1, the plot
            switches to multi-pre layout (5 cols) and uses multivariate OLS CUPED.
        post_suffix: Suffix for the post-experiment column.
        title: Overall title for the plot.
        alternative: scipy ttest alternative ('two-sided' | 'greater' | 'less').
        _use_cache: Whether to use Streamlit caching (default: True).

    Returns:
        Plotly Figure object.
    """
    if STREAMLIT_AVAILABLE and _use_cache:
        return _create_cuped_analysis_plotly_cached(
            df, tuple(base_metrics), group_column,
            tuple(pre_suffixes), post_suffix, title, alternative,
        )
    return _create_cuped_analysis_plotly_impl(
        df, base_metrics, group_column, pre_suffixes, post_suffix, title, alternative,
    )


def _create_did_analysis_plotly_impl(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    pre_suffixes: Tuple[str, ...] = ("_pre",),
    post_suffix: str = "_post",
    title: str = "Difference-in-Differences Analysis",
) -> go.Figure:
    """
    Create Plotly visualization for DiD analysis.

    Single-pre (len(pre_suffixes) == 1): 4-col layout
        Pre Mean (bar) | Post Mean (bar) | % Change | Δ DiD Gap heatmap
    Multi-pre (len(pre_suffixes) > 1): 4-col layout, col 1 becomes a
        Parallel-Trends line plot (one line per group across all periods).
        % Change and DiD Gap use the row-wise mean of pre-period columns
        as the baseline.
    """
    groups = sorted(df[group_column].unique())
    n_groups = len(groups)

    if n_groups < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 groups for analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig

    multi_pre = len(pre_suffixes) > 1

    valid_metrics: List[str] = []
    for metric in base_metrics:
        pre_cols = [metric + s for s in pre_suffixes]
        post_col = metric + post_suffix
        if post_col in df.columns and all(c in df.columns for c in pre_cols):
            valid_metrics.append(metric)

    if not valid_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No valid metrics found. Expected columns with pre suffixes {list(pre_suffixes)} and post '{post_suffix}'",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=14)
        )
        fig.update_layout(title=title, height=400)
        return fig

    # Subplot titles depend on layout
    if multi_pre:
        col_subs = ["Parallel Trends", "Post-Period Mean", "% Change (Post − Avg Pre)", "Δ DiD Gap"]
    else:
        col_subs = ["Pre-Period Mean", "Post-Period Mean", "% Change", "Δ DiD Gap"]

    n_rows = len(valid_metrics)
    vspace, fig_height = _vspace_and_height(n_rows)
    fig = make_subplots(
        rows=n_rows, cols=4,
        subplot_titles=[f"{metric}: {sub}" for metric in valid_metrics for sub in col_subs],
        vertical_spacing=vspace,
        horizontal_spacing=0.08,
    )

    # Stable palette per group (used by the parallel-trends line)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    period_labels = list(pre_suffixes) + [post_suffix]

    for row_idx, metric in enumerate(valid_metrics, start=1):
        pre_cols = [metric + s for s in pre_suffixes]
        post_col = metric + post_suffix

        group_indices = {g: df.index[df[group_column] == g] for g in groups}

        per_period_means: Dict = {}
        post_means: Dict = {}
        pre_means_avg: Dict = {}
        for g in groups:
            idx = group_indices[g]
            means = [df.loc[idx, c].mean() for c in pre_cols + [post_col]]
            per_period_means[g] = means
            post_means[g] = means[-1]
            # Average across pre-period values; with single pre this is just that value
            pre_means_avg[g] = float(np.nanmean(means[:-1]))

        pct_changes: Dict = {}
        for g in groups:
            base = pre_means_avg[g]
            pct_changes[g] = ((post_means[g] - base) / base) * 100 if base != 0 else 0.0

        # Column 1: parallel-trends line (multi-pre) OR pre-period mean bar (single-pre)
        if multi_pre:
            for g in groups:
                fig.add_trace(
                    go.Scatter(
                        x=period_labels,
                        y=per_period_means[g],
                        mode='lines+markers',
                        name=str(g),
                        marker=dict(color=group_color[g], size=8),
                        line=dict(color=group_color[g], width=2),
                        legendgroup=str(g),
                        showlegend=(row_idx == 1),
                    ),
                    row=row_idx, col=1,
                )
        else:
            pre_only_means = {g: per_period_means[g][0] for g in groups}
            fig.add_trace(
                go.Bar(
                    x=list(pre_only_means.keys()),
                    y=list(pre_only_means.values()),
                    name=f"{metric} Pre",
                    marker_color='lightblue',
                    text=list(pre_only_means.values()),
                    texttemplate='%{text:.3f}',
                    textposition='outside',
                    cliponaxis=False,
                    showlegend=False,
                ),
                row=row_idx, col=1,
            )

        # Column 2: post-period means (bar)
        fig.add_trace(
            go.Bar(
                x=list(post_means.keys()),
                y=list(post_means.values()),
                name=f"{metric} Post",
                marker_color='lightgreen',
                text=list(post_means.values()),
                texttemplate='%{text:.3f}',
                textposition='outside',
                cliponaxis=False,
                showlegend=False,
            ),
            row=row_idx, col=2,
        )

        # Column 3: % change vs (averaged) pre baseline
        fig.add_trace(
            go.Bar(
                x=list(pct_changes.keys()),
                y=list(pct_changes.values()),
                name=f"{metric} % Change",
                marker_color='orange',
                text=[f"{v:+.1f}%" for v in pct_changes.values()],
                textposition='outside',
                cliponaxis=False,
                showlegend=False,
            ),
            row=row_idx, col=3,
        )

        # Column 4: DiD gap heatmap using averaged pre baseline
        def did_gap(g1, g2, pma=pre_means_avg, pm=post_means):
            d_pre = pma[g2] - pma[g1]
            d_post = pm[g2] - pm[g1]
            return d_post - d_pre

        did_mat = _pairwise_matrix(groups, did_gap)
        fig.add_trace(
            go.Heatmap(
                z=did_mat.values,
                x=did_mat.columns,
                y=did_mat.index,
                colorscale='RdBu_r',
                text=did_mat.values,
                texttemplate='%{text:.3f}',
                textfont={"size": 10},
                showscale=(row_idx == 1),
                showlegend=False,
            ),
            row=row_idx, col=4,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20, color='#1f77b4')),
        height=fig_height,
        showlegend=multi_pre,  # only useful when parallel-trends line is shown
        template='plotly_white',
    )

    for i in range(1, n_rows + 1):
        if multi_pre:
            fig.update_xaxes(title_text="Period", row=i, col=1)
            fig.update_yaxes(title_text="Mean", row=i, col=1)
        else:
            fig.update_xaxes(title_text="Group", row=i, col=1)
            fig.update_yaxes(title_text="Pre-Period Mean", row=i, col=1)

        fig.update_xaxes(title_text="Group", row=i, col=2)
        fig.update_yaxes(title_text="Post-Period Mean", row=i, col=2)

        fig.update_xaxes(title_text="Group", row=i, col=3)
        fig.update_yaxes(title_text="% Change", row=i, col=3)

        fig.update_xaxes(title_text="Group", row=i, col=4)
        fig.update_yaxes(title_text="Group", row=i, col=4)

    return fig


# Cached version for Streamlit (Issue #4 fix)
if STREAMLIT_AVAILABLE:
    @st.cache_data(show_spinner=False)
    def _create_did_analysis_plotly_cached(
        df: pd.DataFrame,
        base_metrics: tuple,
        group_col: str,
        pre_suffixes: tuple,
        post_suffix: str,
        plot_title: str,
    ) -> go.Figure:
        """Cached version that Streamlit can use."""
        return _create_did_analysis_plotly_impl(
            df, list(base_metrics), group_col, pre_suffixes, post_suffix, plot_title,
        )


def create_did_analysis_plotly(
    df: pd.DataFrame,
    base_metrics: List[str],
    group_column: str,
    pre_suffixes: Tuple[str, ...] = ("_pre",),
    post_suffix: str = "_post",
    title: str = "Difference-in-Differences Analysis",
    _use_cache: bool = True
) -> go.Figure:
    """
    Create Plotly visualization for DiD analysis.

    Args:
        df: DataFrame with group assignments and pre/post metrics.
        base_metrics: List of base metric names (without suffixes).
        group_column: Name of the column containing group assignments.
        pre_suffixes: Tuple of pre-period suffixes. Length > 1 switches col 1
            to a parallel-trends line plot and uses the row-wise mean of pre
            columns as the DiD baseline.
        post_suffix: Suffix for the post-experiment column.
        title: Overall title for the plot.
        _use_cache: Whether to use Streamlit caching (default: True).

    Returns:
        Plotly Figure object.
    """
    if STREAMLIT_AVAILABLE and _use_cache:
        return _create_did_analysis_plotly_cached(
            df, tuple(base_metrics), group_column,
            tuple(pre_suffixes), post_suffix, title,
        )
    return _create_did_analysis_plotly_impl(
        df, base_metrics, group_column, pre_suffixes, post_suffix, title,
    )
