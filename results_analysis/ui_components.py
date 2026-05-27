"""
UI components for results analysis page
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from results_analysis.plots import (
    create_basic_analysis_plotly,
    create_cuped_analysis_plotly,
    create_did_analysis_plotly
)
from results_analysis.tooltips import PARAMETER_TOOLTIPS
from utils.streamlit_downloads import render_download_button, render_plot_with_download
from scipy.stats import ttest_ind
from utils.artifact_builder import ArtifactBuilder


_SCIPY_ALTERNATIVE = {"two-sided": "two-sided", "larger": "greater", "smaller": "less"}


def _get_ttest_alternative() -> str:
    """Return the scipy `alternative` value based on the user's selection."""
    label = st.session_state.get("results_ttest_alternative", "two-sided")
    return _SCIPY_ALTERNATIVE.get(label, "two-sided")


def _init_pre_period_state() -> None:
    """Seed default session-state entries for shared pre/post suffix config."""
    if "results_suffix_post" not in st.session_state:
        st.session_state["results_suffix_post"] = "_post"
    if "results_pre_slots" not in st.session_state:
        st.session_state["results_pre_slots"] = [0]
        st.session_state["results_pre_suffix_0"] = "_pre"
        st.session_state["results_next_pre_slot"] = 1


def _get_post_suffix() -> str:
    return st.session_state.get("results_suffix_post", "_post") or "_post"


def _get_pre_suffixes() -> list:
    """Active pre-suffixes; honours the multi-pre toggle."""
    slots = st.session_state.get("results_pre_slots", [0])
    multi = st.session_state.get("results_multi_pre_enabled", False)
    active = slots if multi else slots[:1]
    out = [st.session_state.get(f"results_pre_suffix_{s}", "_pre") for s in active]
    out = [s for s in out if s]
    return out or ["_pre"]


def _is_multi_pre() -> bool:
    return bool(st.session_state.get("results_multi_pre_enabled", False)) and len(_get_pre_suffixes()) > 1


from utils.stats import smd as _smd, cuped_adjust as _cuped_adjust, cuped_adjust_multi as _cuped_adjust_multi
from utils.streamlit_upload import render_csv_upload_with_dummy
from utils.streamlit_validation import validate_data_and_group
from utils.streamlit_errors import handle_plot_error

def render_data_upload():
    """Render the data upload section"""
    # Initialize artifact builder
    if "results_analysis_artifact" not in st.session_state:
        st.session_state.results_analysis_artifact = ArtifactBuilder(page_name="results_analysis")

    render_csv_upload_with_dummy(
        header="📤 Upload Experiment Results Data",
        description="Upload a CSV file containing your experiment results with group assignments and metric values",
        data_state_key="results_uploaded_data",
        filename_state_key="results_filename",
        uploader_label="Or choose a CSV file",
        uploader_key="results_file_upload",
        uploader_help="Upload a CSV file with experiment results",
        # Required legacy positional kwargs (ignored when dummy_datasets is set)
        dummy_file_path=None,
        dummy_button_key=None,
        dummy_loaded_filename=None,
        dummy_datasets=[
            {
                "file_path": os.path.join("dummy_data", "results_analysis_dummy.csv"),
                "button_key": "results_load_dummy",
                "button_label": "🎲 Single pre-period",
                "loaded_filename": "dummy_results_data.csv",
                "heading": "Standard",
                "caption": "Single pre-period + post. Good for Basic / CUPED / DiD with flat effect.",
            },
            {
                "file_path": os.path.join("dummy_data", "results_analysis_multi_pre_dummy.csv"),
                "button_key": "results_load_dummy_multi_pre",
                "button_label": "🎲 Multi pre-period",
                "loaded_filename": "dummy_results_data_multi_pre.csv",
                "heading": "Multi pre-period",
                "caption": "Three pre-windows (`_pre_w1/_pre_w2/_pre_w3`) for CUPED+ and parallel-trends DiD.",
            },
            {
                "file_path": os.path.join("dummy_data", "results_analysis_heterogeneous_dummy.csv"),
                "button_key": "results_load_dummy_heterogeneous",
                "button_label": "🎲 Heterogeneous effect",
                "loaded_filename": "dummy_results_data_heterogeneous.csv",
                "heading": "Heterogeneous",
                "caption": "Real per-user variation (engagement_tier × region × tenure). Best for 🧬 Heterogeneity Analysis.",
            },
        ],
        artifact=st.session_state.results_analysis_artifact,
        artifact_df_name="uploaded_data",
        artifact_df_description="Original uploaded data",
        artifact_log_category="data_upload",
        show_overview=True,
    )


def render_configuration():
    """Render the configuration section"""
    if st.session_state.get('results_uploaded_data') is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' tab")
        return
    
    df = st.session_state.results_uploaded_data.copy()
    
    st.header("⚙️ Configuration")
    
    # Group column selection
    st.subheader("📋 Select Group Column")
    group_column = st.selectbox(
        "Group Column",
        options=[""] + df.columns.tolist(),
        index=0,  # Default to empty
        key="results_group_column",
        help=PARAMETER_TOOLTIPS.get("group_column", "")
    )
    
    if group_column and group_column != "":
        groups = sorted(df[group_column].unique())
        n_groups = len(groups)
        st.info(f"📊 Found {n_groups} groups: {', '.join(map(str, groups))}")
        
        # Show group sizes
        group_sizes = df[group_column].value_counts()
        st.write("**Group Sizes:**")
        cols = st.columns(min(n_groups, 4))
        for idx, group_name in enumerate(groups):
            with cols[idx % len(cols)]:
                size = group_sizes.get(group_name, 0)
                st.metric(str(group_name), f"{size:,}", f"{size/len(df)*100:.1f}%")
        
        # Store groups list (group_column is already stored by the widget with key="results_group_column")
        st.session_state.results_groups = groups

        st.divider()

        # Pre/Post Period Configuration (shared by CUPED & DiD)
        _init_pre_period_state()
        st.subheader("📅 Pre/Post Period Configuration")
        st.caption("Suffixes shared by CUPED and DiD analyses. Default: `_pre` / `_post`.")

        col_post, col_pre = st.columns(2)
        slots = st.session_state["results_pre_slots"]
        with col_post:
            st.text_input(
                "Post-period suffix",
                key="results_suffix_post",
                help=PARAMETER_TOOLTIPS.get("results_suffix_post", "")
            )
        with col_pre:
            st.text_input(
                "Pre-period suffix #1",
                key=f"results_pre_suffix_{slots[0]}",
                help=PARAMETER_TOOLTIPS.get("results_suffix_pre", "")
            )

        st.checkbox(
            "My data has multiple pre-periods (e.g. several pre-experiment windows for seasonality)",
            key="results_multi_pre_enabled",
            help=PARAMETER_TOOLTIPS.get("results_multi_pre_enabled", "")
        )

        if st.session_state.get("results_multi_pre_enabled"):
            st.caption(
                "Each additional suffix must match real columns in your data. "
                "For metric `revenue` with suffix `_pre_w2`, the tool expects column `revenue_pre_w2`."
            )

            for i, slot in enumerate(slots[1:], start=2):
                key = f"results_pre_suffix_{slot}"
                if key not in st.session_state:
                    st.session_state[key] = "_pre"
                col_text, col_btn = st.columns([6, 1])
                with col_text:
                    st.text_input(f"Pre-period suffix #{i}", key=key)
                with col_btn:
                    st.write("")
                    st.write("")
                    if st.button("✕", key=f"results_remove_pre_{slot}", help="Remove this pre-period"):
                        st.session_state["results_pre_slots"].remove(slot)
                        st.session_state.pop(key, None)
                        st.rerun()

            col_add, _ = st.columns([2, 5])
            with col_add:
                if st.button("➕ Add another pre-period", key="results_add_pre"):
                    new_slot = st.session_state["results_next_pre_slot"]
                    st.session_state["results_next_pre_slot"] += 1
                    st.session_state[f"results_pre_suffix_{new_slot}"] = "_pre"
                    st.session_state["results_pre_slots"].append(new_slot)
                    st.rerun()

        active_pre_suffixes = _get_pre_suffixes()
        active_post_suffix = _get_post_suffix()
        st.info(
            f"Active pre-suffixes: `{', '.join(active_pre_suffixes)}` · post-suffix: `{active_post_suffix}`"
        )

        st.divider()

        # T-Test Type (optional, default two-sided)
        st.subheader("🧪 T-Test Type (optional)")
        ttest_alternative = st.selectbox(
            "Alternative hypothesis",
            options=["two-sided", "larger", "smaller"],
            index=0,
            key="results_ttest_alternative",
            help=PARAMETER_TOOLTIPS.get("ttest_alternative", "")
        )
        st.caption("Applies to p-values in Basic, CUPED, and DiD analyses. Default: two-sided.")

        # Add configuration to artifact
        artifact = st.session_state.get('results_analysis_artifact')
        if artifact:
            artifact.set_config({
                'group_column': group_column,
                'n_groups': n_groups,
                'group_names': groups,
                'ttest_alternative': ttest_alternative,
                'pre_suffixes': active_pre_suffixes,
                'post_suffix': active_post_suffix,
                'multi_pre_enabled': _is_multi_pre(),
            })
            artifact.add_log(
                category='configuration',
                message=f'Group column selected: {group_column} with {n_groups} groups',
                details={
                    'group_column': group_column,
                    'groups': groups,
                    'group_sizes': group_sizes.to_dict(),
                    'ttest_alternative': ttest_alternative,
                    'pre_suffixes': active_pre_suffixes,
                    'post_suffix': active_post_suffix,
                    'multi_pre_enabled': _is_multi_pre(),
                }
            )


def render_basic_analysis():
    """Render basic treatment effect analysis"""
    st.header("📊 Basic Treatment Effect Analysis")
    st.markdown("Analyze treatment effects, uplifts, and statistical significance")
    
    # Check if data is available
    is_valid, df, group_column = validate_data_and_group(
        'results_uploaded_data',
        'results_group_column'
    )
    if not is_valid:
        return
    
    groups = st.session_state.results_groups
    
    st.divider()
    
    # Column selection
    st.subheader("📊 Select Metrics to Analyze")
    
    from utils.data_filtering import is_id_column
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out group column and ID columns only.
    # Pre/post columns are intentionally kept: spotting an uplift on a pre-period
    # column is a useful balance diagnostic that prompts using CUPED/DiD.
    base_numeric_cols = [col for col in numeric_cols if col != group_column and not is_id_column(df, col)]
    
    if not base_numeric_cols:
        st.error("❌ No numeric columns found for analysis")
        return
    
    value_columns = st.multiselect(
        "Metric Columns",
        options=base_numeric_cols,
        default=base_numeric_cols,
        key="results_basic_metrics",
        help=PARAMETER_TOOLTIPS.get("metric_columns", "")
    )
    
    if not value_columns:
        st.info("ℹ️ Please select at least one metric column")
        return
    
    st.divider()
    
    # Always generate the plot for artifact (regardless of view mode)
    try:
        fig = create_basic_analysis_plotly(
            df,
            value_columns,
            group_column,
            title="Basic Treatment Effect Analysis",
            alternative=_get_ttest_alternative()
        )
        
        # Add plot to artifact
        artifact = st.session_state.get('results_analysis_artifact')
        if artifact:
            artifact.add_plot('basic_analysis', fig, 'Basic treatment effect analysis')
            artifact.add_log(
                category='analysis',
                message='Basic analysis performed',
                details={
                    'metrics': value_columns,
                    'group_column': group_column
                }
            )
    except Exception as e:
        handle_plot_error('basic analysis plot', e)
        fig = None
    
    # View switcher
    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="basic_analysis_view_mode"
    )
    
    if view_mode == "Summary":
        # Summary tables
        st.subheader("📋 Treatment Effect Summary")
        
        summary_data = []
        for col in value_columns:
            group_data = {}
            group_means = {}
            
            for g in groups:
                data = df[df[group_column] == g][col].dropna()
                if len(data) >= 2:
                    group_data[g] = data
                    group_means[g] = data.mean()
            
            if len(group_means) < 2:
                continue
            
            # Calculate pairwise statistics
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    if g1 in group_data and g2 in group_data:
                        x1, x2 = group_data[g1], group_data[g2]
                        m1, m2 = group_means[g1], group_means[g2]
                        
                        # Uplift
                        uplift_pct = ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan
                        
                        # SMD
                        smd_val = _smd(x1, x2)
                        
                        # p-value
                        try:
                            _, p = ttest_ind(x1, x2, equal_var=False, alternative=_get_ttest_alternative())
                        except:
                            p = np.nan
                        
                        summary_data.append({
                            'Metric': col,
                            'Group 1': str(g1),
                            'Group 2': str(g2),
                            'Mean G1': f"{m1:.3f}",
                            'Mean G2': f"{m2:.3f}",
                            'Uplift (%)': f"{uplift_pct:+.2f}" if np.isfinite(uplift_pct) else "N/A",
                            'SMD': f"{smd_val:.3f}" if np.isfinite(smd_val) else "N/A",
                            'p-value': f"{p:.4f}" if np.isfinite(p) else "N/A",
                            'Significant': "Yes" if np.isfinite(p) and p < 0.05 else "No"
                        })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Summary as CSV",
                data=csv,
                file_name="basic_analysis_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No valid data for summary")
    
    else:
        # Visual report - display the plot that was already generated
        st.subheader("📈 Treatment Effect Visualizations")
        
        if fig is not None:
            render_plot_with_download(
                fig,
                "basic_analysis_report.html",
                "Download interactive treatment effect analysis report"
            )
        else:
            st.error("Could not display plot. Please check the error message above.")


def render_cuped_analysis():
    """Render CUPED-adjusted analysis"""
    st.header("🔬 CUPED-Adjusted Analysis")
    st.markdown("Analyze treatment effects using CUPED (Controlled-experiment Using Pre-Experiment Data) adjustment")

    is_valid, df, group_column = validate_data_and_group(
        'results_uploaded_data',
        'results_group_column'
    )
    if not is_valid:
        return

    groups = st.session_state.results_groups
    pre_suffixes = _get_pre_suffixes()
    post_suffix = _get_post_suffix()
    multi_pre = _is_multi_pre()

    st.divider()

    st.caption(
        f"ℹ️ Pre/post suffixes are configured in the **⚙️ Configuration** tab. "
        f"Active pre: `{', '.join(pre_suffixes)}` · post: `{post_suffix}` "
        f"({'multivariate OLS CUPED' if multi_pre else 'single-pre CUPED'})."
    )

    # Find metrics that have the post column AND every pre column
    all_cols = df.columns.tolist()
    from utils.data_filtering import is_id_column

    base_metrics = []
    for col in all_cols:
        if col.endswith(post_suffix):
            base = col[:-len(post_suffix)]
            needed_pre = [base + s for s in pre_suffixes]
            if all(c in all_cols for c in needed_pre):
                if not any(is_id_column(df, c) for c in needed_pre + [col]):
                    base_metrics.append(base)

    if not base_metrics:
        st.error(
            f"❌ No metrics found that have post suffix '{post_suffix}' and all required pre suffixes "
            f"{pre_suffixes}."
        )
        st.info(
            f"ℹ️ Looking for columns like: " +
            ", ".join([f"`metric{s}`" for s in pre_suffixes + [post_suffix]])
        )
        return

    st.info(f"✅ Found {len(base_metrics)} metrics with CUPED data: {', '.join(base_metrics[:5])}{'...' if len(base_metrics) > 5 else ''}")

    selected_metrics = st.multiselect(
        "Select Metrics for CUPED Analysis",
        options=base_metrics,
        default=base_metrics,
        key="cuped_metrics",
        help=PARAMETER_TOOLTIPS.get("cuped_metrics", "")
    )

    if not selected_metrics:
        st.info("ℹ️ Please select at least one metric")
        return

    st.divider()

    # Always generate the plot for artifact (regardless of view mode)
    try:
        fig = create_cuped_analysis_plotly(
            df,
            selected_metrics,
            group_column,
            pre_suffixes=tuple(pre_suffixes),
            post_suffix=post_suffix,
            title="CUPED-Adjusted Treatment Effect Analysis",
            alternative=_get_ttest_alternative()
        )

        artifact = st.session_state.get('results_analysis_artifact')
        if artifact:
            artifact.add_plot('cuped_analysis', fig, 'CUPED-adjusted treatment effect analysis')
            artifact.add_log(
                category='analysis',
                message='CUPED analysis performed',
                details={
                    'metrics': selected_metrics,
                    'group_column': group_column,
                    'pre_suffixes': pre_suffixes,
                    'post_suffix': post_suffix,
                    'multi_pre': multi_pre,
                }
            )
    except Exception as e:
        handle_plot_error('CUPED analysis plot', e)
        fig = None

    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="cuped_analysis_view_mode"
    )

    if view_mode == "Summary":
        st.subheader("📋 CUPED-Adjusted Treatment Effect Summary")
        if multi_pre:
            st.caption("Multi-pre mode: each metric is adjusted via OLS on all pre-period columns. **CUPED R²** column shows the fraction of post-period variance explained by the pre-periods.")

        summary_data = []
        for metric in selected_metrics:
            post_col = metric + post_suffix
            pre_cols = [metric + s for s in pre_suffixes]

            r2: float | None = None
            if multi_pre:
                result = _cuped_adjust_multi(df[pre_cols], df[post_col])
                if result is None:
                    continue
                adj, r2, _theta = result
            else:
                adj = _cuped_adjust(df[pre_cols[0]], df[post_col])
                if adj is None:
                    continue

            cuped_data = {}
            cuped_means = {}
            for g in groups:
                data = adj[df[group_column] == g].dropna()
                if len(data) >= 2:
                    cuped_data[g] = data
                    cuped_means[g] = data.mean()

            if len(cuped_means) < 2:
                continue

            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    if g1 in cuped_data and g2 in cuped_data:
                        y1, y2 = cuped_data[g1], cuped_data[g2]
                        m1, m2 = cuped_means[g1], cuped_means[g2]

                        uplift_pct = ((m2 - m1) / m1 * 100) if m1 != 0 else np.nan
                        smd_val = _smd(y1, y2)

                        try:
                            _, p = ttest_ind(y1, y2, equal_var=False, alternative=_get_ttest_alternative())
                        except:
                            p = np.nan

                        row = {
                            'Metric': metric,
                            'Group 1': str(g1),
                            'Group 2': str(g2),
                            'CUPED Mean G1': f"{m1:.3f}",
                            'CUPED Mean G2': f"{m2:.3f}",
                            'Uplift (%)': f"{uplift_pct:+.2f}" if np.isfinite(uplift_pct) else "N/A",
                            'CUPED SMD': f"{smd_val:.3f}" if np.isfinite(smd_val) else "N/A",
                            'CUPED p-value': f"{p:.4f}" if np.isfinite(p) else "N/A",
                            'Significant': "Yes" if np.isfinite(p) and p < 0.05 else "No",
                        }
                        if multi_pre and r2 is not None:
                            row['CUPED R²'] = f"{r2:.3f}"
                        summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download CUPED Summary as CSV",
                data=csv,
                file_name="cuped_analysis_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No valid data for CUPED summary")

    else:
        st.subheader("📈 CUPED-Adjusted Visualizations")
        if fig is not None:
            render_plot_with_download(
                fig,
                "cuped_analysis_report.html",
                "Download interactive CUPED analysis report"
            )
        else:
            st.error("Could not display plot. Please check the error message above.")


def render_did_analysis():
    """Render Difference-in-Differences analysis"""
    st.header("📉 Difference-in-Differences Analysis")
    st.markdown("Analyze treatment effects using Difference-in-Differences (DiD) methodology")
    
    # Check if data is available
    is_valid, df, group_column = validate_data_and_group(
        'results_uploaded_data',
        'results_group_column'
    )
    if not is_valid:
        return
    
    groups = st.session_state.results_groups
    pre_suffixes = _get_pre_suffixes()
    post_suffix = _get_post_suffix()
    multi_pre = _is_multi_pre()

    st.divider()

    st.caption(
        f"ℹ️ Pre/post suffixes are configured in the **⚙️ Configuration** tab. "
        f"Active pre: `{', '.join(pre_suffixes)}` · post: `{post_suffix}` "
        f"({'multi-pre DiD (averaged baseline + parallel-trends plot)' if multi_pre else 'single-pre DiD'})."
    )

    all_cols = df.columns.tolist()
    from utils.data_filtering import is_id_column

    base_metrics = []
    for col in all_cols:
        if col.endswith(post_suffix):
            base = col[:-len(post_suffix)]
            needed_pre = [base + s for s in pre_suffixes]
            if all(c in all_cols for c in needed_pre):
                if not any(is_id_column(df, c) for c in needed_pre + [col]):
                    base_metrics.append(base)

    if not base_metrics:
        st.error(
            f"❌ No metrics found that have post suffix '{post_suffix}' and all required pre suffixes "
            f"{pre_suffixes}."
        )
        st.info(
            f"ℹ️ Looking for columns like: " +
            ", ".join([f"`metric{s}`" for s in pre_suffixes + [post_suffix]])
        )
        return

    st.info(f"✅ Found {len(base_metrics)} metrics with DiD data: {', '.join(base_metrics[:5])}{'...' if len(base_metrics) > 5 else ''}")

    selected_metrics = st.multiselect(
        "Select Metrics for DiD Analysis",
        options=base_metrics,
        default=base_metrics,
        key="did_metrics",
        help=PARAMETER_TOOLTIPS.get("did_metrics", "")
    )

    if not selected_metrics:
        st.info("ℹ️ Please select at least one metric")
        return

    st.divider()

    try:
        fig = create_did_analysis_plotly(
            df,
            selected_metrics,
            group_column,
            pre_suffixes=tuple(pre_suffixes),
            post_suffix=post_suffix,
            title="Difference-in-Differences Analysis"
        )

        artifact = st.session_state.get('results_analysis_artifact')
        if artifact:
            artifact.add_plot('did_analysis', fig, 'Difference-in-Differences analysis')
            artifact.add_log(
                category='analysis',
                message='DiD analysis performed',
                details={
                    'metrics': selected_metrics,
                    'group_column': group_column,
                    'pre_suffixes': pre_suffixes,
                    'post_suffix': post_suffix,
                    'multi_pre': multi_pre,
                }
            )
    except Exception as e:
        handle_plot_error('DiD analysis plot', e)
        fig = None

    view_mode = st.radio(
        "View Mode",
        options=["Summary", "Visual Report"],
        index=0,
        horizontal=True,
        key="did_analysis_view_mode"
    )

    if view_mode == "Summary":
        st.subheader("📋 DiD Gap Summary")
        if multi_pre:
            st.caption("Multi-pre mode: 'Pre' values are the **row-wise mean** across all pre-period columns. DiD gap and p-value use this averaged baseline.")

        summary_data = []
        for metric in selected_metrics:
            post_col = metric + post_suffix
            pre_cols = [metric + s for s in pre_suffixes]
            # Row-wise mean baseline (equals the single pre col when only one is active)
            baseline = df[pre_cols].mean(axis=1)

            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    g1mask = df[group_column] == g1
                    g2mask = df[group_column] == g2

                    g1_pre = baseline[g1mask].mean()
                    g1_post = df.loc[g1mask, post_col].mean()
                    g2_pre = baseline[g2mask].mean()
                    g2_post = df.loc[g2mask, post_col].mean()

                    diff_pre = g2_pre - g1_pre
                    diff_post = g2_post - g1_post
                    did_gap = diff_post - diff_pre

                    g1_change = (df.loc[g1mask, post_col] - baseline[g1mask]).dropna()
                    g2_change = (df.loc[g2mask, post_col] - baseline[g2mask]).dropna()

                    try:
                        if len(g1_change) > 1 and len(g2_change) > 1:
                            _, p = ttest_ind(g2_change, g1_change, equal_var=False, alternative=_get_ttest_alternative())
                        else:
                            p = np.nan
                    except:
                        p = np.nan

                    pre_label = "Pre (avg)" if multi_pre else "Pre"
                    summary_data.append({
                        'Metric': metric,
                        'Group 1': str(g1),
                        'Group 2': str(g2),
                        f'G1 {pre_label}': f"{g1_pre:.3f}",
                        'G1 Post': f"{g1_post:.3f}",
                        f'G2 {pre_label}': f"{g2_pre:.3f}",
                        'G2 Post': f"{g2_post:.3f}",
                        'Gap Pre (G2-G1)': f"{diff_pre:.3f}",
                        'Gap Post (G2-G1)': f"{diff_post:.3f}",
                        'Δ DiD Gap': f"{did_gap:.3f}",
                        'p-value': f"{p:.4f}" if np.isfinite(p) else "N/A",
                        'Significant': "Yes" if np.isfinite(p) and p < 0.05 else "No"
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download DiD Summary as CSV",
                data=csv,
                file_name="did_analysis_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No valid data for DiD summary")
    
    else:
        # Visual report - display the plot that was already generated
        st.subheader("📈 DiD Visualizations")
        
        if fig is not None:
            render_plot_with_download(
                fig,
                "did_analysis_report.html",
                "Download interactive DiD analysis report"
            )
        else:
            st.error("Could not display plot. Please check the error message above.")


# =============================================================================
# Heterogeneity Analysis (CATE / uplift modelling)
# =============================================================================

def _detect_cuped_candidates(df: pd.DataFrame, pre_suffixes, post_suffix):
    """Return base-metric names that have all required pre + post columns."""
    from utils.data_filtering import is_id_column
    cols = set(df.columns)
    out = []
    for col in df.columns:
        if not str(col).endswith(post_suffix):
            continue
        base = col[: -len(post_suffix)] if post_suffix else col
        needed = [base + s for s in pre_suffixes]
        if all(c in cols for c in needed) and not any(is_id_column(df, c) for c in needed + [col]):
            out.append(base)
    return out


def render_heterogeneity_setup():
    """Setup sub-tab for the Heterogeneity Analysis."""
    st.header("🧬 Heterogeneity Analysis — Setup")
    st.markdown(
        "Estimate **per-user treatment effects (CATE)** with a meta-learner. "
        "Tune features, learner, and base model — the Results sub-tab will show the CATE distribution, "
        "feature importance, Qini curve on a held-out sample, and subgroup drill-down."
    )

    is_valid, df, group_column = validate_data_and_group('results_uploaded_data', 'results_group_column')
    if not is_valid:
        return

    groups = st.session_state.results_groups
    if len(groups) < 2:
        st.warning("⚠️ Need at least 2 groups for heterogeneity analysis.")
        return

    from utils.data_filtering import is_id_column

    st.divider()
    st.subheader("🎯 Outcome")

    # Outcome column choice + optional CUPED-adjusted outcome
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_all = [c for c in numeric_cols_all if c != group_column and not is_id_column(df, c)]
    if not numeric_cols_all:
        st.error("❌ No numeric columns available as outcome.")
        return

    use_cuped = st.checkbox(
        "Use CUPED-adjusted outcome",
        value=False,
        key="het_use_cuped",
        help=PARAMETER_TOOLTIPS.get("het_use_cuped", ""),
    )

    if use_cuped:
        pre_suffixes = _get_pre_suffixes()
        post_suffix = _get_post_suffix()
        multi_pre = _is_multi_pre()
        cuped_candidates = _detect_cuped_candidates(df, pre_suffixes, post_suffix)
        if not cuped_candidates:
            st.error(
                f"❌ No metrics with post suffix '{post_suffix}' and all required pre suffixes {pre_suffixes} found. "
                f"Configure suffixes in the ⚙️ Configuration tab."
            )
            return
        base_metric = st.selectbox(
            "CUPED-adjusted metric",
            options=cuped_candidates,
            key="het_cuped_base_metric",
            help=PARAMETER_TOOLTIPS.get("het_cuped_base_metric", ""),
        )
        st.caption(
            f"Adjustment: {'multivariate OLS (CUPED+)' if multi_pre else 'single-pre CUPED'} "
            f"on pre suffixes `{', '.join(pre_suffixes)}` and post `{post_suffix}`."
        )
        outcome_label = f"{base_metric}{post_suffix} (CUPED-adjusted)"
    else:
        outcome_col = st.selectbox(
            "Outcome column",
            options=numeric_cols_all,
            key="het_outcome_col",
            help=PARAMETER_TOOLTIPS.get("het_outcome_col", ""),
        )
        outcome_label = outcome_col

    st.divider()
    st.subheader("🎛️ Treatment")

    col_ctrl, col_treat = st.columns(2)
    with col_ctrl:
        control_level = st.selectbox(
            "Control level",
            options=groups,
            index=0,
            key="het_control_level",
            help=PARAMETER_TOOLTIPS.get("het_control_level", ""),
        )
    with col_treat:
        # Default to a non-control level
        default_treat_idx = 1 if len(groups) > 1 else 0
        treatment_level = st.selectbox(
            "Treatment level",
            options=groups,
            index=default_treat_idx,
            key="het_treatment_level",
            help=PARAMETER_TOOLTIPS.get("het_treatment_level", ""),
        )

    if control_level == treatment_level:
        st.error("❌ Control and treatment must differ.")
        return

    excluded = [g for g in groups if g not in (control_level, treatment_level)]
    if excluded:
        st.info(f"ℹ️ Rows from groups {excluded} will be excluded from this analysis.")

    st.divider()
    st.subheader("🧩 Features")

    # Build candidate feature pool: numeric columns excluding the outcome,
    # plus the categorical columns excluding the group column.
    if use_cuped:
        outcome_col_for_filter = (st.session_state.get("het_cuped_base_metric") or "") + _get_post_suffix()
        outcome_pre_cols = [
            (st.session_state.get("het_cuped_base_metric") or "") + s for s in _get_pre_suffixes()
        ]
    else:
        outcome_col_for_filter = st.session_state.get("het_outcome_col")
        outcome_pre_cols = []

    numeric_feature_options = [
        c for c in numeric_cols_all if c != outcome_col_for_filter and c not in outcome_pre_cols
    ]
    categorical_options = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns
        if c != group_column and not is_id_column(df, c)
    ]

    numeric_features = st.multiselect(
        "Numeric features",
        options=numeric_feature_options,
        default=numeric_feature_options,
        key="het_numeric_features",
        help=PARAMETER_TOOLTIPS.get("het_numeric_features", ""),
    )
    categorical_features = st.multiselect(
        "Categorical features",
        options=categorical_options,
        default=categorical_options,
        key="het_categorical_features",
        help=PARAMETER_TOOLTIPS.get("het_categorical_features", ""),
    )

    # Per-column encoding choices (collapsed by default to reduce clutter)
    if categorical_features:
        with st.expander("Categorical encoding (per column)", expanded=False):
            current_encodings = st.session_state.get("het_cat_encodings", {})
            new_encodings = {}
            for col in categorical_features:
                n_levels = df[col].nunique(dropna=True)
                default_enc = current_encodings.get(col, "one-hot")
                if default_enc not in ("one-hot", "ordinal", "target"):
                    default_enc = "one-hot"
                idx = ("one-hot", "ordinal", "target").index(default_enc)
                new_encodings[col] = st.selectbox(
                    f"`{col}` ({n_levels} levels)",
                    options=("one-hot", "ordinal", "target"),
                    index=idx,
                    key=f"het_enc_{col}",
                    help=PARAMETER_TOOLTIPS.get("het_cat_encoding", ""),
                )
            st.session_state["het_cat_encodings"] = new_encodings

    if not numeric_features and not categorical_features:
        st.error("❌ Pick at least one feature.")
        return

    st.divider()
    st.subheader("🧠 Model")

    from results_analysis.heterogeneity import LEARNER_CHOICES, BASE_MODEL_CHOICES

    col_l, col_b = st.columns(2)
    with col_l:
        learner = st.selectbox(
            "Meta-learner",
            options=LEARNER_CHOICES,
            index=0,
            key="het_learner",
            help=PARAMETER_TOOLTIPS.get("het_learner", ""),
        )
    with col_b:
        base_model = st.selectbox(
            "Base ML model",
            options=BASE_MODEL_CHOICES,
            index=0,
            key="het_base_model",
            help=PARAMETER_TOOLTIPS.get("het_base_model", ""),
        )

    with st.expander("Advanced hyperparameters", expanded=False):
        n_estimators = st.number_input(
            "n_estimators (RF / GBM / CausalForest)",
            min_value=4, max_value=2000, value=100, step=4, key="het_hp_n_estimators",
            help=PARAMETER_TOOLTIPS.get("het_hp_n_estimators", ""),
        )
        max_depth = st.number_input(
            "max_depth (0 = unlimited)",
            min_value=0, max_value=50, value=0, step=1, key="het_hp_max_depth",
            help=PARAMETER_TOOLTIPS.get("het_hp_max_depth", ""),
        )
        min_samples_leaf = st.number_input(
            "min_samples_leaf",
            min_value=1, max_value=500, value=10, step=1, key="het_hp_min_samples_leaf",
            help=PARAMETER_TOOLTIPS.get("het_hp_min_samples_leaf", ""),
        )
        learning_rate = st.number_input(
            "learning_rate (GBM only)",
            min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f",
            key="het_hp_learning_rate",
            help=PARAMETER_TOOLTIPS.get("het_hp_learning_rate", ""),
        )

    st.divider()
    st.subheader("🔬 Validation")

    col_ts, col_bs, col_seed = st.columns(3)
    with col_ts:
        test_size = st.slider(
            "Holdout split",
            min_value=0.10, max_value=0.50, value=0.20, step=0.05,
            key="het_test_size",
            help=PARAMETER_TOOLTIPS.get("het_test_size", "Fraction of rows held out for the Qini curve."),
        )
    with col_bs:
        bootstrap_iters = st.number_input(
            "Bootstrap iterations (0 = off)",
            min_value=0, max_value=1000, value=200, step=50,
            key="het_bootstrap_iters",
            help=PARAMETER_TOOLTIPS.get("het_bootstrap_iters", "0 disables the CI."),
        )
    with col_seed:
        seed = st.number_input(
            "Random seed",
            min_value=0, max_value=2**31 - 1, value=42, step=1,
            key="het_random_seed",
            help=PARAMETER_TOOLTIPS.get("het_random_seed", ""),
        )

    st.divider()
    if st.button("🚀 Fit heterogeneity model", type="primary", use_container_width=True):
        _fit_and_store_heterogeneity(
            df=df,
            group_column=group_column,
            control_level=control_level,
            treatment_level=treatment_level,
            outcome_label=outcome_label,
            use_cuped=use_cuped,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            encodings=st.session_state.get("het_cat_encodings", {}),
            learner=learner,
            base_model=base_model,
            hyperparams=dict(
                n_estimators=int(n_estimators),
                max_depth=(None if int(max_depth) == 0 else int(max_depth)),
                min_samples_leaf=int(min_samples_leaf),
                learning_rate=float(learning_rate),
                random_state=int(seed),
            ),
            test_size=float(test_size),
            bootstrap_iters=int(bootstrap_iters),
            seed=int(seed),
        )


def _fit_and_store_heterogeneity(
    *, df, group_column, control_level, treatment_level, outcome_label,
    use_cuped, numeric_features, categorical_features, encodings,
    learner, base_model, hyperparams, test_size, bootstrap_iters, seed,
):
    """Heavy lifting for the Fit button: prepare data, fit, store results."""
    from sklearn.model_selection import train_test_split
    from results_analysis.heterogeneity import (
        prepare_features, fit_learner, estimate_cate, feature_importance,
        qini_curve, auuc_score, bootstrap_ate,
    )

    # Filter to control + treatment rows
    mask = df[group_column].isin([control_level, treatment_level])
    sub = df.loc[mask].copy()
    T = (sub[group_column] == treatment_level).astype(int).to_numpy()

    # Build outcome
    if use_cuped:
        base_metric = st.session_state.get("het_cuped_base_metric")
        pre_suffixes = _get_pre_suffixes()
        post_suffix = _get_post_suffix()
        pre_cols = [base_metric + s for s in pre_suffixes]
        post_col = base_metric + post_suffix
        if _is_multi_pre():
            res = _cuped_adjust_multi(sub[pre_cols], sub[post_col])
            if res is None:
                st.error("❌ CUPED+ adjustment failed (rank-deficient design or all NaN).")
                return
            Y_series = res[0]
        else:
            adj = _cuped_adjust(sub[pre_cols[0]], sub[post_col])
            if adj is None:
                st.error("❌ CUPED adjustment failed (zero variance pre).")
                return
            Y_series = adj
        Y = Y_series.to_numpy(dtype=float)
    else:
        outcome_col = st.session_state.get("het_outcome_col")
        Y = sub[outcome_col].to_numpy(dtype=float)

    # Build features
    try:
        X, feature_names = prepare_features(
            sub, numeric_features, categorical_features, encodings,
            y=pd.Series(Y, index=sub.index),
        )
    except Exception as exc:
        st.error(f"❌ Feature preparation failed: {exc}")
        return

    # Drop rows with any NaN in X or Y
    finite_mask = np.isfinite(Y) & X.notna().all(axis=1).to_numpy()
    n_dropped = int((~finite_mask).sum())
    X = X.loc[finite_mask]
    Y = Y[finite_mask]
    T = T[finite_mask]

    if X.shape[0] < 50:
        st.error(f"❌ Only {X.shape[0]} usable rows after dropping NaN; need at least 50.")
        return

    if n_dropped > 0:
        st.warning(f"⚠️ Dropped {n_dropped:,} rows with missing values in features or outcome.")

    # Holdout split
    X_tr, X_ho, Y_tr, Y_ho, T_tr, T_ho = train_test_split(
        X, Y, T, test_size=test_size, random_state=seed, stratify=T,
    )

    # Fit
    with st.spinner(f"Fitting {learner} ({base_model})..."):
        try:
            est = fit_learner(learner, X_tr, T_tr, Y_tr, base_model, hyperparams)
        except Exception as exc:
            st.error(f"❌ Model fitting failed: {exc}")
            return

    # Predictions
    cate_full = estimate_cate(est, X)
    cate_holdout = estimate_cate(est, X_ho)
    imp = feature_importance(est, feature_names)
    q_df = qini_curve(cate_holdout, Y_ho, T_ho)
    auuc = auuc_score(q_df)

    ci = None
    if bootstrap_iters > 0:
        with st.spinner(f"Bootstrapping ATE CI ({bootstrap_iters} iterations)..."):
            m, lo, hi = bootstrap_ate(est, X, n_iters=bootstrap_iters, seed=seed)
            ci = (lo, hi)
            ate_value = m
    else:
        ate_value = float(cate_full.mean())

    # Stash everything for the Results sub-tab
    st.session_state["het_fitted_result"] = {
        "learner": learner,
        "base_model": base_model,
        "outcome_label": outcome_label,
        "control_level": control_level,
        "treatment_level": treatment_level,
        "feature_names": feature_names,
        "importance": imp,
        "ate": ate_value,
        "ate_ci": ci,
        "cate_full": cate_full,
        "cate_holdout": cate_holdout,
        "qini_df": q_df,
        "auuc": auuc,
        "n_train": int(X_tr.shape[0]),
        "n_holdout": int(X_ho.shape[0]),
        "row_index": X.index,
        "df_filtered_index": sub.loc[finite_mask].index,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    # Add to artifact: plots, per-user CATE dataframe, and a full log entry
    artifact = st.session_state.get("results_analysis_artifact")
    if artifact:
        from results_analysis.heterogeneity_plots import (
            cate_histogram, feature_importance_bar, qini_plot,
        )

        # Drop any previously-rendered subgroup chart from a stale fit so the
        # artifact only contains plots that match the *current* model.
        artifact.remove_log(category="heterogeneity")
        for plot_name in list(getattr(artifact, "plots", {}).keys()):
            if plot_name.startswith("het_"):
                try:
                    del artifact.plots[plot_name]
                except Exception:
                    pass

        artifact.add_plot(
            "het_cate_histogram",
            cate_histogram(cate_full, ate=ate_value, ci=ci),
            "Heterogeneity: per-user CATE distribution",
        )
        if imp is not None:
            artifact.add_plot(
                "het_feature_importance",
                feature_importance_bar(imp, top_k=15),
                "Heterogeneity: feature importance",
            )
        artifact.add_plot(
            "het_qini_curve",
            qini_plot(q_df, auuc),
            "Heterogeneity: Qini curve on holdout",
        )

        # Per-user CATE — useful both for reproducibility and downstream targeting
        cate_export = pd.DataFrame({
            "row_index": X.index,
            "cate": cate_full,
            "group": sub.loc[finite_mask, group_column].values,
        })
        artifact.add_df(
            "heterogeneity_per_user_cate",
            cate_export,
            "Heterogeneity: predicted CATE per user (row_index aligns with input dataframe)",
        )

        artifact.add_log(
            category="heterogeneity",
            message=f"Heterogeneity model fit: {learner} ({base_model}) on '{outcome_label}'",
            details={
                "learner": learner,
                "base_model": base_model,
                "hyperparams": hyperparams,
                "outcome": outcome_label,
                "use_cuped_outcome": use_cuped,
                "control_level": control_level,
                "treatment_level": treatment_level,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "categorical_encodings": encodings,
                "n_features_expanded": len(feature_names),
                "n_train": int(X_tr.shape[0]),
                "n_holdout": int(X_ho.shape[0]),
                "n_rows_dropped_nan": n_dropped,
                "test_size": test_size,
                "bootstrap_iters": bootstrap_iters,
                "ate": ate_value,
                "ate_ci_lower": ci[0] if ci else None,
                "ate_ci_upper": ci[1] if ci else None,
                "auuc_vs_random": auuc,
                "top_features": list(imp.head(10).index) if imp is not None else None,
            },
        )

    st.success(
        f"✅ Fit complete · ATE = {ate_value:+.4f}"
        + (f" (95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])" if ci else "")
        + f" · AUUC = {auuc:+.2f}. See the **📈 Results** sub-tab."
    )


def render_heterogeneity_results():
    """Results sub-tab for the Heterogeneity Analysis."""
    st.header("🧬 Heterogeneity Analysis — Results")

    res = st.session_state.get("het_fitted_result")
    if res is None:
        st.info("ℹ️ Configure and fit a model in the **⚙️ Setup** sub-tab to see results.")
        return

    from results_analysis.heterogeneity_plots import (
        cate_histogram, feature_importance_bar, qini_plot, subgroup_bar,
    )
    from results_analysis.heterogeneity import (
        subgroup_cate_by_categorical, subgroup_cate_by_quantile,
    )

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Learner", res["learner"])
    with col2:
        ate_disp = f"{res['ate']:+.4f}"
        if res.get("ate_ci"):
            lo, hi = res["ate_ci"]
            st.metric("ATE", ate_disp, delta=f"95% CI [{lo:+.3f}, {hi:+.3f}]", delta_color="off")
        else:
            st.metric("ATE", ate_disp)
    with col3:
        st.metric("AUUC vs random", f"{res['auuc']:+.2f}")
    with col4:
        st.metric("Train / holdout", f"{res['n_train']:,} / {res['n_holdout']:,}")

    st.caption(
        f"Outcome: **{res['outcome_label']}**  ·  "
        f"Treatment level: **{res['treatment_level']}** vs control **{res['control_level']}**  ·  "
        f"Base model: **{res['base_model']}**"
    )

    st.divider()

    # CATE histogram
    fig_hist = cate_histogram(res["cate_full"], ate=res["ate"], ci=res.get("ate_ci"))
    render_plot_with_download(fig_hist, "het_cate_histogram.html", "Download CATE histogram")

    # Feature importance
    st.subheader("🔍 Feature importance")
    imp = res.get("importance")
    if imp is None:
        st.info("ℹ️ Feature importance not available for this base model (Linear/Logistic exposes coefficients only).")
    else:
        fig_fi = feature_importance_bar(imp, top_k=15)
        render_plot_with_download(fig_fi, "het_feature_importance.html", "Download feature importance")

    # Qini
    st.subheader("📈 Qini curve (held-out validation)")
    fig_q = qini_plot(res["qini_df"], res["auuc"])
    render_plot_with_download(fig_q, "het_qini.html", "Download Qini curve")
    st.caption(
        "How to read: the blue curve shows cumulative incremental outcome when you target the top X% "
        "of users by predicted CATE. The dashed line is random targeting. "
        "A model that's better than random sits above the dashed line; AUUC quantifies that area."
    )

    # Subgroup drill-down
    st.divider()
    st.subheader("🧮 Subgroup drill-down")

    is_valid, df, group_column = validate_data_and_group('results_uploaded_data', 'results_group_column')
    if not is_valid:
        return

    drill_options = [c for c in res["categorical_features"] + res["numeric_features"]]
    if not drill_options:
        st.info("No features available for drill-down.")
        return

    drill_col = st.selectbox(
        "Slice CATE by",
        options=drill_options,
        key="het_drill_col",
        help=PARAMETER_TOOLTIPS.get("het_drill_col", ""),
    )

    series = df.loc[res["row_index"], drill_col]
    if drill_col in res["categorical_features"]:
        sg = subgroup_cate_by_categorical(res["cate_full"], series)
        fig_sg = subgroup_bar(sg, drill_col, f"Mean CATE by {drill_col}")
    else:
        n_bins = st.slider(
            "Number of quantile bins", 2, 10, 5,
            key="het_drill_bins",
            help=PARAMETER_TOOLTIPS.get("het_drill_bins", ""),
        )
        sg = subgroup_cate_by_quantile(res["cate_full"], series, n_bins=n_bins)
        fig_sg = subgroup_bar(sg, f"{drill_col} (quantile)", f"Mean CATE by {drill_col} quantile")
    render_plot_with_download(fig_sg, f"het_subgroup_{drill_col}.html", "Download subgroup chart")
    st.dataframe(sg, use_container_width=True)

    # Capture this drill-down in the artifact (overwrites prior subgroup chart)
    artifact = st.session_state.get("results_analysis_artifact")
    if artifact:
        artifact.add_plot(
            f"het_subgroup_{drill_col}",
            fig_sg,
            f"Heterogeneity: mean CATE sliced by {drill_col}",
        )

    st.divider()
    st.subheader("💾 Per-user CATE export")
    out = pd.DataFrame({"row_index": res["row_index"], "cate": res["cate_full"]})
    csv = out.to_csv(index=False)
    st.download_button("💾 Download per-user CATE CSV", data=csv, file_name="cate_per_user.csv", mime="text/csv")
