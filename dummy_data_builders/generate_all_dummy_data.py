"""
Generate all dummy data files for the application
Run this script locally to create CSV files that will be loaded by the UI
"""
import pandas as pd
import os
from dummy_data_builders.power_analysis_builder import generate_power_analysis_data
from dummy_data_builders.group_selection_builder import generate_group_selection_data
from dummy_data_builders.rebalancer_builder import generate_rebalancer_data
from dummy_data_builders.results_analysis_builder import (
    generate_results_analysis_data,
    generate_results_analysis_data_multi_pre,
    generate_results_analysis_data_heterogeneous,
)
def run_data():
    # Create output directory
    output_dir = "dummy_data2"
    os.makedirs(output_dir, exist_ok=True)

    print("🎲 Generating dummy data files...\n")

    # 1. Power Analysis (with missing values)
    print("1. Generating Power Analysis data...")
    df_power = generate_power_analysis_data(
        n_rows=10000,
        n_numeric_metrics=4,
        n_categorical=2,
        include_outliers=True,
        missing_pct=0.05,
        random_seed=42
    )
    df_power.to_csv(os.path.join(output_dir, "power_analysis_dummy.csv"), index=False)
    print(f"   ✅ Created: {len(df_power)} rows, {len(df_power.columns)} columns")

    # 2. Group Selection (no missing values)
    print("2. Generating Group Selection data...")
    df_group = generate_group_selection_data(
        n_rows=10000,
        n_numeric_metrics=4,
        n_categorical=2,
        include_outliers=True,
        missing_pct=0.0,  # No missing values
        random_seed=42
    )
    df_group.to_csv(os.path.join(output_dir, "group_selection_dummy.csv"), index=False)
    print(f"   ✅ Created: {len(df_group)} rows, {len(df_group.columns)} columns")

    # 3. Rebalancer (with slightly imbalanced groups, no missing values)
    print("3. Generating Rebalancer data...")
    df_rebalancer = generate_rebalancer_data(
        n_rows=10000,
        n_groups=3,
        imbalance_level=0.05,  # Small imbalance (mostly balanced)
        n_numeric_metrics=4,
        n_categorical=2,
        include_outliers=True,
        missing_pct=0.0,  # No missing values
        random_seed=42
    )
    df_rebalancer.to_csv(os.path.join(output_dir, "rebalancer_dummy.csv"), index=False)
    print(f"   ✅ Created: {len(df_rebalancer)} rows, {len(df_rebalancer.columns)} columns")
    print(f"   📊 Groups: {df_rebalancer['group'].value_counts().to_dict()}")

    # 4. Results Analysis (with treatment effects, no missing values)
    print("4. Generating Results Analysis data...")
    df_results = generate_results_analysis_data(
        n_rows=10000,
        n_groups=2,
        treatment_effect=0.15,  # 15% uplift
        cuped_suffix_pre="_pre",
        cuped_suffix_post="_post",
        random_seed=42
    )
    df_results.to_csv(os.path.join(output_dir, "results_analysis_dummy.csv"), index=False)
    print(f"   ✅ Created: {len(df_results)} rows, {len(df_results.columns)} columns")
    print(f"   📊 Groups: {df_results['group'].value_counts().to_dict()}")

    # 5. Results Analysis — multi-pre variant (CUPED+ / parallel-trends DiD)
    print("5. Generating Results Analysis (multi-pre) data...")
    df_results_multi = generate_results_analysis_data_multi_pre(
        n_rows=10000,
        n_groups=2,
        treatment_effect=0.15,
        pre_suffixes=("_pre_w1", "_pre_w2", "_pre_w3"),
        post_suffix="_post",
        random_seed=7,
    )
    df_results_multi.to_csv(os.path.join(output_dir, "results_analysis_multi_pre_dummy.csv"), index=False)
    print(f"   ✅ Created: {len(df_results_multi)} rows, {len(df_results_multi.columns)} columns")
    print(f"   📊 Groups: {df_results_multi['group'].value_counts().to_dict()}")

    # 6. Results Analysis — heterogeneous treatment effects (for the Heterogeneity tab)
    print("6. Generating Results Analysis (heterogeneous HTE) data...")
    df_results_het = generate_results_analysis_data_heterogeneous(
        n_rows=10000,
        pre_suffixes=("_pre_w1", "_pre_w2", "_pre_w3"),
        post_suffix="_post",
        random_seed=11,
    )
    df_results_het.to_csv(os.path.join(output_dir, "results_analysis_heterogeneous_dummy.csv"), index=False)
    print(f"   ✅ Created: {len(df_results_het)} rows, {len(df_results_het.columns)} columns")
    print(f"   📊 Groups: {df_results_het['group'].value_counts().to_dict()}")

    print(f"\n✅ All dummy data files generated in '{output_dir}/' directory!")
    print("\nFiles created:")
    print("  - power_analysis_dummy.csv")
    print("  - group_selection_dummy.csv")
    print("  - rebalancer_dummy.csv")
    print("  - results_analysis_dummy.csv")
    print("  - results_analysis_multi_pre_dummy.csv")
    print("  - results_analysis_heterogeneous_dummy.csv")
