"""
Generate dummy data for Results Analysis page
Creates realistic experiment data with pre/post columns showing treatment effects
"""
import pandas as pd
import numpy as np


def generate_results_analysis_data(
    n_rows: int = 10000,
    n_groups: int = 2,
    group_names: list = None,
    treatment_effect: float = 0.15,  # 15% uplift for treatment group
    cuped_suffix_pre: str = "_pre",
    cuped_suffix_post: str = "_post",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic experiment results data with pre/post columns.
    
    Creates data with:
    - ID column (customer_id)
    - 3 numeric columns with pre/post versions:
      1. Both groups increase, but treatment increases more
      2. Similar pre values, but in post one goes up, other stays same
      3. One group increases, other decreases (interesting contrast)
    - 2 numeric columns without pre/post:
      1. customer_churned (0-1): lower in treatment group
      2. new_customer (0-1): higher in treatment group
    - 2 categorical columns (region, device_type)
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate
    n_groups : int
        Number of groups (typically 2 for A/B test)
    group_names : list
        Optional list of group names (if None, generates Control, Treatment, etc.)
    treatment_effect : float
        Treatment effect as multiplier (0.15 = 15% uplift)
    cuped_suffix_pre : str
        Suffix for pre columns (used for both CUPED and DiD)
    cuped_suffix_post : str
        Suffix for post columns (used for both CUPED and DiD)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame: Generated data with groups and metrics (pre/post versions)
    """
    np.random.seed(random_seed)
    
    # Generate group names
    if group_names is None:
        if n_groups == 2:
            group_names = ['Control', 'Treatment']
        else:
            group_names = [f"Group_{chr(65+i)}" for i in range(n_groups)]
    else:
        n_groups = len(group_names)
    
    # Balanced group sizes
    base_size = n_rows // n_groups
    sizes = [base_size] * n_groups
    sizes[0] += (n_rows - sum(sizes))  # Add remainder to first group
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_rows + 1)]
    
    # Assign groups
    group_assignments = []
    for i, size in enumerate(sizes):
        group_assignments.extend([group_names[i]] * size)
    
    # Shuffle to randomize
    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    group_assignments = [group_assignments[i] for i in indices]
    customer_ids = [customer_ids[i] for i in indices]
    
    data = {
        'customer_id': customer_ids,
        'group': group_assignments
    }
    
    # ===== METRIC 1: Both groups increase, but treatment increases more =====
    # Column: revenue
    metric1_pre = np.zeros(n_rows)
    metric1_post = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        # Pre-period: Both groups have similar baseline (with some variation)
        pre_base = np.random.lognormal(mean=3.5, sigma=1.2, size=n_group)
        metric1_pre[group_mask] = np.round(pre_base, 2)
        
        # Post-period: Both increase, but treatment increases more
        if group_idx == 0:  # Control
            # Control: small increase (5-8%)
            increase = np.random.normal(loc=1.06, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.03, 1.10)
        else:  # Treatment
            # Treatment: larger increase (15-20%)
            increase = np.random.normal(loc=1.17, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.13, 1.22)
        
        metric1_post[group_mask] = np.round(pre_base * increase, 2)
    
    data['revenue'] = metric1_post
    data['revenue' + cuped_suffix_pre] = metric1_pre
    data['revenue' + cuped_suffix_post] = metric1_post
    
    # ===== METRIC 2: Similar pre values, but in post one goes up, other stays same =====
    # Column: engagement_score
    metric2_pre = np.zeros(n_rows)
    metric2_post = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        # Pre-period: Very similar values for both groups
        pre_base = np.random.normal(loc=50, scale=5, size=n_group)  # Tight distribution
        pre_base = np.clip(pre_base, 0, 100)
        metric2_pre[group_mask] = np.round(pre_base, 2)
        
        # Post-period: Treatment goes up, Control stays almost same
        if group_idx == 0:  # Control
            # Control: stays almost the same (slight variation ±2%)
            variation = np.random.normal(loc=1.0, scale=0.01, size=n_group)
            variation = np.clip(variation, 0.98, 1.02)
            metric2_post[group_mask] = np.round(pre_base * variation, 2)
        else:  # Treatment
            # Treatment: significant increase (20-25%)
            increase = np.random.normal(loc=1.22, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.18, 1.27)
            metric2_post[group_mask] = np.round(pre_base * increase, 2)
            metric2_post[group_mask] = np.clip(metric2_post[group_mask], 0, 100)
    
    data['engagement_score'] = metric2_post
    data['engagement_score' + cuped_suffix_pre] = metric2_pre
    data['engagement_score' + cuped_suffix_post] = metric2_post
    
    # ===== METRIC 3: One group increases, other decreases (interesting contrast) =====
    # Column: support_tickets
    metric3_pre = np.zeros(n_rows)
    metric3_post = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        # Pre-period: Both groups start with similar baseline
        pre_base = np.random.poisson(lam=3, size=n_group).astype(float)
        metric3_pre[group_mask] = np.round(pre_base, 2)
        
        # Post-period: Treatment decreases (good!), Control increases slightly
        if group_idx == 0:  # Control
            # Control: slight increase (bad - more tickets)
            increase = np.random.normal(loc=1.08, scale=0.02, size=n_group)
            increase = np.clip(increase, 1.05, 1.12)
            metric3_post[group_mask] = np.round(pre_base * increase, 2)
        else:  # Treatment
            # Treatment: decreases (good - fewer tickets, better service)
            decrease = np.random.normal(loc=0.75, scale=0.03, size=n_group)
            decrease = np.clip(decrease, 0.70, 0.82)
            metric3_post[group_mask] = np.round(pre_base * decrease, 2)
            metric3_post[group_mask] = np.maximum(metric3_post[group_mask], 0)  # No negatives
    
    data['support_tickets'] = metric3_post
    data['support_tickets' + cuped_suffix_pre] = metric3_pre
    data['support_tickets' + cuped_suffix_post] = metric3_post
    
    # ===== METRIC 4: customer_churned (0-1) - lower in treatment =====
    churned_values = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        if group_idx == 0:  # Control
            # Control: higher churn rate (0.15-0.20)
            churn_rate = np.random.beta(a=2, b=8, size=n_group)  # Mean ~0.2
        else:  # Treatment
            # Treatment: lower churn rate (0.08-0.12)
            churn_rate = np.random.beta(a=2, b=15, size=n_group)  # Mean ~0.12
        
        # Convert to binary (0 or 1)
        churned_values[group_mask] = (np.random.random(n_group) < churn_rate).astype(int)
    
    data['customer_churned'] = churned_values.astype(int)
    
    # ===== METRIC 5: new_customer (0-1) - higher in treatment =====
    new_customer_values = np.zeros(n_rows)
    
    for group_idx, group_name in enumerate(group_names):
        group_mask = np.array([g == group_name for g in group_assignments])
        n_group = np.sum(group_mask)
        
        if n_group == 0:
            continue
        
        if group_idx == 0:  # Control
            # Control: lower new customer rate (0.10-0.15)
            new_rate = np.random.beta(a=2, b=12, size=n_group)  # Mean ~0.14
        else:  # Treatment
            # Treatment: higher new customer rate (0.20-0.25)
            new_rate = np.random.beta(a=3, b=10, size=n_group)  # Mean ~0.23
        
        # Convert to binary (0 or 1)
        new_customer_values[group_mask] = (np.random.random(n_group) < new_rate).astype(int)
    
    data['new_customer'] = new_customer_values.astype(int)
    
    # Add categorical columns (2 as per standard)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_rows)
    data['device_type'] = np.random.choice(['Desktop', 'Mobile', 'Tablet'], size=n_rows, p=[0.5, 0.4, 0.1])

    df = pd.DataFrame(data)

    return df


def generate_results_analysis_data_multi_pre(
    n_rows: int = 10000,
    n_groups: int = 2,
    group_names: list = None,
    treatment_effect: float = 0.15,
    pre_suffixes: tuple = ("_pre_w1", "_pre_w2", "_pre_w3"),
    post_suffix: str = "_post",
    random_seed: int = 7,
) -> pd.DataFrame:
    """
    Multi-pre experiment dataset for testing CUPED+ and parallel-trends DiD.

    Produces three pre-period windows per metric with a deliberate **seasonal
    drift** between windows (so the parallel-trends DiD plot is non-trivial)
    and varying correlation profiles between pre-windows and post-period —
    so the CUPED+ coefficient bar shows distinct θᵢ values.

    Metrics included:
      - revenue: highly correlated across all 3 pre-windows + post; CUPED+
        R² should be high, all θᵢ positive.
      - engagement_score: only the *most recent* pre-window (w3) carries
        signal; w1/w2 are mostly noise. Multi-pre coefficients should show
        w3 dominating.
      - support_tickets: declining seasonal trend, all 3 pre-windows
        contribute roughly equally. Treatment reduces post further.
      - customer_churned, new_customer: binary outcomes (no pre/post).

    Group naming, categoricals, and overall shape match the standard
    single-pre dummy dataset for consistency.
    """
    np.random.seed(random_seed)

    if group_names is None:
        group_names = ['Control', 'Treatment'] if n_groups == 2 else [f"Group_{chr(65+i)}" for i in range(n_groups)]
    else:
        n_groups = len(group_names)

    base_size = n_rows // n_groups
    sizes = [base_size] * n_groups
    sizes[0] += (n_rows - sum(sizes))

    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_rows + 1)]

    group_assignments = []
    for i, size in enumerate(sizes):
        group_assignments.extend([group_names[i]] * size)

    indices = np.arange(n_rows)
    np.random.shuffle(indices)
    group_assignments = [group_assignments[i] for i in indices]
    customer_ids = [customer_ids[i] for i in indices]

    data = {'customer_id': customer_ids, 'group': group_assignments}
    group_arr = np.array(group_assignments)

    # ===== Metric 1: revenue — strong correlation across all pre windows =====
    # Per-user latent baseline that shows up consistently in every period.
    latent = np.random.lognormal(mean=3.5, sigma=1.0, size=n_rows)
    drift_w1, drift_w2, drift_w3, drift_post = 1.00, 1.04, 1.09, 1.15

    rev_w1 = latent * drift_w1 + np.random.normal(0, 8.0, n_rows)
    rev_w2 = latent * drift_w2 + np.random.normal(0, 8.0, n_rows)
    rev_w3 = latent * drift_w3 + np.random.normal(0, 8.0, n_rows)
    rev_post = latent * drift_post + np.random.normal(0, 8.0, n_rows)
    treat_mask = group_arr == group_names[1] if n_groups > 1 else np.zeros(n_rows, dtype=bool)
    # Modest ~2.5% lift — significant on raw data, more so after CUPED.
    rev_post[treat_mask] *= 1.025

    data['revenue' + pre_suffixes[0]] = np.round(np.clip(rev_w1, 0, None), 2)
    data['revenue' + pre_suffixes[1]] = np.round(np.clip(rev_w2, 0, None), 2)
    data['revenue' + pre_suffixes[2]] = np.round(np.clip(rev_w3, 0, None), 2)
    data['revenue' + post_suffix] = np.round(np.clip(rev_post, 0, None), 2)

    # ===== Metric 2: engagement_score — only w3 carries signal =====
    # Tiny lift on raw data (p around 0.05–0.2). CUPED+ with w3 dominating
    # should sharpen it noticeably.
    eng_latent = np.random.normal(50, 8, n_rows)
    eng_w1 = np.random.normal(50, 10, n_rows)              # noise
    eng_w2 = np.random.normal(50, 10, n_rows)              # noise
    eng_w3 = eng_latent + np.random.normal(0, 3, n_rows)   # signal
    eng_post = eng_latent + np.random.normal(0, 4, n_rows)
    eng_post[treat_mask] *= 1.01  # ~1% bump

    data['engagement_score' + pre_suffixes[0]] = np.round(np.clip(eng_w1, 0, 100), 2)
    data['engagement_score' + pre_suffixes[1]] = np.round(np.clip(eng_w2, 0, 100), 2)
    data['engagement_score' + pre_suffixes[2]] = np.round(np.clip(eng_w3, 0, 100), 2)
    data['engagement_score' + post_suffix] = np.round(np.clip(eng_post, 0, 100), 2)

    # ===== Metric 3: support_tickets — declining trend, all 3 pre-windows informative =====
    # Noisy reduction — marginal on raw, clear after CUPED+ using all 3 baselines.
    tick_latent = np.random.poisson(lam=4, size=n_rows).astype(float)
    tick_w1 = tick_latent + np.random.normal(0, 1.5, n_rows)
    tick_w2 = tick_latent * 0.92 + np.random.normal(0, 1.5, n_rows)
    tick_w3 = tick_latent * 0.85 + np.random.normal(0, 1.5, n_rows)
    tick_post = tick_latent * 0.78 + np.random.normal(0, 1.5, n_rows)
    tick_post[treat_mask] *= 0.97  # ~3% reduction in tickets

    data['support_tickets' + pre_suffixes[0]] = np.round(np.clip(tick_w1, 0, None), 2)
    data['support_tickets' + pre_suffixes[1]] = np.round(np.clip(tick_w2, 0, None), 2)
    data['support_tickets' + pre_suffixes[2]] = np.round(np.clip(tick_w3, 0, None), 2)
    data['support_tickets' + post_suffix] = np.round(np.clip(tick_post, 0, None), 2)

    # ===== Binary outcomes (no pre periods) =====
    churn = np.zeros(n_rows, dtype=int)
    new_cust = np.zeros(n_rows, dtype=int)
    for g_idx, g_name in enumerate(group_names):
        mask = group_arr == g_name
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        if g_idx == 0:
            churn_rate = np.random.beta(2, 8, n_g)
            new_rate = np.random.beta(2, 12, n_g)
        else:
            churn_rate = np.random.beta(2, 15, n_g)
            new_rate = np.random.beta(3, 10, n_g)
        churn[mask] = (np.random.random(n_g) < churn_rate).astype(int)
        new_cust[mask] = (np.random.random(n_g) < new_rate).astype(int)

    data['customer_churned'] = churn
    data['new_customer'] = new_cust

    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=n_rows)
    data['device_type'] = np.random.choice(['Desktop', 'Mobile', 'Tablet'], size=n_rows, p=[0.5, 0.4, 0.1])

    return pd.DataFrame(data)


def generate_results_analysis_data_heterogeneous(
    n_rows: int = 10000,
    pre_suffixes: tuple = ("_pre_w1", "_pre_w2", "_pre_w3"),
    post_suffix: str = "_post",
    random_seed: int = 11,
) -> pd.DataFrame:
    """
    Multi-pre experiment dataset with **genuine heterogeneous treatment effects**
    for testing the 🧬 Heterogeneity Analysis tab.

    Unlike the homogeneous multi-pre dummy (which applies a flat +2.5% lift to
    every treated user), here the per-user treatment effect varies in
    interpretable ways:

      * **engagement_tier** (low / medium / high) — main HTE driver
            low:    0%   lift   (no responders)
            medium: +4%  lift
            high:   +10% lift   (strongest responders)
      * **region** (East / West / North / South) — secondary driver
            East:  +3%, West: +2%, North: 0%, South: −1%
      * **tenure_months** (numeric, 1..60) — gentle continuous driver
            New users (<6 months) see no extra lift; very tenured users see +3%
      * **device_type, plan_type** — included as features for realism but do
            not modify the treatment effect

    Total per-user lift = tier_lift + region_lift + tenure_lift, applied
    multiplicatively to the post-period revenue of treated users.

    Realistic ATE (≈ +5.8%) hides a wide CATE distribution roughly spanning
    [−1%, +16%], so:
      * the CATE histogram is *wide*
      * Qini AUUC is clearly positive
      * feature importance highlights engagement_tier, region, tenure_months
      * subgroup drill-down by engagement_tier shows a clear gradient
    """
    np.random.seed(random_seed)
    n = n_rows

    # Balanced 50/50 group assignment
    group = np.array(["Control"] * (n // 2) + ["Treatment"] * (n - n // 2))
    np.random.shuffle(group)
    treat_mask = group == "Treatment"

    customer_id = [f"CUST_{i:06d}" for i in range(1, n + 1)]

    # ---- Features that drive heterogeneity ----
    engagement_tier = np.random.choice(["low", "medium", "high"], n, p=[0.35, 0.45, 0.20])
    region = np.random.choice(["East", "West", "North", "South"], n, p=[0.30, 0.25, 0.25, 0.20])
    device_type = np.random.choice(["Desktop", "Mobile", "Tablet"], n, p=[0.5, 0.4, 0.1])
    plan_type = np.random.choice(["Basic", "Premium", "Enterprise"], n, p=[0.5, 0.35, 0.15])
    tenure_months = np.clip(np.random.exponential(scale=14, size=n), 1, 60).astype(int)
    new_customer = (tenure_months < 6).astype(int)

    # ---- Latent baseline (correlated with plan_type and engagement_tier) ----
    plan_mult = np.array([{"Basic": 30.0, "Premium": 60.0, "Enterprise": 120.0}[p] for p in plan_type])
    engagement_mult = np.array([{"low": 0.7, "medium": 1.0, "high": 1.5}[e] for e in engagement_tier])
    latent = plan_mult * engagement_mult * np.random.lognormal(0.0, 0.30, n)

    # ---- Pre-period seasonal drift + noise ----
    drift_w1, drift_w2, drift_w3, drift_post = 1.00, 1.03, 1.06, 1.10
    noise = 8.0
    rev_w1 = latent * drift_w1 + np.random.normal(0, noise, n)
    rev_w2 = latent * drift_w2 + np.random.normal(0, noise, n)
    rev_w3 = latent * drift_w3 + np.random.normal(0, noise, n)
    rev_post = latent * drift_post + np.random.normal(0, noise, n)

    # ---- Heterogeneous treatment effect ----
    tier_lift = np.array([{"low": 0.00, "medium": 0.04, "high": 0.10}[e] for e in engagement_tier])
    region_lift = np.array([{"East": 0.03, "West": 0.02, "North": 0.00, "South": -0.01}[r] for r in region])
    tenure_lift = np.clip((tenure_months - 6) / 24.0, 0, 1) * 0.03
    user_lift = tier_lift + region_lift + tenure_lift

    rev_post[treat_mask] *= (1 + user_lift[treat_mask])

    data = {
        "customer_id": customer_id,
        "group": group,
        "revenue" + pre_suffixes[0]: np.round(np.clip(rev_w1, 0, None), 2),
        "revenue" + pre_suffixes[1]: np.round(np.clip(rev_w2, 0, None), 2),
        "revenue" + pre_suffixes[2]: np.round(np.clip(rev_w3, 0, None), 2),
        "revenue" + post_suffix: np.round(np.clip(rev_post, 0, None), 2),
        "engagement_tier": engagement_tier,
        "region": region,
        "device_type": device_type,
        "plan_type": plan_type,
        "tenure_months": tenure_months,
        "new_customer": new_customer,
    }
    return pd.DataFrame(data)
