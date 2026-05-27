"""
Heterogeneous Treatment Effect (HTE) estimation.

Core math/model module for the Heterogeneity Analysis tab. Wraps
sklearn-compatible estimators and econml meta-learners with a uniform
interface so the UI layer doesn't have to know about each library's quirks.
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
import warnings

import numpy as np
import pandas as pd


# Public constants exposed in the UI
LEARNER_CHOICES = ("T-learner", "S-learner", "X-learner", "DR-learner", "CausalForest")
BASE_MODEL_CHOICES = ("RandomForest", "GradientBoosting", "Linear/Logistic")
ENCODING_CHOICES = ("one-hot", "ordinal", "target")


def prepare_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    encodings: Dict[str, str],
    y: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a numeric design matrix from a mixed-type dataframe.

    Args:
        df: source dataframe
        numeric_cols: numeric features to pass through unchanged
        categorical_cols: categorical features to encode
        encodings: mapping {column_name: 'one-hot' | 'ordinal' | 'target'}
        y: outcome series (required when any column uses 'target' encoding)

    Returns:
        (X, feature_names) — index aligned to df.index. Columns are float.
    """
    parts: List[pd.DataFrame] = []
    feature_names: List[str] = []

    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").astype(float)
        parts.append(s.to_frame(name=col))
        feature_names.append(col)

    for col in categorical_cols:
        enc = encodings.get(col, "one-hot")
        s = df[col].astype("string").fillna("__NA__")

        if enc == "one-hot":
            dummies = pd.get_dummies(s, prefix=col, drop_first=False, dtype=float)
            parts.append(dummies)
            feature_names.extend(dummies.columns.tolist())
        elif enc == "ordinal":
            # Stable level ordering: most frequent first
            levels = s.value_counts().index.tolist()
            mapping = {lvl: i for i, lvl in enumerate(levels)}
            parts.append(s.map(mapping).astype(float).to_frame(name=col))
            feature_names.append(col)
        elif enc == "target":
            if y is None:
                raise ValueError(f"Target encoding requested for '{col}' but no y provided")
            y_arr = pd.Series(np.asarray(y, dtype=float), index=df.index)
            global_mean = float(y_arr.mean())
            level_means = y_arr.groupby(s).mean()
            encoded = s.map(level_means).fillna(global_mean).astype(float)
            parts.append(encoded.to_frame(name=col))
            feature_names.append(col)
        else:
            raise ValueError(f"Unknown encoding: {enc}")

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    return X, feature_names


def _build_base_model(name: str, hyperparams: Dict[str, Any], *, classification: bool):
    """Factory for the inner sklearn-compatible model."""
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression

    hp = dict(hyperparams or {})
    seed = hp.get("random_state", 42)

    if name == "RandomForest":
        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=hp.get("max_depth", None),
            min_samples_leaf=int(hp.get("min_samples_leaf", 10)),
            random_state=seed,
            n_jobs=-1,
        )
    if name == "GradientBoosting":
        cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        return cls(
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=int(hp.get("max_depth", 3)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            random_state=seed,
        )
    if name == "Linear/Logistic":
        if classification:
            return LogisticRegression(max_iter=1000, random_state=seed)
        return LinearRegression()

    raise ValueError(f"Unknown base model: {name}")


def fit_learner(
    learner_choice: str,
    X: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    base_model_name: str,
    hyperparams: Dict[str, Any],
):
    """
    Fit a meta-learner (or CausalForest). Returns the fitted econml estimator.

    Binary treatment is assumed: T is 0/1. Inputs are expected to be aligned
    and free of NaN.
    """
    from econml.metalearners import SLearner, TLearner, XLearner
    from econml.dr import DRLearner
    from econml.dml import CausalForestDML
    from sklearn.base import clone

    outcome_model = _build_base_model(base_model_name, hyperparams, classification=False)
    propensity_model = _build_base_model(base_model_name, hyperparams, classification=True)

    X_arr = X.to_numpy(dtype=float)
    Y_arr = np.asarray(Y, dtype=float).ravel()
    T_arr = np.asarray(T, dtype=int).ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if learner_choice == "S-learner":
            est = SLearner(overall_model=clone(outcome_model))
            est.fit(Y=Y_arr, T=T_arr, X=X_arr)
        elif learner_choice == "T-learner":
            est = TLearner(models=clone(outcome_model))
            est.fit(Y=Y_arr, T=T_arr, X=X_arr)
        elif learner_choice == "X-learner":
            est = XLearner(
                models=clone(outcome_model),
                propensity_model=clone(propensity_model),
                cate_models=clone(outcome_model),
            )
            est.fit(Y=Y_arr, T=T_arr, X=X_arr)
        elif learner_choice == "DR-learner":
            est = DRLearner(
                model_propensity=clone(propensity_model),
                model_regression=clone(outcome_model),
                model_final=clone(outcome_model),
                random_state=hyperparams.get("random_state", 42),
            )
            est.fit(Y=Y_arr, T=T_arr, X=X_arr)
        elif learner_choice == "CausalForest":
            # CausalForestDML requires n_estimators divisible by subforest_size (default 4)
            n_est_raw = int(hyperparams.get("n_estimators", 100))
            n_est = max(4, ((n_est_raw + 3) // 4) * 4)
            est = CausalForestDML(
                model_y=clone(outcome_model),
                model_t=clone(propensity_model),
                discrete_treatment=True,
                n_estimators=n_est,
                max_depth=hyperparams.get("max_depth", None),
                min_samples_leaf=int(hyperparams.get("min_samples_leaf", 10)),
                random_state=hyperparams.get("random_state", 42),
            )
            est.fit(Y=Y_arr, T=T_arr, X=X_arr)
        else:
            raise ValueError(f"Unknown learner: {learner_choice}")

    return est


def estimate_cate(est, X: pd.DataFrame) -> np.ndarray:
    """Per-row CATE prediction."""
    arr = est.effect(X.to_numpy(dtype=float))
    return np.asarray(arr, dtype=float).ravel()


def feature_importance(est, feature_names: List[str]) -> Optional[pd.Series]:
    """
    Best-effort feature importance extraction across learner types.

    CausalForest exposes feature_importances_ directly. Meta-learners store
    their fitted inner models on attributes like ``models``, ``overall_model``,
    or ``model_final_``; we average importances across inner models when there
    are multiple. S-learner is special-cased because its inner model takes
    treatment as an extra feature, so the FI vector has length n_features + 1.

    Returns None if no inner model exposes feature_importances_ (e.g. a
    Linear/Logistic base model).
    """
    n = len(feature_names)

    # 1) Native feature_importances_ on the estimator (CausalForest)
    if hasattr(est, "feature_importances_"):
        try:
            vals = np.asarray(est.feature_importances_, dtype=float).ravel()
            if len(vals) == n:
                return pd.Series(vals, index=feature_names).sort_values(ascending=False)
        except Exception:
            pass

    # 2) Walk known meta-learner attributes
    candidate_attrs = ("models", "overall_model", "cate_models", "model_final_")
    for attr in candidate_attrs:
        m = getattr(est, attr, None)
        if m is None:
            continue
        candidates = m if isinstance(m, (list, tuple)) else [m]
        imps: List[np.ndarray] = []
        for sub in candidates:
            if hasattr(sub, "feature_importances_"):
                vec = np.asarray(sub.feature_importances_, dtype=float).ravel()
                # S-learner appends one-hot treatment columns to the features
                # (length n+1 or n+2 depending on encoding). Slice to the
                # genuine feature block.
                if n < len(vec) <= n + 2:
                    vec = vec[:n]
                if len(vec) == n:
                    imps.append(vec)
        if imps:
            avg = np.mean(np.vstack(imps), axis=0)
            return pd.Series(avg, index=feature_names).sort_values(ascending=False)

    return None


def qini_curve(cate: np.ndarray, Y: np.ndarray, T: np.ndarray) -> pd.DataFrame:
    """
    Compute the Qini curve on a (typically held-out) sample.

    Ranks units by predicted CATE (descending) and accumulates the observed
    incremental treated-vs-control outcome at each top-k slice. Returns a
    dataframe with columns:
        frac  — fraction of population targeted (0..1)
        qini  — model's cumulative incremental outcome
        random — straight-line baseline (random targeting)
    """
    n = len(cate)
    df = pd.DataFrame({"cate": cate, "y": np.asarray(Y, dtype=float), "t": np.asarray(T, dtype=int)})
    df = df.sort_values("cate", ascending=False).reset_index(drop=True)

    df["frac"] = (np.arange(n) + 1) / n
    cumsum_yt = (df["y"] * df["t"]).cumsum()
    cumsum_yc = (df["y"] * (1 - df["t"])).cumsum()
    cumsum_nt = df["t"].cumsum()
    cumsum_nc = (1 - df["t"]).cumsum()

    safe_nc = cumsum_nc.replace(0, np.nan)
    qini = cumsum_yt - cumsum_yc * (cumsum_nt / safe_nc)
    df["qini"] = qini.fillna(0.0)
    df["random"] = df["qini"].iloc[-1] * df["frac"]
    return df[["frac", "qini", "random"]]


def auuc_score(qini_df: pd.DataFrame) -> float:
    """Area under the Qini curve above the random baseline (positive = lift)."""
    diff = qini_df["qini"].to_numpy() - qini_df["random"].to_numpy()
    return float(np.trapezoid(diff, qini_df["frac"].to_numpy()))


def bootstrap_ate(
    est,
    X: pd.DataFrame,
    n_iters: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the ATE by resampling the prediction set.

    This is a fast post-hoc CI — it does NOT re-fit the model on each draw,
    so it captures sampling variation of the population mean given a fixed
    estimator. For inference that accounts for model uncertainty use
    DR-learner or CausalForestDML's native `ate_inference()` methods.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    cates = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        c = est.effect(X.iloc[idx].to_numpy(dtype=float))
        cates.append(float(np.mean(c)))
    arr = np.asarray(cates)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def subgroup_cate_by_categorical(
    cate: np.ndarray,
    categorical: pd.Series,
) -> pd.DataFrame:
    """Mean CATE per categorical level, with sample size."""
    df = pd.DataFrame({"cate": np.asarray(cate, dtype=float), "level": categorical.astype("string").fillna("__NA__").values})
    g = df.groupby("level").agg(mean_cate=("cate", "mean"), n=("cate", "count"))
    return g.sort_values("mean_cate", ascending=False)


def subgroup_cate_by_quantile(
    cate: np.ndarray,
    numeric: pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Mean CATE per quantile bin of a numeric feature."""
    qs = pd.qcut(pd.to_numeric(numeric, errors="coerce"), q=n_bins, duplicates="drop")
    df = pd.DataFrame({"cate": np.asarray(cate, dtype=float), "bin": qs.astype("string")})
    g = df.groupby("bin", observed=True).agg(mean_cate=("cate", "mean"), n=("cate", "count"))
    # Keep bin order natural (ascending)
    return g
