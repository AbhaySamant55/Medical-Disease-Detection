"""Per-disease loading, cleaning and feature metadata."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_disease(key: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Return (X, y, info) for one disease, with dataset-specific cleaning applied."""
    spec = config.DISEASES[key]
    path = config.disease_path(key)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.download_data` first.")

    df = pd.read_csv(path)
    target = spec["target"]

    # Some datasets encode "not measured" as a literal 0 in columns where zero is
    # physiologically impossible (you cannot have a BMI of 0). Left alone, the
    # models learn that 0 is a meaningful value; converted to NaN, the imputer
    # in the pipeline handles them properly.
    for col in spec.get("zero_is_missing", []):
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    info = {
        "key": key,
        "name": spec["name"],
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "positive_rate": float(y.mean()),
        "majority_rate": float(max(y.mean(), 1 - y.mean())),
        "missing_cells": int(X.isna().sum().sum()),
        "missing_pct": float(X.isna().sum().sum() / (X.shape[0] * X.shape[1])),
        "source": spec["source"],
        "blurb": spec["blurb"],
        "positive_label": spec["positive_label"],
        "negative_label": spec["negative_label"],
        "icon": spec["icon"],
    }
    return X, y, info


def feature_metadata(X: pd.DataFrame, numeric_cols, categorical_cols) -> dict:
    """Ranges and defaults used to build the input widgets in the Streamlit app."""
    meta = {}
    for col in numeric_cols:
        s = X[col].dropna()
        if s.empty:
            meta[col] = {"kind": "numeric", "min": 0.0, "max": 1.0,
                         "default": 0.0, "step": 0.1, "integer": False}
            continue
        lo, hi = float(s.min()), float(s.max())
        span = hi - lo
        # widen slightly so a real patient outside the training range still fits
        pad = span * 0.15 if span > 0 else max(abs(lo) * 0.15, 1.0)
        is_int = bool(np.all(np.equal(np.mod(s, 1), 0))) and span > 3
        meta[col] = {
            "kind": "numeric",
            "min": float(np.floor(lo - pad)) if is_int else float(lo - pad),
            "max": float(np.ceil(hi + pad)) if is_int else float(hi + pad),
            "default": float(round(float(s.median()), 4)),
            "step": 1.0 if is_int else float(max(span / 200, 1e-4)),
            "integer": is_int,
            "observed_min": lo,
            "observed_max": hi,
            "mean": float(s.mean()),
        }
    for col in categorical_cols:
        values = sorted(str(v) for v in X[col].dropna().unique())
        mode = X[col].mode()
        meta[col] = {
            "kind": "categorical",
            "options": values,
            "default": str(mode.iloc[0]) if len(mode) else (values[0] if values else ""),
        }
    return meta


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified hold-out. The test split is never used for model selection or
    for the ensemble weights — only for the final numbers reported."""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)


def dataset_overview() -> pd.DataFrame:
    """One row per disease — used by the README and the app's landing page."""
    rows = []
    for key in config.DISEASES:
        try:
            X, y, info = load_disease(key)
        except FileNotFoundError:
            continue
        rows.append({
            "Disease": info["name"],
            "Samples": info["n_samples"],
            "Features": info["n_features"],
            "Positive %": round(100 * info["positive_rate"], 1),
            "Missing %": round(100 * info["missing_pct"], 1),
            "Source": info["source"],
        })
    return pd.DataFrame(rows)
