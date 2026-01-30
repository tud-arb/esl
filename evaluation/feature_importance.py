from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ImportanceResult:
    df: pd.DataFrame            
    method: str                 


def _ensure_series(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D importance array, got shape={arr.shape}")
    return arr


def importance_logreg_pipeline(model, feature_names) -> ImportanceResult:
    clf = model.named_steps["clf"]
    coefs = np.asarray(clf.coef_)  
    imp = np.mean(np.abs(coefs), axis=0)  
    imp = _ensure_series(imp)

    df = pd.DataFrame({"feature": list(feature_names), "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return ImportanceResult(df=df, method="logreg_abs_coef_mean_over_classes")


def importance_tree_model(model, feature_names) -> ImportanceResult:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        raise ValueError("Model has no attribute feature_importances_")

    imp = _ensure_series(imp)

    df = pd.DataFrame({"feature": list(feature_names), "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return ImportanceResult(df=df, method="model_feature_importances_")


def normalise_importance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["importance"].sum()
    if s == 0 or np.isnan(s):
        out["importance_norm"] = 0.0
    else:
        out["importance_norm"] = out["importance"] / s
    return out


def aggregate_importance_across_folds(fold_importances: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for fold, df in fold_importances.items():
        tmp = df[["feature", "importance_norm"]].copy()
        tmp = tmp.rename(columns={"importance_norm": f"fold_{fold}"})
        merged = tmp if merged is None else merged.merge(tmp, on="feature", how="outer")

    merged = merged.fillna(0.0)

    fold_cols = [c for c in merged.columns if c.startswith("fold_")]
    merged["importance_mean"] = merged[fold_cols].mean(axis=1)
    merged["importance_std"] = merged[fold_cols].std(axis=1)

    ranks = []
    for c in fold_cols:
        ranks.append(merged[c].rank(ascending=False, method="average"))
    merged["rank_mean"] = pd.concat(ranks, axis=1).mean(axis=1)

    merged = merged.sort_values(["rank_mean", "importance_mean"], ascending=[True, False])
    return merged.reset_index(drop=True)


def consensus_top_features(importance_tables: Dict[str, pd.DataFrame], top_k: int = 20) -> pd.DataFrame:
    rows = []
    for model_name, df in importance_tables.items():
        top = df.head(top_k).copy()
        top["model"] = model_name
        top["in_top_k"] = 1
        rows.append(top[["feature", "model", "in_top_k"]])

    if not rows:
        return pd.DataFrame(columns=["feature", "freq_in_top_k"])

    cat = pd.concat(rows, ignore_index=True)
    freq = cat.groupby("feature")["in_top_k"].sum().reset_index()
    freq = freq.rename(columns={"in_top_k": f"freq_in_top_{top_k}"})
    freq = freq.sort_values(f"freq_in_top_{top_k}", ascending=False).reset_index(drop=True)
    return freq
