import sys
from pathlib import Path

import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
Path("figures").mkdir(exist_ok=True)

from src.save_load import load_dataset
from evaluation.metrics import classification_metrics
from evaluation.stats import aggregate_results
from evaluation.plot_utils import plot_metric_boxplot
from evaluation.cv import time_series_cv

PROCESSED_DIR = Path("data") / "BTCUSDT" / "processed"

datasets = {
    "candle":    load_dataset(PROCESSED_DIR / "dataset_candle.parquet"),
    "orderbook": load_dataset(PROCESSED_DIR / "dataset_orderbook.parquet"),
    "combined":  load_dataset(PROCESSED_DIR / "dataset_combined.parquet"),
}

def make_models(random_state: int = 42):
    models = {}
    models["logreg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,                 
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ))
    ])

    import xgboost as xgb
    models["xgboost"] = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        n_estimators=500,         
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
    )

    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=500,      
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    except ImportError:
        pass

    return models
    


all_fold_results = {}   
results = {}            

N_SPLITS = 5
GAP = 0

for ds_name, df in datasets.items():
    if "target" not in df.columns:
        raise ValueError(f"{ds_name} dataset has no 'target' column")

    y = df["target"].map({-1: 0, 0: 1, 1: 2}).astype(int)
    X = df.drop(columns=["target"]).copy()
    
    obj_cols = X.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    all_fold_results[ds_name] = {}
    results[ds_name] = {}
    TEST_SIZE = max(500, min(5000, len(df) // (N_SPLITS + 2)))

    for model_name, model in make_models().items():
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(
            time_series_cv(
                len(df),
                n_splits=N_SPLITS,
                test_size=TEST_SIZE,
                gap=GAP,
            )
        ):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = classification_metrics(y_test, preds)
            metrics["fold"] = fold

            fold_results.append(metrics)

        all_fold_results[ds_name][model_name] = fold_results
        results[ds_name][model_name] = aggregate_results(fold_results)

rows = []
for dataset, models in results.items():
    for model, metrics in models.items():
        row = {"dataset": dataset, "model": model}
        row.update(metrics)
        rows.append(row)

summary_df = pd.DataFrame(rows).sort_values(["dataset", "model"])
summary_path = Path("figures") / "results_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary table -> {summary_path.resolve()}")
print(summary_df.to_string(index=False))

fold_path = Path("figures") / "fold_results.json"
with open(fold_path, "w", encoding="utf-8") as f:
    json.dump(all_fold_results, f, indent=2)
print(f"\nSaved fold-level results -> {fold_path.resolve()}")

for metric in ["accuracy", "balanced_accuracy", "f1_macro"]:
    plot_metric_boxplot(
        all_fold_results,
        metric=metric,
        save_path=f"figures/{metric}_boxplot.png",
    )
