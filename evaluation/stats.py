import numpy as np
import pandas as pd


def aggregate_results(fold_results):
    df = pd.DataFrame(fold_results)

    summary = {}
    for col in df.columns:
        if col == "fold":
            continue
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()

    return summary
