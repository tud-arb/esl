# Visualizations for report

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_boxplot(
    fold_results: dict,
    metric: str = "f1_macro",
    save_path: str | None = None,
):

    rows = []

    for dataset, models in fold_results.items():
        for model, folds in models.items():
            for fold_dict in folds:
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    metric: fold_dict[metric],
                })

    df_plot = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x="model",
        y=metric,
        hue="dataset",
    )
    plt.title(f"Cross-Validation {metric} by Model and Dataset")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
