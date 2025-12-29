# Save/load datasets
import pandas as pd

def save_dataset(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=True)

def load_dataset(path: str):
    return pd.read_parquet(path)