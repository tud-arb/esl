import pandas as pd
from pathlib import Path
from glob import glob

from src.data_loader import fetch_candles, iter_bybit_orderbooks
from src.features import compute_orderbook_features
from src.save_load import save_dataset, load_dataset

# CONFIG
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

BASE_DATA_DIR = Path("data")
RAW_DIR = BASE_DATA_DIR / SYMBOL / "raw"
PROCESSED_DIR = BASE_DATA_DIR / SYMBOL / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PRED_HORIZON = 1
STABILITY_THRESHOLD = 0.0002
BATCH_SIZE = 10_000

def create_labels(close: pd.Series) -> pd.Series:
    future_return = close.shift(-PRED_HORIZON) / close - 1
    labels = pd.Series(0, index=close.index)
    labels[future_return > STABILITY_THRESHOLD] = 1
    labels[future_return < -STABILITY_THRESHOLD] = -1
    return labels


def main():
    # 1. Candles
    candles = fetch_candles(SYMBOL, INTERVAL, limit=1000)
    candle_df = candles.copy()
    candle_df["return"] = candle_df["close"].pct_change()
    candle_df.dropna(inplace=True)

    # 2. Stream order books
    orderbook_files = sorted(glob(str(RAW_DIR / f"2025-12-*_{SYMBOL}_ob200.*")))

    rows = []
    timestamps = []
    parquet_path = PROCESSED_DIR / "orderbook_features.parquet"

    for file in orderbook_files:
        print(f"Processing {file}")
        for ts, bids, asks in iter_bybit_orderbooks(file, max_depth=10):
            rows.append(compute_orderbook_features(bids, asks))
            timestamps.append(ts)

            if len(rows) >= BATCH_SIZE:
                df = pd.DataFrame(rows, index=timestamps)
                save_dataset(df, parquet_path)
                rows.clear()
                timestamps.clear()

    # Final flush
    if rows:
        df = pd.DataFrame(rows, index=timestamps)
        save_dataset(df, parquet_path)

    # 3. Load OB features
    ob_df = load_dataset(parquet_path)

    # 4. Align
    common_idx = candle_df.index.intersection(ob_df.index)
    candle_df = candle_df.loc[common_idx]
    ob_df = ob_df.loc[common_idx]

    # 5. Labels
    labels = create_labels(candle_df["close"])
    candle_df = candle_df.loc[labels.index]
    ob_df = ob_df.loc[labels.index]

    # 6. Final datasets
    candle_ds = candle_df.copy()
    candle_ds["target"] = labels

    ob_ds = ob_df.copy()
    ob_ds["target"] = labels

    combined_ds = pd.concat([candle_df, ob_df], axis=1)
    combined_ds["target"] = labels

    # 7. Save datasets
    save_dataset(candle_ds, PROCESSED_DIR / "dataset_candle.parquet")
    save_dataset(ob_ds, PROCESSED_DIR / "dataset_orderbook.parquet")
    save_dataset(combined_ds, PROCESSED_DIR / "dataset_combined.parquet")

    print("Finished successfully.")


if __name__ == "__main__":
    main()
