# Predicting Short-term Price Movements from Market Microstructure Data

## Data Sources

### Candles

* **Exchange:** Binance
* **Endpoint:** `/api/v3/klines`
* **Resolution:** 1 minute
* **Fields:** open, high, low, close, volume

### Order Books

* **Exchange:** Bybit
* **Format:** Line-delimited JSON snapshots
* **Depth:** Top-200 levels (top-K configurable)
* **Frequency:** Exchange snapshot frequency (sub-second)

---

## Features

### Candle Features

Each feature has its own function and is orchestrated by a single wrapper.

Computed features:
* Returns
* RSI
* Bollinger Bands (upper, lower, width)
* ATR

All rolling-window NaNs are removed automatically.

### Order Book Features

Computed per snapshot:
* Best bid / ask
* Bid-ask spread
* Midprice
* Bid depth (top-K)
* Ask depth (top-K)
* Order book imbalance
* Weighted imbalance

Order books are processed **snapshot-by-snapshot** to keep memory usage bounded.

---

## Labels
Targets are generated using **future returns**:

* `+1` → price increases beyond threshold
* `-1` → price decreases beyond threshold
* `0` → stable / no significant movement

The prediction horizon and stability threshold are configurable.

---

## Project Structure

```text
project/
├── data/
│   └── BTCUSDT/
│       ├── raw/                 # Raw Bybit order book files (.data)
│       └── processed/           # Final Parquet datasets (features + labels)
│
├── src/
│   ├── data_loader.py           # Streaming loaders (Binance, Bybit)
│   ├── preprocessing.py         # Data cleaning, alignment, resampling
│   ├── features.py              # Candle & order-book feature engineering
│   └── save_load.py             # Parquet persistence utilities
│
├── models/
│   ├── baseline.py              # Naive / benchmark model
│   ├── lightgbm_model.py        # LightGBM training + inference
│   ├── xgboost_model.py         # XGBoost training + inference
│   └── train_utils.py           # Shared training utilities (splits, scaling, CV)
│
├── evaluation/
│   ├── metrics.py               # Accuracy, F1, Sharpe, directional accuracy
│   ├── plot_utils.py            # Feature plots, prediction diagnostics
│   └── stats.py                 # Statistical tests, distributions, summaries
│
├── experiments/
│
├── notebooks/
│   └── exploration.ipynb        # Ad-hoc analysis & sanity checks
│
├── main.py                      # End-to-end data → features → datasets
├── requirements.txt
└── README.md
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## Running the Pipeline

1. **Place Bybit order book files** in:

   ```text
   data/<SYMBOL>/raw/
   ```

   Example filename pattern:

   ```text
   2025-12-01_BTCUSDT_ob200.data
   ```

2. **Configure parameters** in `main.py`:

   ```python
   SYMBOL = "BTCUSDT"
   CANDLE_INTERVAL = "1m"
   OB_RESAMPLE_FREQ = "1min"

   PRED_HORIZON = 1
   STABILITY_THRESHOLD = 0.0002
   BATCH_SIZE = 10_000
   ```

3. **Run the pipeline**:
   ```bash
   python main.py
   ```

Progress bars will appear while loading large order book files. It takes about 5 minutes to load 1 day of data on my machine. For the sake of consistency, I have already downloaded and processed a week of data for BTCUSDT, from 01.12.2025 to 07.12.2025. DO NOT push the raw data files to Github. 

---

## Outputs

Three datasets are produced:
* `dataset_candle.parquet`
* `dataset_orderbook.parquet`
* `dataset_combined.parquet`

Each row represents **one aligned timestamp**, ready for ML training.

---

## Disclaimer

This project is for **research and educational purposes only**. It is not financial advice and should not be used for live trading without extensive validation.
