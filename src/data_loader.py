import json
import numpy as np
import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm

def fetch_candles(symbol: str, interval: str = "1m", limit: int = 1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    resp = requests.get(url, params=params)
    data = resp.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ],
    )

    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)

    return df

def iter_bybit_orderbooks(file_path: Path, max_depth: int = 10):
    """
    Generator that streams Bybit orderbook snapshots one-by-one
    """
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, "r") as f:
        for line in tqdm(f, total=total_lines, desc=f"Loading {file_path}"):
            obj = json.loads(line)

            ts = pd.to_datetime(obj["ts"], unit="ms")
            bids = obj["data"]["b"][:max_depth]
            asks = obj["data"]["a"][:max_depth]

            if not bids or not asks:
                continue

            yield (
                ts,
                np.asarray(bids, dtype=np.float64),
                np.asarray(asks, dtype=np.float64),
            )