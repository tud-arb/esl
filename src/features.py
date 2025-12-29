# Candle & Order-Book Features
import pandas as pd
import numpy as np

# ============================================================
# =================== CANDLE FEATURES ========================
# ============================================================
def compute_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change()

def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(df: pd.DataFrame, window: int = 20):
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()

    upper = sma + 2 * std
    lower = sma - 2 * std
    width = upper - lower

    return upper, lower, width

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_candle_features(
    df: pd.DataFrame,
    rsi_window: int = 14,
    bb_window: int = 20,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Orchestrator for all candle-based features.
    Operates on full OHLCV DataFrame.
    """
    out = df.copy()

    out["return"] = compute_returns(out)
    out["RSI"] = compute_rsi(out, rsi_window)

    bb_u, bb_l, bb_w = compute_bollinger_bands(out, bb_window)
    out["BB_upper"] = bb_u
    out["BB_lower"] = bb_l
    out["BB_width"] = bb_w

    out["ATR"] = compute_atr(out, atr_window)

    out.dropna(inplace=True)
    return out


# ============================================================
# ================= ORDER BOOK FEATURES ======================
# ============================================================

def compute_best_prices_np(bids: np.ndarray, asks: np.ndarray):
    best_bid = bids[0, 0]
    best_ask = asks[0, 0]
    midprice = (best_bid + best_ask) / 2
    return best_bid, best_ask, midprice

def compute_bid_ask_spread(best_bid: float, best_ask: float) -> float:
    return best_ask - best_bid

def compute_depth(bids: np.ndarray, asks: np.ndarray, k: int = 5):
    k = min(k, len(bids), len(asks))
    bid_depth = bids[:k, 1].sum()
    ask_depth = asks[:k, 1].sum()
    return bid_depth, ask_depth

def compute_orderbook_imbalance(bid_depth: float, ask_depth: float) -> float:
    total = bid_depth + ask_depth
    if total == 0:
        return np.nan
    return (bid_depth - ask_depth) / total

def compute_weighted_imbalance(bids: np.ndarray, asks: np.ndarray, k: int = 5) -> float:
    k = min(k, len(bids), len(asks))
    weights = 1.0 / np.arange(1, k + 1)

    w_b = np.sum(weights * bids[:k, 1])
    w_a = np.sum(weights * asks[:k, 1])

    if (w_b + w_a) == 0:
        return 0.0
    return (w_b - w_a) / (w_b + w_a)

def compute_orderbook_features(
    bids: np.ndarray,
    asks: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Orchestrator for ALL order book features (single snapshot).
    Returns a plain dict (faster than Series in loops).
    """
    best_bid, best_ask, midprice = compute_best_prices_np(bids, asks)
    bid_depth, ask_depth = compute_depth(bids, asks, k)

    return {
        "bid_ask_spread": compute_bid_ask_spread(best_bid, best_ask),
        "midprice": midprice,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "orderbook_imbalance": compute_orderbook_imbalance(bid_depth, ask_depth),
        "weighted_imbalance": compute_weighted_imbalance(bids, asks, k),
    }
