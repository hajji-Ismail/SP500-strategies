"""Build a leakage-free, ticker-level data set for the ML strategy."""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "all_stocks_5yr.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_data.csv"
SPLIT_DATE = pd.Timestamp("2017-01-01")


def load_data(filepath=RAW_DATA_PATH):
    """Load and order OHLCV observations before any per-ticker calculation."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={"Name": "ticker"})
    required = {"date", "open", "high", "low", "close", "volume", "ticker"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def feature_engineering(df):
    """Compute indicators using information available by the close of each date.

    ``forward_return`` is return(D+1, D+2), deliberately aligned to date D.
    It is the return used for both the target and the backtest.
    """
    frames = []
    for ticker, group in df.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        close = group["close"].astype(float)

        delta = close.diff()
        gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        group["RSI_14"] = 100 - (100 / (1 + rs))

        ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
        group["MACD"] = ema_fast - ema_slow
        group["MACD_SIGNAL"] = group["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
        group["MACD_DIFF"] = group["MACD"] - group["MACD_SIGNAL"]

        middle = close.rolling(window=20, min_periods=20).mean()
        std = close.rolling(window=20, min_periods=20).std(ddof=0)
        group["BB_MIDDLE"] = middle
        group["BB_UPPER"] = middle + 2 * std
        group["BB_LOWER"] = middle - 2 * std

        # On D, only prices through D are features; this is the held-period return.
        group["forward_return"] = close.shift(-2).div(close.shift(-1)).sub(1)
        group["target"] = np.sign(group["forward_return"]).astype(float)
        frames.append(group)

    features = pd.concat(frames, ignore_index=True)
    required = ["RSI_14", "MACD", "MACD_SIGNAL", "MACD_DIFF", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "forward_return", "target"]
    features = features.dropna(subset=required)
    return features.set_index(["date", "ticker"]).sort_index()


def split_train_test(df, split_date=SPLIT_DATE):
    dates = df.index.get_level_values("date")
    return df.loc[dates < split_date].copy(), df.loc[dates >= split_date].copy()


def main(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """Create, persist and return the train/test data sets."""
    engineered = feature_engineering(load_data(input_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.reset_index().to_csv(output_path, index=False)
    return split_train_test(engineered)


if __name__ == "__main__":
    train_df, test_df = main()
    print(f"Saved {PROCESSED_DATA_PATH}")
    print(f"Train: {train_df.shape}, {train_df.index.get_level_values('date').min().date()} to {train_df.index.get_level_values('date').max().date()}")
    print(f"Test:  {test_df.shape}, {test_df.index.get_level_values('date').min().date()} to {test_df.index.get_level_values('date').max().date()}")
