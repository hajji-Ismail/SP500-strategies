import pandas as pd
import numpy as np


def load_data(filepath="./data/all_stocks_5yr.csv"):
   

    df = pd.read_csv(filepath)

    

    df["date"] = pd.to_datetime(df["date"])

    df.sort_values(["Name", "date"], inplace=True)

    return df

def feature_engineering(df : pd.DataFrame):
   
    frames = []
    for _, group in df.groupby("Name", sort=False):
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

        group["forward_return"] = close.shift(-2).div(close.shift(-1)).sub(1)
        group["target"] = np.sign(group["forward_return"]) + 2
        frames.append(group)

    features = pd.concat(frames, ignore_index=True)
    required = ["RSI_14", "MACD", "MACD_SIGNAL", "MACD_DIFF", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "forward_return", "target"]
    features = features.dropna(subset=required)
    return features.set_index(["date", "Name"]).sort_index()


def split_train_test(df):


    train = df[df.index.get_level_values("date") < "2017-01-01"].copy()

    test = df[df.index.get_level_values("date") >= "2017-01-01"].copy()

    return train, test

def main() :

    print("Loading data...")
    df = load_data()
    print("Performing feature engineering...")
    df = feature_engineering(df)
    print("Splitting train/test...")
    train_df, test_df = split_train_test(df)

    return train_df, test_df



if __name__ == "__main__":

    print("Loading data...")
    df = load_data()

    print("Performing feature engineering...")
    df = feature_engineering(df)

    print("Splitting train/test...")
    train_df, test_df = split_train_test(df)

    print(f"Train Shape : {train_df.shape}")
    print(f"Test Shape  : {test_df.shape}")

    print("\nTrain period:")
    print(train_df.index.get_level_values("date").min(), "->",
          train_df.index.get_level_values("date").max())

    print("\nTest period:")
    print(test_df.index.get_level_values("date").min(), "->",
          test_df.index.get_level_values("date").max())