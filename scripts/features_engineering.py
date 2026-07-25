import pandas as pd
import numpy as np
import ta


def load_data(filepath="./data/all_stocks_5yr.csv"):
   

    df = pd.read_csv(filepath)

    

    df["date"] = pd.to_datetime(df["date"])

    # Sort BEFORE computing indicators
    df.sort_values(["Name", "date"], inplace=True)

    return df


def feature_engineering(df):


    print("Computing technical indicators...")

    feature_frames = []

    for Name, group in df.groupby("Name"):

        group = group.copy()

        group["RSI_14"] = ta.momentum.rsi(
            close=group["close"],
            window=14
        )

     
        group["MACD"] = ta.trend.macd(
            close=group["close"]
        )

        group["MACD_SIGNAL"] = ta.trend.macd_signal(
            close=group["close"]
        )

        group["MACD_DIFF"] = ta.trend.macd_diff(
            close=group["close"]
        )

        group["BB_UPPER"] = ta.volatility.bollinger_hband(
            close=group["close"]
        )

        group["BB_MIDDLE"] = ta.volatility.bollinger_mavg(
            close=group["close"]
        )

        group["BB_LOWER"] = ta.volatility.bollinger_lband(
            close=group["close"]
        )

   
        close_d1 = group["close"].shift(-1)
        close_d2 = group["close"].shift(-2)

        group["target"] = np.sign(
            (close_d2 - close_d1) / close_d1
        )

        feature_frames.append(group)

    df = pd.concat(feature_frames)

    # Remove rows with missing indicators/target
    required_columns = [
        "RSI_14",
        "MACD",
        "MACD_SIGNAL",
        "MACD_DIFF",
        "BB_UPPER",
        "BB_MIDDLE",
        "BB_LOWER",
        "target",
    ]

    df.dropna(subset=required_columns, inplace=True)

    # Multi-index required by the project
    df.set_index(["date", "Name"], inplace=True)
    df.sort_index(inplace=True)

    return df


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