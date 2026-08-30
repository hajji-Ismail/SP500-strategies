import pickle
from pathlib import Path
import pandas as pd
from sklearn.base import clone

from features_engineering import feature_engineering, load_data, split_train_test

data_path = "./data/all_stocks_5yr.csv"
model_path = "./results/selected-model/selected_model.pkl"
output_path = "./results/selected-model/ml_signal.csv"

def generate_signals():
    df = feature_engineering(load_data(data_path))

    with open(model_path, "rb") as f:
        selected_model = pickle.load(f)

    non_feature_cols = {"Name", "date", "forward_return", "high", "low", "open", "close", "volume", "target"}
    features = [c for c in df.columns if c not in non_feature_cols]

    train, test = split_train_test(df)

    full_model = clone(selected_model).fit(train[features], train["target"])

    train_signals = pd.Series(full_model.predict(train[features]), index=train.index, name="signal")
    test_signals = pd.Series(full_model.predict(test[features]), index=test.index, name="signal")

    full_signals = pd.concat([train_signals, test_signals]).sort_index()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    full_signals.to_frame().to_csv(output_path)
    print(f"[✓] Saved signals to {output_path}")

if __name__ == "__main__":
    generate_signals()