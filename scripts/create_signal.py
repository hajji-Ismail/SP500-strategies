import pickle
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from features_engineering import feature_engineering, load_data, split_train_test
from gridsearch import create_constrained_time_series_splits


data_path= "./data/all_stocks_5yr.csv"
model_path="./results/selected-model/selected_model.pkl"
output_path="./results/selected-model/ml_signal.csv"

def generate_signals():
   
    df = feature_engineering(load_data(data_path))

    with open(model_path, "rb") as f:
        selected_model = pickle.load(f)
    features = [c for c in df.columns if c not in {"Name", "forward_return"}]
    train, test = split_train_test(df)
    
    signals = []
    for _, (train_idx, val_idx) in enumerate(create_constrained_time_series_splits(train), 1):
        model = clone(selected_model).fit(train.iloc[train_idx][features], (train.iloc[train_idx]["target"] > 0).astype(int))
        values = model.predict_proba(train.iloc[val_idx][features])[:, 1]
        signals.append(pd.Series(values, index=train.iloc[val_idx].index, name="signal"))
    if not test.empty:
        full_model = clone(selected_model).fit(train[features], (train["target"] > 0).astype(int))
        signals.append(pd.Series(full_model.predict_proba(test[features])[:, 1], index=test.index, name="signal"))
    pd.concat(signals).sort_index().to_frame().to_csv(output_path)


if __name__ == "__main__":
    generate_signals()
