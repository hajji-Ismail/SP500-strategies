"""Generate leakage-free out-of-fold train signals and final-model test signals."""
import pickle
from pathlib import Path

import pandas as pd

from features_engineering import feature_engineering, load_data
from gridsearch import create_constrained_time_series_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_signals(data_path=PROJECT_ROOT / "data/all_stocks_5yr.csv", model_path=PROJECT_ROOT / "results/selected-model/selected_model.pkl", output_path=PROJECT_ROOT / "results/selected-model/ml_signal.csv"):
    data_path, model_path, output_path = map(Path, (data_path, model_path, output_path))
    df = feature_engineering(load_data(data_path))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run model_selection.py first.")
    with model_path.open("rb") as handle:
        selected_model = pickle.load(handle)
    features = [c for c in df.columns if c not in {"target", "forward_return"}]
    train = df.loc[df.index.get_level_values("date") < pd.Timestamp("2017-01-01")]
    test = df.loc[df.index.get_level_values("date") >= pd.Timestamp("2017-01-01")]
    signals = []
    # A fresh clone is fitted for each fold: no validation observation is used to fit it.
    from sklearn.base import clone
    for _, (train_idx, val_idx) in enumerate(create_constrained_time_series_splits(train), 1):
        model = clone(selected_model).fit(train.iloc[train_idx][features], (train.iloc[train_idx]["target"] > 0).astype(int))
        values = model.predict_proba(train.iloc[val_idx][features])[:, 1]
        signals.append(pd.Series(values, index=train.iloc[val_idx].index, name="signal"))
    if not test.empty:
        full_model = clone(selected_model).fit(train[features], (train["target"] > 0).astype(int))
        signals.append(pd.Series(full_model.predict_proba(test[features])[:, 1], index=test.index, name="signal"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(signals).sort_index().to_frame().to_csv(output_path)
    print(f"Saved leakage-free signals to {output_path}")


if __name__ == "__main__":
    generate_signals()
