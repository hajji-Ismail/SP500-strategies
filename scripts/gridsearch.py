"""Temporal hyperparameter search for the stock-direction classifier."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features_engineering import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_constrained_time_series_splits(df, n_splits=10, min_train_years=2.0):
    """Return expanding folds whose boundaries are dates, never individual rows."""
    unique_dates = df.index.get_level_values("date").unique().sort_values()
    min_train_days = int(min_train_years * 252) + 1  # strictly more than two years
    if len(unique_dates) <= min_train_days + n_splits:
        raise ValueError("Not enough unique dates for the requested folds and initial history.")
    validation_edges = np.linspace(min_train_days, len(unique_dates), n_splits + 1, dtype=int)
    all_dates = df.index.get_level_values("date")
    splits = []
    for fold in range(n_splits):
        train_dates = unique_dates[:validation_edges[fold]]
        validation_dates = unique_dates[validation_edges[fold]:validation_edges[fold + 1]]
        train_idx = np.flatnonzero(all_dates.isin(train_dates))
        validation_idx = np.flatnonzero(all_dates.isin(validation_dates))
        if len(validation_idx) == 0:
            raise ValueError("A validation fold is empty.")
        splits.append((train_idx, validation_idx))
    return splits


def save_cv_plot(df, cv_splits, save_path=PROJECT_ROOT / "results/cross-validation/Time_series_split.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dates = df.index.get_level_values("date")
    fig, ax = plt.subplots(figsize=(12, 6))
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        ax.scatter(dates[train_idx], np.full(len(train_idx), fold), color="navy", s=1, label="Train" if fold == 1 else None)
        ax.scatter(dates[val_idx], np.full(len(val_idx), fold), color="darkorange", s=1, label="Validation" if fold == 1 else None)
    ax.set(xlabel="Date", ylabel="Fold", title="Expanding Time Series Cross-Validation (initial train > 2 years)")
    ax.set_yticks(range(1, len(cv_splits) + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def run_grid_search():
    train, _ = main()
    feature_cols = [c for c in train.columns if c not in {"target", "forward_return"}]
    X, y = train[feature_cols], (train["target"] > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1))])
    splits = create_constrained_time_series_splits(train)
    save_cv_plot(train, splits)
    search = GridSearchCV(pipeline, {"classifier__n_estimators": [100, 200], "classifier__max_depth": [3, 5], "classifier__learning_rate": [0.01, 0.1]}, cv=splits, scoring="roc_auc", n_jobs=-1, verbose=1)
    search.fit(X, y)
    output = PROJECT_ROOT / "results/hyperparameters"
    output.mkdir(parents=True, exist_ok=True)
    (output / "best_params.json").write_text(json.dumps(search.best_params_, indent=2))
    print(f"Best ROC-AUC: {search.best_score_:.4f}\nBest parameters: {search.best_params_}")
    return search


if __name__ == "__main__":
    run_grid_search()
