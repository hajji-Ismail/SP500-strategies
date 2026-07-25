"""Evaluate the selected pipeline by fold and save the production artifact."""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features_engineering import main
from gridsearch import create_constrained_time_series_splits, save_cv_plot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_DIR, MODEL_DIR = PROJECT_ROOT / "results/cross-validation", PROJECT_ROOT / "results/selected-model"


def build_pipeline(params=None):
    return Pipeline([("scaler", StandardScaler()), ("classifier", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1, **(params or {})))])


def _safe_auc(y, probabilities):
    return roc_auc_score(y, probabilities) if y.nunique() > 1 else float("nan")


def run_model_selection():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, _ = main()
    features = [c for c in train.columns if c not in {"target", "forward_return"}]
    X, y = train[features], (train["target"] > 0).astype(int)
    params = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.01}
    params_path = PROJECT_ROOT / "results/hyperparameters/best_params.json"
    if params_path.exists():
        params = {key.removeprefix("classifier__"): value for key, value in json.loads(params_path.read_text()).items()}

    splits = create_constrained_time_series_splits(train)
    save_cv_plot(train, splits)
    metrics, importance = [], []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = build_pipeline(params).fit(X_tr, y_tr)
        for sample, x_data, y_data in (("train", X_tr, y_tr), ("validation", X_val, y_val)):
            probability = model.predict_proba(x_data)[:, 1]
            metrics.append({"fold": fold, "sample": sample, "accuracy": accuracy_score(y_data, model.predict(x_data)), "auc": _safe_auc(y_data, probability), "logloss": log_loss(y_data, probability, labels=[0, 1])})
        values = model.named_steps["classifier"].feature_importances_
        importance.append(pd.DataFrame({"fold": fold, "feature": features, "importance": values}).nlargest(10, "importance"))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(CV_DIR / "ml_metrics_train.csv", index=False)
    pd.concat(importance, ignore_index=True).to_csv(CV_DIR / "top_10_feature_importance.csv", index=False)
    train_metrics = metrics_df.query("sample == 'train'")
    validation = metrics_df.query("sample == 'validation'")
    fig, ax = plt.subplots(figsize=(10, 5))
    folds = np.arange(1, len(validation) + 1)
    width = 0.38
    ax.bar(folds - width / 2, train_metrics["auc"], width, label="Train AUC", color="navy")
    ax.bar(folds + width / 2, validation["auc"], width, label="Validation AUC", color="darkorange")
    ax.set(xlabel="Fold number", ylabel="AUC", title="Train and validation AUC across temporal folds")
    ax.set_xticks(folds)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(CV_DIR / "metric_train.png", dpi=200); plt.close(fig)

    final_model = build_pipeline(params).fit(X, y)
    with (MODEL_DIR / "selected_model.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    summary = metrics_df.groupby("sample")[["accuracy", "auc", "logloss"]].mean().round(4)
    (MODEL_DIR / "selected_model.txt").write_text(
        "Selected pipeline\n=================\n"
        "StandardScaler -> XGBClassifier\n"
        f"Hyperparameters: {params}\n\nMean fold metrics:\n{summary.to_string()}\n"
    )
    print(f"Saved model and cross-validation artifacts under {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    run_model_selection()
