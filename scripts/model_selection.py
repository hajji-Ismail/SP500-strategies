import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier

from features_engineering import main
# Import the custom time series split logic from Task 2 (gridsearch.py)
from gridsearch import create_constrained_time_series_splits

# Output directories setup
CV_DIR = "results/cross-validation"
MODEL_DIR = "results/selected-model"
os.makedirs(CV_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def build_pipeline(params=None):
    """
    Constructs the exact preprocessing and model pipeline as Task 2.
    """
    if params is None:
        params = {}

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', **params))
    ])
    return pipeline


def run_model_selection():
    # 1. Load data from features_engineering
    df_train, df_test = main()

    # 2. Extract feature columns and target
    non_feature_cols = ["open", "high", "low", "close", "volume", "target", "Name", "date"]
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

    X_train = df_train[feature_cols]
    y_train = (df_train["target"] > 0).astype(int)  # Binary target {0, 1}

    # 3. Define hyperparameters (either from Task 2 search or defaults)
    # Removing 'classifier__' prefix if passed directly to XGBClassifier
    best_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.01
    }

    # If you saved best_params.json in Task 2, load them automatically:
    params_json_path = "results/hyperparameters/best_params.json"
    if os.path.exists(params_json_path):
        with open(params_json_path, "r") as f:
            raw_params = json.load(f)
            best_params = {k.replace('classifier__', ''): v for k, v in raw_params.items()}

    # 4. Create custom time-series splits
    cv_splits = create_constrained_time_series_splits(df_train, n_splits=10, min_train_years=2.0)

    metrics_list = []
    feature_importances_list = []

    print("\n--- Starting Task 3: Cross-Validation Evaluation ---")

    # 5. Execute 10-Fold Evaluation Loop
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit Pipeline on Fold Train set
        pipeline = build_pipeline(best_params)
        pipeline.fit(X_tr, y_tr)

        # Predict Probabilities & Classes
        tr_preds = pipeline.predict(X_tr)
        tr_probs = pipeline.predict_proba(X_tr)[:, 1]

        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        # Calculate Metrics
        fold_metrics = {
            'fold': fold + 1,
            'train_accuracy': accuracy_score(y_tr, tr_preds),
            'val_accuracy': accuracy_score(y_val, val_preds),
            'train_auc': roc_auc_score(y_tr, tr_probs),
            'val_auc': roc_auc_score(y_val, val_probs),
            'train_logloss': log_loss(y_tr, tr_probs),
            'val_logloss': log_loss(y_val, val_probs)
        }
        metrics_list.append(fold_metrics)

        # Extract Feature Importances for XGBoost
        clf = pipeline.named_steps['classifier']
        fold_importance = pd.DataFrame({
            'fold': fold + 1,
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)

        feature_importances_list.append(fold_importance)

    # Convert Metrics to DataFrame & calculate Mean
    metrics_df = pd.DataFrame(metrics_list)
    mean_row = metrics_df.mean(numeric_only=True).to_dict()
    mean_row['fold'] = 'Mean'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save ml_metrics_train.csv
    metrics_df.to_csv(os.path.join(CV_DIR, "ml_metrics_train.csv"), index=False)
    print(f"[✓] Saved metrics table to: {CV_DIR}/ml_metrics_train.csv")

    # 6. Aggregate Top 10 Feature Importances Across Folds
    all_importances = pd.concat(feature_importances_list, ignore_index=True)
    top_10_features = (
        all_importances.groupby('feature')['importance']
        .mean()
        .reset_index()
        .sort_values('importance', ascending=False)
        .head(10)
    )
    top_10_features.to_csv(os.path.join(CV_DIR, "top_10_feature_importance.csv"), index=False)
    print(f"[✓] Saved top 10 features to: {CV_DIR}/top_10_feature_importance.csv")

    # 7. Save metric_train.png (Visualization across Folds)
    plt.figure(figsize=(10, 5))
    cv_only = metrics_df[metrics_df['fold'] != 'Mean']
    plt.plot(cv_only['fold'], cv_only['val_auc'], label='Validation ROC-AUC', marker='o', color='navy')
    plt.plot(cv_only['fold'], cv_only['val_accuracy'], label='Validation Accuracy', marker='s', color='darkorange')
    plt.title('Validation Performance Across Expanding Folds')
    plt.xlabel('Fold Number')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CV_DIR, "metric_train.png"))
    plt.close()
    print(f"[✓] Saved validation plot to: {CV_DIR}/metric_train.png")

    # 8. Train Selected Pipeline on ALL Data & Serialize
    print("\nTraining final pipeline on entire training dataset...")
    final_pipeline = build_pipeline(best_params)
    final_pipeline.fit(X_train, y_train)

    # Save Pickle object (.pkl)
    pkl_path = os.path.join(MODEL_DIR, "selected_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(final_pipeline, f)
    print(f"[✓] Saved fitted model pipeline to: {pkl_path}")

    # Save Summary (.txt)
    txt_path = os.path.join(MODEL_DIR, "selected_model.txt")
    with open(txt_path, "w") as f:
        f.write("Selected Model Pipeline & Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Classifier Family: XGBClassifier\n")
        f.write(f"Hyperparameters: {best_params}\n\n")
        f.write("Cross-Validation Metrics (10 Folds Mean):\n")
        f.write(f"  - Mean Validation Accuracy: {mean_row['val_accuracy']:.4f}\n")
        f.write(f"  - Mean Validation ROC-AUC:  {mean_row['val_auc']:.4f}\n")
        f.write(f"  - Mean Validation LogLoss:  {mean_row['val_logloss']:.4f}\n\n")
        f.write("Top 10 Most Important Features Across Folds:\n")
        for idx, row in top_10_features.iterrows():
            f.write(f"  - {row['feature']}: {row['importance']:.4f}\n")

    print(f"[✓] Saved human-readable summary to: {txt_path}\n")


if __name__ == '__main__':
    run_model_selection()