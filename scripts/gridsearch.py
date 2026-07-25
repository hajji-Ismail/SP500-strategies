import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features_engineering import main


def create_constrained_time_series_splits(df, n_splits=10, min_train_years=2.0):
    """
    Creates custom time-series splits for a MultiIndex DataFrame indexed by ('date', 'Name').
    """
    # 1. Get unique sorted dates from the MultiIndex level 'date'
    dates_level = df.index.get_level_values("date")
    unique_dates = dates_level.unique().sort_values().values
    total_dates = len(unique_dates)

    # 2. Estimate trading days per year (~252 days/year)
    min_train_days = int(min_train_years * 252)

    # Check remaining dates for 10 validation splits
    remaining_dates = total_dates - min_train_days
    val_size = remaining_dates // n_splits

    splits = []
    for i in range(n_splits):
        val_start_idx = min_train_days + (i * val_size)
        val_end_idx = val_start_idx + val_size if i < n_splits - 1 else total_dates

        train_dates = unique_dates[0:val_start_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]

        # Match dates against the 'date' level of MultiIndex
        train_indices = np.where(dates_level.isin(train_dates))[0]
        val_indices = np.where(dates_level.isin(val_dates))[0]

        splits.append((train_indices, val_indices))

    return splits


def save_cv_plot(df, cv_splits, save_path='results/cross-validation/Time_series_split.png'):
    """
    Generates a horizontal time-series split chart displaying scatter dots 
    alongside a connecting solid line for both Train and Validation sets.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    dates_level = df.index.get_level_values("date")

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        train_dates = dates_level[train_idx]
        val_dates = dates_level[val_idx]

        # 1. Solid horizontal lines spanning the min to max date range
        ax.plot([train_dates.min(), train_dates.max()], [fold, fold], 
                color='navy', linewidth=2.5, alpha=0.8)
        ax.plot([val_dates.min(), val_dates.max()], [fold, fold], 
                color='darkorange', linewidth=2.5, alpha=0.8)

        # 2. Original scatter dots rendered along the line
        ax.scatter(train_dates, [fold] * len(train_dates), 
                   c='navy', s=2, alpha=0.6, label='Train' if fold == 0 else "")
        ax.scatter(val_dates, [fold] * len(val_dates), 
                   c='darkorange', s=2, alpha=0.6, label='Validation' if fold == 0 else "")

    # Formatting and axis setup
    ax.set_yticks(range(len(cv_splits)))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(len(cv_splits))])
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Folds', fontsize=11)
    ax.set_title('Expanding Time Series Split (Initial Train > 2 Years)', fontsize=13, pad=12)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Saved split plot to: {save_path}")


def run_grid_search():
    # 1. Load engineered datasets
    df_train, df_test = main()

    # 2. Extract feature columns and target
    non_feature_cols = ["open", "high", "low", "close", "volume", "target", "Name", "date"]
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

    X_train = df_train[feature_cols]
    y_train = df_train["target"]

    # Convert continuous target to binary classification classes {-1, 1} -> {0, 1} if required by XGBoost
    # np.sign gives -1, 0, 1. For binary classification, map >0 to 1 else 0
    y_train_binary = (y_train > 0).astype(int)

    # 3. Pipeline setup
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    # 4. Hyperparameter Grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1]
    }

    # 5. Create splits & save CV visualization
    cv_splits = create_constrained_time_series_splits(df_train, n_splits=10, min_train_years=2.0)
    save_cv_plot(df_train, cv_splits)

    # 6. Grid Search Execution
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_binary)

    print(f"\nBest ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(grid_search)

    return grid_search


if __name__ == '__main__':
    run_grid_search()