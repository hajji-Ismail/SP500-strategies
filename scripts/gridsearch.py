import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd 
from features_engineering import main
from model_selection import run_model_selection
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
MODEL_DIR = "results/selected-model"
CV_DIR = "results/cross-validation"


def time_series_splits(df, n_splits=10, min_train_years=2.0):
   
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

def evaluate_single_fold(fold, pipeline, X_tr, y_tr, X_val, y_val, feature_cols):
    
    pipeline.fit(X_tr, y_tr)

    tr_preds = pipeline.predict(X_tr)
    tr_probs = pipeline.predict_proba(X_tr)[:, 1]

    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    fold_metrics = {
        'fold': fold + 1,
        'train_accuracy': accuracy_score(y_tr, tr_preds),
        'val_accuracy': accuracy_score(y_val, val_preds),
        'train_auc': roc_auc_score(y_tr, tr_probs),
        'val_auc': roc_auc_score(y_val, val_probs),
        'train_logloss': log_loss(y_tr, tr_probs),
        'val_logloss': log_loss(y_val, val_probs)
    }

    clf = pipeline.named_steps['model']
    fold_importance = pd.DataFrame({
        'fold': fold + 1,
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    return fold_metrics, fold_importance
def run_cross_validation(df_train, X_train, y_train, feature_cols, pipeline, n_splits=10):
    cv_splits = time_series_splits(df_train, n_splits=n_splits, min_train_years=2.0)

    metrics_list = []
    feature_importances_list = []

    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_metrics, fold_importance = evaluate_single_fold(
            fold, pipeline, X_tr, y_tr, X_val, y_val, feature_cols
        )

        metrics_list.append(fold_metrics)
        feature_importances_list.append(fold_importance)

    return metrics_list, feature_importances_list
def save_cv_plot(df, cv_splits, save_path='results/cross-validation/Time_series_split.png'):
  
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

def process_and_save_metrics(metrics_list):
    metrics_df = pd.DataFrame(metrics_list)
    mean_row = metrics_df.mean(numeric_only=True).to_dict()
    mean_row['fold'] = 'Mean'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

    csv_path = os.path.join(CV_DIR, "ml_metrics_train.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved metrics table to: {csv_path}")

    return metrics_df, mean_row


def process_and_save_feature_importances(feature_importances_list):
    all_importances = pd.concat(feature_importances_list, ignore_index=True)
    top_10_features = (
        all_importances.groupby('feature')['importance']
        .mean()
        .reset_index()
        .sort_values('importance', ascending=False)
        .head(10)
    )

    csv_path = os.path.join(CV_DIR, "top_10_feature_importance.csv")
    top_10_features.to_csv(csv_path, index=False)
    print(f"[✓] Saved top 10 features to: {csv_path}")

    return top_10_features


def save_validation_plot(metrics_df):
  
    plt.figure(figsize=(11, 5))
    
    # Exclude aggregate 'Mean' row
    cv_only = metrics_df[metrics_df['fold'] != 'Mean'].copy()
    folds = cv_only['fold'].astype(int).tolist()
    
    # Positioning for side-by-side bars per fold
    x = np.arange(len(folds))
    bar_width = 0.35

    # Plot AUC for Train and Validation sets
    plt.bar(x - bar_width/2, cv_only['train_auc'], width=bar_width, label='Train AUC', color='#1f77b4', alpha=0.9)
    plt.bar(x + bar_width/2, cv_only['val_auc'], width=bar_width, label='Validation AUC', color='#ff7f0e', alpha=0.9)

    # Chart details
    plt.title('auc on train and validation set on all folds of the train set ', fontsize=13, pad=12)
    plt.xlabel('Fold Number', fontsize=11)
    plt.ylabel('AUC Score', fontsize=11)
    
    plt.xticks(x, [f'Fold {f}' for f in folds])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()

    # Save output image
    plot_path = os.path.join(CV_DIR, "metric_train.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Saved AUC fold chart to: {plot_path}")


def run_grid_search():
    os.makedirs(CV_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df_train, _ = main()

    non_feature_cols = ["target", "Name", "date", "forward_return"]
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

    X_train = df_train[feature_cols]
    y_train = df_train["target"]

 
    y_train_binary = (y_train > 0).astype(int)

    model = run_model_selection()

    param_grid = {
        'model__iterations': [100, 200],
        'model__depth': [3, 5],
        'model__learning_rate': [0.01, 0.1],
        'model__l2_leaf_reg': [1, 3, 5]
    }
    cv_splits = time_series_splits(df_train, n_splits=10, min_train_years=2.0)
    save_cv_plot(df_train, cv_splits)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_binary)
    metrics_list, feature_importances_list = run_cross_validation(df_train, X_train, y_train_binary, feature_cols, model)
    metrics_df, mean_row = process_and_save_metrics(metrics_list)
    top_10_features = process_and_save_feature_importances(feature_importances_list)

    save_validation_plot(metrics_df)
    finale_model = grid_search.best_estimator_
    pkl_path = os.path.join(MODEL_DIR, "selected_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(finale_model, f)
    txt_path = os.path.join(MODEL_DIR, "selected_model.txt")
    with open(txt_path, "w") as f:
        f.write("Selected Model Pipeline & Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Classifier Family: XGBClassifier\n")
        f.write(f"Hyperparameters: {model.get_params()}\n\n")
        f.write("Cross-Validation Metrics (10 Folds Mean):\n")
        f.write(f"  - Mean Validation Accuracy: {mean_row['val_accuracy']:.4f}\n")
        f.write(f"  - Mean Validation ROC-AUC:  {mean_row['val_auc']:.4f}\n")
        f.write(f"  - Mean Validation LogLoss:  {mean_row['val_logloss']:.4f}\n\n")
        f.write("Top 10 Most Important Features Across Folds:\n")
        for _, row in top_10_features.iterrows():
            f.write(f"  - {row['feature']}: {row['importance']:.4f}\n")




    


if __name__ == '__main__':
    run_grid_search()