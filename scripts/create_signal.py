import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def generate_signals(data_path="data/processed_data.csv", 
                     model_path="results/selected-model/selected_model.pkl", 
                     output_path="results/selected-model/ml_signal.csv"):
    
    print("--- [Task 4] Starting Signal Generation ---")
    
    # 1. Load Processed Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file non-existent at {data_path}. Run features_engineering.py first.")
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['date', 'ticker'], inplace=True)
    df.sort_index(inplace=True)
    
    # Separate Features (X) and Target (y)
    feature_cols = [c for c in df.columns if c not in ['target', 'forward_return']]
    X = df[feature_cols]
    
    # Split Train (< 2017) and Test (>= 2017)
    train_mask = X.index.get_level_values('date') < '2017-01-01'
    test_mask = ~train_mask
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    
    # Load Model Pipeline
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Selected model non-existent at {model_path}. Run gridsearch.py first.")
        
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    # 2. Generate Out-of-Fold (OOF) Signals on Training Set
    unique_train_dates = X_train.index.get_level_values('date').unique().sort_values()
    tscv = TimeSeriesSplit(n_splits=10)
    
    oof_signal_series = pd.Series(index=X_train.index, dtype=float)
    
    print("Generating Out-of-Fold (OOF) predictions across 10 temporal folds...")
    for fold, (train_date_idx, val_date_idx) in enumerate(tscv.split(unique_train_dates), 1):
        train_dates = unique_train_dates[train_date_idx]
        val_dates = unique_train_dates[val_date_idx]
        
        train_fold_mask = X_train.index.get_level_values('date').isin(train_dates)
        val_fold_mask = X_train.index.get_level_values('date').isin(val_dates)
        
        X_tr_fold = X_train[train_fold_mask]
        y_tr_fold = df.loc[X_tr_fold.index, 'target']
        X_val_fold = X_train[val_fold_mask]
        
        # Fit fold model strictly on fold train set
        pipeline.fit(X_tr_fold, y_tr_fold)
        
        # Predict continuous probabilities P(Target = 1)
        probs = pipeline.predict_proba(X_val_fold)[:, 1]
        oof_signal_series.loc[X_val_fold.index] = probs
        print(f"  Fold {fold}/10 complete: Val dates {val_dates.min().date()} to {val_dates.max().date()}")
        
    # Drop initial unpredicted dates from first training fold
    oof_signal = oof_signal_series.dropna()
    
    # 3. Generate Test Set Signals
    print("Training pipeline on complete train set (< 2017) for test set predictions...")
    y_train_full = df.loc[X_train.index, 'target']
    pipeline.fit(X_train, y_train_full)
    
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_signal = pd.Series(test_probs, index=X_test.index)
    
    # 4. Merge Signals and Export
    full_signal = pd.concat([oof_signal, test_signal]).rename('signal')
    full_signal = full_signal.to_frame()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_signal.to_csv(output_path)
    print(f"--- [Task 4] ML Signal exported to {output_path} ---")

if __name__ == "__main__":
    generate_signals()