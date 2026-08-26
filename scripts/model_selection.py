from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from features_engineering import main
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')








# def load_best_hyperparameters():
  
#     default_params = {
#         'n_estimators': 100,
#         'max_depth': 3,
#         'learning_rate': 0.01
#     }


    
#     return default_params


# def build_pipeline(params=None):
   
    
#     return Pipeline([
#         ('scaler', StandardScaler()),
#         ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', **params))
#     ])


# def evaluate_single_fold(fold, pipeline, X_tr, y_tr, X_val, y_val, feature_cols):
    
#     pipeline.fit(X_tr, y_tr)

#     tr_preds = pipeline.predict(X_tr)
#     tr_probs = pipeline.predict_proba(X_tr)[:, 1]

#     val_preds = pipeline.predict(X_val)
#     val_probs = pipeline.predict_proba(X_val)[:, 1]

#     fold_metrics = {
#         'fold': fold + 1,
#         'train_accuracy': accuracy_score(y_tr, tr_preds),
#         'val_accuracy': accuracy_score(y_val, val_preds),
#         'train_auc': roc_auc_score(y_tr, tr_probs),
#         'val_auc': roc_auc_score(y_val, val_probs),
#         'train_logloss': log_loss(y_tr, tr_probs),
#         'val_logloss': log_loss(y_val, val_probs)
#     }

#     clf = pipeline.named_steps['classifier']
#     fold_importance = pd.DataFrame({
#         'fold': fold + 1,
#         'feature': feature_cols,
#         'importance': clf.feature_importances_
#     }).sort_values('importance', ascending=False)

#     return fold_metrics, fold_importance


# def run_cross_validation(df_train, X_train, y_train, feature_cols, best_params, n_splits=10):
#     cv_splits = create_constrained_time_series_splits(df_train, n_splits=n_splits, min_train_years=2.0)

#     metrics_list = []
#     feature_importances_list = []

    
#     for fold, (train_idx, val_idx) in enumerate(cv_splits):
#         X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#         pipeline = build_pipeline(best_params)
#         fold_metrics, fold_importance = evaluate_single_fold(
#             fold, pipeline, X_tr, y_tr, X_val, y_val, feature_cols
#         )

#         metrics_list.append(fold_metrics)
#         feature_importances_list.append(fold_importance)

#     return metrics_list, feature_importances_list


# def process_and_save_metrics(metrics_list):
#     metrics_df = pd.DataFrame(metrics_list)
#     mean_row = metrics_df.mean(numeric_only=True).to_dict()
#     mean_row['fold'] = 'Mean'
#     metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_row])], ignore_index=True)

#     csv_path = os.path.join(CV_DIR, "ml_metrics_train.csv")
#     metrics_df.to_csv(csv_path, index=False)
#     print(f"[✓] Saved metrics table to: {csv_path}")

#     return metrics_df, mean_row


# def process_and_save_feature_importances(feature_importances_list):
#     all_importances = pd.concat(feature_importances_list, ignore_index=True)
#     top_10_features = (
#         all_importances.groupby('feature')['importance']
#         .mean()
#         .reset_index()
#         .sort_values('importance', ascending=False)
#         .head(10)
#     )

#     csv_path = os.path.join(CV_DIR, "top_10_feature_importance.csv")
#     top_10_features.to_csv(csv_path, index=False)
#     print(f"[✓] Saved top 10 features to: {csv_path}")

#     return top_10_features


# def save_validation_plot(metrics_df):
  
#     plt.figure(figsize=(11, 5))
    
#     # Exclude aggregate 'Mean' row
#     cv_only = metrics_df[metrics_df['fold'] != 'Mean'].copy()
#     folds = cv_only['fold'].astype(int).tolist()
    
#     # Positioning for side-by-side bars per fold
#     x = np.arange(len(folds))
#     bar_width = 0.35

#     # Plot AUC for Train and Validation sets
#     plt.bar(x - bar_width/2, cv_only['train_auc'], width=bar_width, label='Train AUC', color='#1f77b4', alpha=0.9)
#     plt.bar(x + bar_width/2, cv_only['val_auc'], width=bar_width, label='Validation AUC', color='#ff7f0e', alpha=0.9)

#     # Chart details
#     plt.title('auc on train and validation set on all folds of the train set ', fontsize=13, pad=12)
#     plt.xlabel('Fold Number', fontsize=11)
#     plt.ylabel('AUC Score', fontsize=11)
    
#     plt.xticks(x, [f'Fold {f}' for f in folds])
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.legend(loc='upper left', frameon=True)
#     plt.tight_layout()

#     # Save output image
#     plot_path = os.path.join(CV_DIR, "metric_train.png")
#     plt.savefig(plot_path, dpi=300)
#     plt.close()
#     print(f"[✓] Saved AUC fold chart to: {plot_path}")


# def train_and_save_final_model(X_train, y_train, best_params, mean_row, top_10_features):
#     """Trains the final model on the entire dataset and serializes artifacts."""
#     print("\nTraining final pipeline on entire training dataset...")
#     final_pipeline = build_pipeline(best_params)
#     final_pipeline.fifrom sklearn.preprocessing import StandardScalert(X_train, y_train)

#     # Serialize object
#     pkl_path = os.path.join(MODEL_DIR, "selected_model.pkl")
#     with open(pkl_path, "wb") as f:
#         pickle.dump(final_pipeline, f)
#     print(f"[✓] Saved fitted model pipeline to: {pkl_path}")

#     # Write textual summary
#     txt_path = os.path.join(MODEL_DIR, "selected_model.txt")
#     with open(txt_path, "w") as f:
#         f.write("Selected Model Pipeline & Summary\n")
#         f.write("=" * 40 + "\n\n")
#         f.write("Classifier Family: XGBClassifier\n")
#         f.write(f"Hyperparameters: {best_params}\n\n")
#         f.write("Cross-Validation Metrics (10 Folds Mean):\n")
#         f.write(f"  - Mean Validation Accuracy: {mean_row['val_accuracy']:.4f}\n")
#         f.write(f"  - Mean Validation ROC-AUC:  {mean_row['val_auc']:.4f}\n")
#         f.write(f"  - Mean Validation LogLoss:  {mean_row['val_logloss']:.4f}\n\n")
#         f.write("Top 10 Most Important Features Across Folds:\n")
#         for _, row in top_10_features.iterrows():
#             f.write(f"  - {row['feature']}: {row['importance']:.4f}\n")

#     print(f"[✓] Saved human-readable summary to: {txt_path}\n")
# def run_model_selection():
#     os.makedirs(CV_DIR, exist_ok=True)
#     os.makedirs(MODEL_DIR, exist_ok=True)


#     df_train, _, X_train, y_train, feature_cols = load_and_prepare_data()
#     best_params = load_best_hyperparameters()

#     metrics_list, feature_importances_list = run_cross_validation(
#         df_train, X_train, y_train, feature_cols, best_params
#     )

#     metrics_df, mean_row = process_and_save_metrics(metrics_list)
#     top_10_features = process_and_save_feature_importances(feature_importances_list)

#     save_validation_plot(metrics_df)
#     train_and_save_final_model(X_train, y_train, best_params, mean_row, top_10_features)
def load_and_prepare_data():
    df_train, df_test = main()

    non_feature_cols = [ "target", "Name", "date","forward_return" ]
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]

    X_train = df_train[feature_cols]
    y_train = (df_train["target"] > 0).astype(int)

    return  X_train, y_train
def candidate_models():
    return [
        (
            "logreg",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, solver="liblinear", random_state=42))
            ])
        ),
        (
            "ridge",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", CalibratedClassifierCV(estimator=RidgeClassifier()))
            ])
        ),
        (
            "lgbm",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMClassifier(random_state=42, verbose=-1))
            ])
        ),
        (
            "xgb",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(random_state=42, eval_metric="logloss"))
            ])
        ),
        (
            "catboost",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", CatBoostClassifier(random_state=42, verbose=0))
            ])
        ),
        (
            "rf",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
            ])
        ),
    ]
def run_model_selection():


    X_train, y_train = load_and_prepare_data()
    best_score = -1.0
    best_model = None
    best_name = ""
    print("let's start the contest a hmadi")
    for name , model in candidate_models() :
        model.fit(X_train, y_train)
        score = model.score(X_train, y_train)
        print(f"the {name}, finished with the score of {score}")
        if score > best_score :
            best_score = score 
            best_model =model
            best_name = name
    print(f"after severe batlle {best_name} win with the score of {best_score} mabrooook")

    return best_model 


   


if __name__ == '__main__':
    model = run_model_selection()
    print(model)