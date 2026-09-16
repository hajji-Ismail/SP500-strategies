from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from features_engineering import main
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
def load_and_prepare_data():
    df_train, _ = main()
    non_feature_cols = [ "target", "Name", "date","forward_return", "high","low", "open", "close", "volume" ]
    feature_cols = [c for c in df_train.columns if c not in non_feature_cols]
    X_train = df_train[feature_cols]
    y_train = df_train["target"] 
    return  X_train, y_train
def candidate_models():
    return [
        (
            "logreg",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
            ])
        ),
 
        (
            "lgbm",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LGBMClassifier(random_state=42, verbose=-1, objective="multiclass", num_class=3))
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