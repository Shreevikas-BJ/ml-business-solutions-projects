# src/models/train_churn.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from src.utils.io import PROCESSED_DATA_DIR
from src.utils.preprocessing import build_telco_preprocessor


def precision_at_k(y_true, y_scores, k: float = 0.2) -> float:
    """
    Precision@k: take the top k proportion of users by predicted score,
    compute precision in that group.
    k is a fraction (0 < k <= 1).
    """
    n = len(y_scores)
    top_k = int(np.ceil(n * k))
    # Get indices of sorted scores (descending)
    idx = np.argsort(-y_scores)[:top_k]
    return precision_score(y_true[idx], np.ones_like(y_true[idx]))


def load_clean_telco() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_DIR, "telco_clean.parquet")
    return pd.read_parquet(path)


def train_logistic(X_train, y_train, preprocessor):
    # class_weight="balanced" helps with churn imbalance
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1
    )
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train, preprocessor):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(name, model, X_test, y_test):
    # Get predicted probabilities for the positive class
    y_proba = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    p_at_20 = precision_at_k(y_test.to_numpy(), y_proba, k=0.2)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC       : {roc:.3f}")
    print(f"PR-AUC        : {pr_auc:.3f}")
    print(f"Precision@20% : {p_at_20:.3f}")

    return {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "precision_at_20": p_at_20,
    }


def main():
    df = load_clean_telco()

    # Build preprocessor and split
    from src.utils.preprocessing import build_telco_preprocessor

    X, y, preprocessor, feat_names = build_telco_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Train models
    logit_pipe = train_logistic(X_train, y_train, preprocessor)
    xgb_pipe = train_xgboost(X_train, y_train, preprocessor)

    # Evaluate
    metrics_logit = evaluate_model("Logistic Regression", logit_pipe, X_test, y_test)
    metrics_xgb = evaluate_model("XGBoost", xgb_pipe, X_test, y_test)

    # Pick best model by PR-AUC (or ROC-AUC, your choice)
    best_model_name = (
        "xgb" if metrics_xgb["pr_auc"] > metrics_logit["pr_auc"] else "logit"
    )
    best_model = xgb_pipe if best_model_name == "xgb" else logit_pipe

    print(f"\nBest model: {best_model_name.upper()}")

    # Save best model
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    model_path = os.path.join(PROCESSED_DATA_DIR, "churn_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Saved best churn model to: {model_path}")


if __name__ == "__main__":
    main()
