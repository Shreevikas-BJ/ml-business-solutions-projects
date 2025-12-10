# src/models/train_uplift.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.utils.io import PROCESSED_DATA_DIR


def load_criteo():
    path = os.path.join(PROCESSED_DATA_DIR, "criteo_small.parquet")
    return pd.read_parquet(path)


def two_model_uplift(X_train, y_train, treatment_train):
    """
    2-model uplift:
    Model A: P(y=1 | T=1)
    Model B: P(y=1 | T=0)
    """
    model_treated = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    )
    model_control = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    )

    model_treated.fit(
        X_train[treatment_train == 1], y_train[treatment_train == 1]
    )
    model_control.fit(
        X_train[treatment_train == 0], y_train[treatment_train == 0]
    )

    return model_treated, model_control


def main():
    df = load_criteo()

    # Split features and target
    y = df["conversion"]
    T = df["treatment"]
    X = df.drop(columns=["conversion", "treatment"])

    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X, y, T, stratify=y, test_size=0.2, random_state=42
    )

    print("Training 2-model uplift...")
    model_treated, model_control = two_model_uplift(X_train, y_train, T_train)

    # Save both models
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    joblib.dump(
        model_treated,
        os.path.join(PROCESSED_DATA_DIR, "uplift_model_treated.pkl"),
    )
    joblib.dump(
        model_control,
        os.path.join(PROCESSED_DATA_DIR, "uplift_model_control.pkl"),
    )

    print("Saved uplift models.")
    print("Use predict_uplift.py to score uplift.")


if __name__ == "__main__":
    main()
