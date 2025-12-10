# src/models/train_clv.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.utils.io import PROCESSED_DATA_DIR


def load_rfm():
    path = os.path.join(PROCESSED_DATA_DIR, "retail_rfm.parquet")
    return pd.read_parquet(path)


def main():
    rfm = load_rfm()

    # Basic log transform helps stability
    rfm["log_Monetary"] = rfm["Monetary"].apply(lambda x: np.log1p(x))

    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["log_Monetary"]  # future value proxy

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    print("MAE:", mae)

    # Save model
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    model_path = os.path.join(PROCESSED_DATA_DIR, "clv_model.pkl")
    joblib.dump(model, model_path)

    print("Saved CLV model to:", model_path)


if __name__ == "__main__":
    import numpy as np
    main()
