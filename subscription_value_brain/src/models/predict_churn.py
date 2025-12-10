# src/models/predict_churn.py

import os
import joblib
import pandas as pd
import numpy as np

from src.utils.io import PROCESSED_DATA_DIR


def main():
    # 1. Load cleaned Telco data
    telco_path = os.path.join(PROCESSED_DATA_DIR, "telco_clean.parquet")
    df = pd.read_parquet(telco_path)

    # 2. Load churn model (pipeline with preprocessor + model)
    model_path = os.path.join(PROCESSED_DATA_DIR, "churn_model.pkl")
    model = joblib.load(model_path)

    # 3. Prepare features (drop target if present)
    X = df.copy()
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])

    # 4. Predict probability of churn
    p_churn = model.predict_proba(X)[:, 1]

    # 5. Build churn scores table
    churn_scores = pd.DataFrame({
        "p_churn": p_churn
    })

    # Optional: keep an index column for debugging
    churn_scores["telco_index"] = np.arange(len(churn_scores))

    # 6. Save
    output_path = os.path.join(PROCESSED_DATA_DIR, "telco_churn_scores.parquet")
    churn_scores.to_parquet(output_path, index=False)
    print(f"Saved churn scores to: {output_path}")


if __name__ == "__main__":
    main()
