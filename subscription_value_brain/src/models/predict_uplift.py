# src/models/predict_uplift.py

import os
import joblib
import numpy as np
import pandas as pd

from src.utils.io import PROCESSED_DATA_DIR


def classify_uplift(uplift):
    """
    Rule-based segmentation used in many marketing teams:
    """
    if uplift > 0.05:
        return "Persuadable"          # good target
    elif uplift > 0:
        return "Sure Thing"
    elif uplift < -0.05:
        return "Do-Not-Disturb"       # negatively affected
    else:
        return "Lost Cause"


def main():
    df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "criteo_small.parquet"))
    
    X = df.drop(columns=["conversion", "treatment"])

    model_treat = joblib.load(
        os.path.join(PROCESSED_DATA_DIR, "uplift_model_treated.pkl")
    )
    model_control = joblib.load(
        os.path.join(PROCESSED_DATA_DIR, "uplift_model_control.pkl")
    )

    # Predict P1 and P0
    P1 = model_treat.predict_proba(X)[:, 1]
    P0 = model_control.predict_proba(X)[:, 1]

    uplift = P1 - P0

    df["uplift"] = uplift
    df["uplift_segment"] = df["uplift"].apply(classify_uplift)

    # Save
    output = os.path.join(PROCESSED_DATA_DIR, "criteo_uplift_scores.parquet")
    df.to_parquet(output, index=False)

    print("Saved uplift scores to:", output)


if __name__ == "__main__":
    main()
