# src/models/predict_clv.py

import os
import joblib
import pandas as pd
import numpy as np

from src.utils.io import PROCESSED_DATA_DIR


def main():
    rfm_path = os.path.join(PROCESSED_DATA_DIR, "retail_rfm.parquet")
    rfm = pd.read_parquet(rfm_path)

    model = joblib.load(os.path.join(PROCESSED_DATA_DIR, "clv_model.pkl"))

    X = rfm[["Recency", "Frequency", "Monetary"]]
    log_pred = model.predict(X)

    rfm["CLV"] = np.expm1(log_pred)

    # Save
    save_path = os.path.join(PROCESSED_DATA_DIR, "retail_clv.parquet")
    rfm.to_parquet(save_path, index=False)

    print("Saved CLV predictions to:", save_path)


if __name__ == "__main__":
    main()
