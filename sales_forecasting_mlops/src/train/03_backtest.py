import uuid
import numpy as np
import pandas as pd
import xgboost as xgb
from src.common.snowflake import read_sql_df, write_df

MODEL_VERSION = "xgb_v1"
FEATURE_COLS = ["LAG_1","LAG_7","ROLL7_MEAN","ROLL28_MEAN","DOW","IS_WEEKEND"]

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def main():
    df = read_sql_df("""
    SELECT SERIES_ID, DS, Y, LAG_1, LAG_7, ROLL7_MEAN, ROLL28_MEAN, DOW, IS_WEEKEND
    FROM FEATURES
    WHERE SERIES_ID='GLOBAL'
    ORDER BY DS
    """)
    df["DS"] = pd.to_datetime(df["DS"]).dt.date
    df["IS_WEEKEND"] = df["IS_WEEKEND"].astype(int)

    run_id = str(uuid.uuid4())
    horizon = 14

    # load trained model
    model = xgb.XGBRegressor()
    model.load_model(f"models/{MODEL_VERSION}.json")

    start_idx = 60 if len(df) > 120 else max(30, horizon + 7)
    cut = start_idx
    rows = []

    while cut + horizon <= len(df):
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:cut + horizon].copy()

        X_test = test[FEATURE_COLS].values
        y_pred = model.predict(X_test)

        rows.append({
            "RUN_ID": run_id,
            "SERIES_ID": "GLOBAL",
            "WINDOW_START": test["DS"].min(),
            "WINDOW_END": test["DS"].max(),
            "MAE": mae(test["Y"].values, y_pred),
            "MAPE": mape(test["Y"].values, y_pred),
            "MODEL_VERSION": MODEL_VERSION
        })

        cut += horizon

    out = pd.DataFrame(rows)
    success, nchunks, nrows, _ = write_df(out, "METRICS_MONITORING")
    print(f"Snowflake metrics write -> success={success}, rows={nrows}")

    print("✅ XGBoost backtest complete.")
    print(out.describe())

if __name__ == "__main__":
    main()
