import json
import uuid
import numpy as np
import pandas as pd
from datetime import timedelta
from src.common.snowflake import read_sql_df, exec_sql, write_df

MODEL_VERSION = "baseline_v1"

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def baseline_predict(train_df, horizon_dates):
    """
    Predict each future date as average of last 7 observed Y values in train_df.
    """
    last7 = train_df["Y"].tail(7).values
    pred = float(np.mean(last7))
    return pd.DataFrame({"DS": horizon_dates, "YHAT": pred})

def rolling_backtest(df, horizon=14):
    df = df.sort_values("DS").reset_index(drop=True)

    # start after we have enough history
    start_idx = 60 if len(df) > 120 else max(30, horizon + 7)
    metrics_rows = []

    cut = start_idx
    while cut + horizon <= len(df):
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:cut + horizon].copy()

        horizon_dates = test["DS"].tolist()
        preds = baseline_predict(train, horizon_dates)

        y_true = test["Y"].values
        y_pred = preds["YHAT"].values

        metrics_rows.append({
            "WINDOW_START": test["DS"].min(),
            "WINDOW_END": test["DS"].max(),
            "MAE": mae(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        })

        cut += horizon

    return pd.DataFrame(metrics_rows)

def main():
    # Pull features (we only need DS, Y for baseline)
    sql = """
    SELECT SERIES_ID, DS, Y
    FROM FEATURES
    WHERE SERIES_ID = 'GLOBAL'
    ORDER BY DS
    """
    df = read_sql_df(sql)
    df["DS"] = pd.to_datetime(df["DS"]).dt.date

    print("Rows:", len(df))
    print(df.head())

    run_id = str(uuid.uuid4())

    bt = rolling_backtest(df, horizon=14)
    bt["RUN_ID"] = run_id
    bt["SERIES_ID"] = "GLOBAL"
    bt["MODEL_VERSION"] = MODEL_VERSION

    # Save model artifact (tiny)
    artifact = {
        "model_version": MODEL_VERSION,
        "type": "baseline_last7_mean",
        "notes": "Predicts each future day as mean of last 7 observed days.",
    }
    with open(f"models/{MODEL_VERSION}.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    # Save metrics artifact (summary)
    summary = {
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "avg_mae": float(bt["MAE"].mean()),
        "avg_mape": float(bt["MAPE"].mean()),
        "n_windows": int(len(bt)),
    }
    with open("models/metrics_baseline_v1.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write to Snowflake METRICS_MONITORING (append)
    out = bt[["RUN_ID","SERIES_ID","WINDOW_START","WINDOW_END","MAE","MAPE","MODEL_VERSION"]].copy()
    success, nchunks, nrows, _ = write_df(out, "METRICS_MONITORING")
    print(f"Snowflake metrics write -> success={success}, rows={nrows}")

    print("✅ Baseline backtest complete.")
    print(summary)

if __name__ == "__main__":
    main()
