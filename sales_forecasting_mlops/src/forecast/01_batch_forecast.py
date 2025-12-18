import uuid
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
from src.common.snowflake import read_sql_df, exec_sql, write_df

XGB_VERSION = "xgb_v1"
BASELINE_VERSION = "baseline_v1"

FEATURE_COLS = ["LAG_1","LAG_7","ROLL7_MEAN","ROLL28_MEAN","DOW","IS_WEEKEND"]

def make_next_dates(last_date, n_days: int):
    return [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

def dow(d):  # 1=Sun..7=Sat (match Spark dayofweek)
    # Python: Monday=0..Sunday=6
    # Convert to Spark style: Sunday=1 ... Saturday=7
    py = d.weekday()
    return 2 if py == 0 else 3 if py == 1 else 4 if py == 2 else 5 if py == 3 else 6 if py == 4 else 7 if py == 5 else 1

def is_weekend_spark_style(dow_val):
    return int(dow_val in [1, 7])

def baseline_forecast(history_y, n_days):
    pred = float(np.mean(history_y[-7:]))
    return np.array([pred] * n_days)

def recursive_xgb_forecast(model, hist_df, n_days):
    """
    Recursive forecasting:
    - predict day t+1 using last known features
    - append predicted Y to history
    - recompute needed lags/rolls for next day, repeat
    """
    hist = hist_df.copy()
    hist = hist.sort_values("DS").reset_index(drop=True)

    preds = []
    last_date = hist["DS"].iloc[-1]

    for i in range(n_days):
        next_day = last_date + timedelta(days=1)

        # Build features from history (using last available rows)
        y_series = hist["Y"].values

        lag1 = float(y_series[-1])
        lag7 = float(y_series[-7]) if len(y_series) >= 7 else float(np.mean(y_series))
        roll7 = float(np.mean(y_series[-7:]))
        roll28 = float(np.mean(y_series[-28:])) if len(y_series) >= 28 else float(np.mean(y_series))

        dow_val = dow(next_day)
        wknd = is_weekend_spark_style(dow_val)

        X = np.array([[lag1, lag7, roll7, roll28, dow_val, wknd]], dtype=float)
        yhat = float(model.predict(X)[0])
        preds.append((next_day, yhat))

        # append predicted value to history for the next step
        hist = pd.concat([hist, pd.DataFrame([{"DS": next_day, "Y": yhat}])], ignore_index=True)
        last_date = next_day

    return preds

def main(n_days=14):
    # Pull latest history from FEATURES (already clean & daily)
    df = read_sql_df("""
    SELECT DS, Y
    FROM FEATURES
    WHERE SERIES_ID='GLOBAL'
    ORDER BY DS
    """)
    df["DS"] = pd.to_datetime(df["DS"]).dt.date

    last_date = df["DS"].iloc[-1]
    next_dates = make_next_dates(last_date, n_days)

    run_id = str(uuid.uuid4())

    # -------- Baseline forecast --------
    base_preds = baseline_forecast(df["Y"].values, n_days)
    base_out = pd.DataFrame({
        "SERIES_ID": "GLOBAL",
        "DS": next_dates,
        "YHAT": base_preds,
        "MODEL_VERSION": BASELINE_VERSION
    })

    # -------- XGBoost forecast --------
    model = xgb.XGBRegressor()
    model.load_model(f"models/{XGB_VERSION}.json")

    xgb_preds = recursive_xgb_forecast(model, df[["DS","Y"]], n_days)
    xgb_out = pd.DataFrame({
        "SERIES_ID": "GLOBAL",
        "DS": [d for d, _ in xgb_preds],
        "YHAT": [y for _, y in xgb_preds],
        "MODEL_VERSION": XGB_VERSION
    })

    # Combine + write to Snowflake (append)
    out = pd.concat([base_out, xgb_out], ignore_index=True)

    success, nchunks, nrows, _ = write_df(out, "FORECASTS")
    print(f"Snowflake forecasts write -> success={success}, rows={nrows}")
    print("✅ Batch forecast complete.")
    print(out.tail(6))

if __name__ == "__main__":
    main(n_days=14)
