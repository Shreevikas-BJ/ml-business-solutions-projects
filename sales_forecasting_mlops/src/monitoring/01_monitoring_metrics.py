import uuid
import numpy as np
import pandas as pd
from datetime import timedelta
from src.common.snowflake import read_sql_df, write_df

MONITOR_VERSION = "monitor_v1"
SERIES_ID = "GLOBAL"
MODEL_TO_MONITOR = "xgb_v1"  # change to baseline_v1 if you want

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def main():
    run_id = str(uuid.uuid4())

    # Actuals from FEATURES
    actuals = read_sql_df(f"""
    SELECT DS, Y
    FROM FEATURES
    WHERE SERIES_ID='{SERIES_ID}'
    ORDER BY DS
    """)
    actuals["DS"] = pd.to_datetime(actuals["DS"]).dt.date

    # Forecasts from FORECASTS (choose model)
    fcst = read_sql_df(f"""
    SELECT DS, YHAT
    FROM FORECASTS
    WHERE SERIES_ID='{SERIES_ID}' AND MODEL_VERSION='{MODEL_TO_MONITOR}'
    ORDER BY DS
    """)
    fcst["DS"] = pd.to_datetime(fcst["DS"]).dt.date

    # Join on dates where both exist (only possible if you forecasted in the past and now have actuals)
    merged = pd.merge(actuals, fcst, on="DS", how="inner")
    if merged.empty:
        print("No overlapping actuals vs forecasts yet. Monitoring will log 0 rows now.")
        return

    # Compute recent vs older performance windows (simple drift)
    merged = merged.sort_values("DS")
    recent_days = 14
    if len(merged) < recent_days * 2:
        # Not enough overlap; compute single window
        recent = merged
        older = merged
    else:
        recent = merged.tail(recent_days)
        older = merged.head(len(merged) - recent_days)

    recent_mae = mae(recent["Y"].values, recent["YHAT"].values)
    recent_mape = mape(recent["Y"].values, recent["YHAT"].values)

    older_mae = mae(older["Y"].values, older["YHAT"].values)
    older_mape = mape(older["Y"].values, older["YHAT"].values)

    # Drift ratio ( >1 means worse recently )
    mae_drift_ratio = float(recent_mae / (older_mae + 1e-9))
    mape_drift_ratio = float(recent_mape / (older_mape + 1e-9))

    rows = [{
        "RUN_ID": run_id,
        "SERIES_ID": SERIES_ID,
        "WINDOW_START": recent["DS"].min(),
        "WINDOW_END": recent["DS"].max(),
        "MAE": recent_mae,
        "MAPE": recent_mape,
        "MODEL_VERSION": MONITOR_VERSION
    }]

    out = pd.DataFrame(rows)
    success, nchunks, nrows, _ = write_df(out, "METRICS_MONITORING")
    print(f"Snowflake monitoring write -> success={success}, rows={nrows}")

    print("✅ Monitoring snapshot logged.")
    print({
        "recent_mae": recent_mae,
        "older_mae": older_mae,
        "mae_drift_ratio": mae_drift_ratio,
        "recent_mape": recent_mape,
        "older_mape": older_mape,
        "mape_drift_ratio": mape_drift_ratio
    })

if __name__ == "__main__":
    main()
