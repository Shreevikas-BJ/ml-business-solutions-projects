import json
import numpy as np
import pandas as pd
import xgboost as xgb
from src.common.snowflake import read_sql_df

MODEL_VERSION = "xgb_v1"
FEATURE_COLS = ["LAG_1","LAG_7","ROLL7_MEAN","ROLL28_MEAN","DOW","IS_WEEKEND"]

def main():
    sql = """
    SELECT SERIES_ID, DS, Y, LAG_1, LAG_7, ROLL7_MEAN, ROLL28_MEAN, DOW, IS_WEEKEND
    FROM FEATURES
    WHERE SERIES_ID='GLOBAL'
    ORDER BY DS
    """
    df = read_sql_df(sql)
    df["DS"] = pd.to_datetime(df["DS"])
    df["IS_WEEKEND"] = df["IS_WEEKEND"].astype(int)

    # Simple time split (train/valid) for initial fit
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    valid_df = df.iloc[split:].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Y"].values
    X_valid = valid_df[FEATURE_COLS].values
    y_valid = valid_df["Y"].values

    model = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # Save model artifact
    model.save_model(f"models/{MODEL_VERSION}.json")

    meta = {
        "model_version": MODEL_VERSION,
        "features": FEATURE_COLS,
        "params": model.get_params(),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
    }
    with open(f"models/{MODEL_VERSION}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ XGBoost trained and saved:", f"models/{MODEL_VERSION}.json")
    print(meta)

if __name__ == "__main__":
    main()
