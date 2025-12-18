import pandas as pd
import streamlit as st
from datetime import date, timedelta
import sys
import os

# Add project root to PYTHONPATH so `src` is discoverable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.common.snowflake import read_sql_df

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📈 Sales Forecasting MLOps Dashboard (PySpark + Snowflake + XGBoost)")
st.caption("Reads FEATURES / FORECASTS / METRICS_MONITORING from Snowflake")

# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

series_id = st.sidebar.text_input("SERIES_ID", value="GLOBAL")
days_history = st.sidebar.slider("History window (days)", min_value=30, max_value=365, value=180, step=30)
future_days = st.sidebar.slider("Future forecast window (days)", min_value=7, max_value=60, value=14, step=7)

show_baseline = st.sidebar.checkbox("Show baseline forecasts", value=True)
show_xgb = st.sidebar.checkbox("Show XGBoost forecasts", value=True)

# Models (must match your pipeline model_version strings)
BASELINE_VERSION = "baseline_v1"
XGB_VERSION = "xgb_v1"
MONITOR_VERSION = "monitor_v1"


@st.cache_data(ttl=60)
def load_actuals(series_id: str) -> pd.DataFrame:
    sql = f"""
    SELECT DS, Y
    FROM FEATURES
    WHERE SERIES_ID = '{series_id}'
    ORDER BY DS
    """
    df = read_sql_df(sql)
    df["DS"] = pd.to_datetime(df["DS"])
    return df


@st.cache_data(ttl=60)
def load_forecasts(series_id: str, model_version: str) -> pd.DataFrame:
    sql = f"""
    SELECT DS, YHAT, MODEL_VERSION, CREATED_AT
    FROM FORECASTS
    WHERE SERIES_ID = '{series_id}' AND MODEL_VERSION = '{model_version}'
    ORDER BY DS
    """
    df = read_sql_df(sql)
    if df.empty:
        return df
    df["DS"] = pd.to_datetime(df["DS"])
    return df


@st.cache_data(ttl=60)
def load_metrics() -> pd.DataFrame:
    sql = """
    SELECT RUN_ID, SERIES_ID, WINDOW_START, WINDOW_END, MAE, MAPE, MODEL_VERSION, CREATED_AT
    FROM METRICS_MONITORING
    ORDER BY CREATED_AT DESC
    """
    df = read_sql_df(sql)
    if df.empty:
        return df
    df["WINDOW_START"] = pd.to_datetime(df["WINDOW_START"])
    df["WINDOW_END"] = pd.to_datetime(df["WINDOW_END"])
    df["CREATED_AT"] = pd.to_datetime(df["CREATED_AT"])
    return df


# ---------- Load data ----------
actuals = load_actuals(series_id)

if actuals.empty:
    st.error(f"No actuals found for SERIES_ID='{series_id}' in FEATURES.")
    st.stop()

last_actual_date = actuals["DS"].max().date()
start_hist = (last_actual_date - timedelta(days=days_history))

actuals_view = actuals[actuals["DS"].dt.date >= start_hist].copy()

# Forecast horizon dates we want to visualize
fcst_start = last_actual_date + timedelta(days=1)
fcst_end = last_actual_date + timedelta(days=future_days)

fcst_frames = []

if show_baseline:
    base = load_forecasts(series_id, BASELINE_VERSION)
    if not base.empty:
        base = base[(base["DS"].dt.date >= fcst_start) & (base["DS"].dt.date <= fcst_end)]
        base = base.rename(columns={"YHAT": f"YHAT_{BASELINE_VERSION}"})
        fcst_frames.append(base[["DS", f"YHAT_{BASELINE_VERSION}"]])
    else:
        st.sidebar.warning("No baseline forecasts found in FORECASTS table.")

if show_xgb:
    xgb = load_forecasts(series_id, XGB_VERSION)
    if not xgb.empty:
        xgb = xgb[(xgb["DS"].dt.date >= fcst_start) & (xgb["DS"].dt.date <= fcst_end)]
        xgb = xgb.rename(columns={"YHAT": f"YHAT_{XGB_VERSION}"})
        fcst_frames.append(xgb[["DS", f"YHAT_{XGB_VERSION}"]])
    else:
        st.sidebar.warning("No XGBoost forecasts found in FORECASTS table.")

# Combine forecast frames on DS
if fcst_frames:
    fcst_view = fcst_frames[0]
    for f in fcst_frames[1:]:
        fcst_view = pd.merge(fcst_view, f, on="DS", how="outer")
    fcst_view = fcst_view.sort_values("DS")
else:
    fcst_view = pd.DataFrame(columns=["DS"])

metrics = load_metrics()

# ---------- Layout ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Actuals + Forecasts (Next N Days)")
    # build chart dataframe
    chart_df = actuals_view.rename(columns={"Y": "ACTUAL"}).set_index("DS")[["ACTUAL"]]

    if not fcst_view.empty:
        fcst_idx = fcst_view.set_index("DS")
        chart_df = chart_df.join(fcst_idx, how="outer")

    st.line_chart(chart_df)

    st.caption(f"Actuals shown from {start_hist} to {last_actual_date}. Forecast horizon: {fcst_start} to {fcst_end}.")

with col2:
    st.subheader("Quick stats")
    st.metric("Last actual date", str(last_actual_date))
    st.metric("History days", str(days_history))
    st.metric("Forecast days", str(future_days))

    st.subheader("Latest forecast rows")
    if not fcst_view.empty:
        st.dataframe(fcst_view.tail(20), use_container_width=True)
    else:
        st.info("No forecast rows found for the selected horizon. Run `python src/forecast/01_batch_forecast.py` first.")

st.divider()

# ---------- Metrics section ----------
st.subheader("Backtest Metrics Summary (MAE / MAPE)")
if metrics.empty:
    st.info("No rows found in METRICS_MONITORING yet. Run backtests first.")
else:
    # Filter to this series if present
    m = metrics.copy()
    if "SERIES_ID" in m.columns:
        m = m[m["SERIES_ID"] == series_id]

    # Summary by model_version (excluding monitor rows)
    summary = (
        m[m["MODEL_VERSION"].isin([BASELINE_VERSION, XGB_VERSION])]
        .groupby("MODEL_VERSION")[["MAE", "MAPE"]]
        .mean()
        .reset_index()
        .sort_values("MODEL_VERSION")
    )

    st.dataframe(summary, use_container_width=True)

    st.subheader("Recent metric rows (latest 50)")
    st.dataframe(m.head(50), use_container_width=True)

st.divider()

st.subheader("Monitoring Rows (if available)")
if not metrics.empty:
    mon = metrics[(metrics["MODEL_VERSION"] == MONITOR_VERSION)]
    if mon.empty:
        st.info("No monitoring rows yet (normal if no overlap between forecasts and actuals).")
    else:
        st.dataframe(mon.head(50), use_container_width=True)

st.caption("Tip: Streamlit caches queries for 60s to reduce Snowflake calls.")
