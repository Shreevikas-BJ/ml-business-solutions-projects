***📈 Sales Forecasting MLOps Pipeline***

**PySpark · Snowflake · XGBoost · Streamlit**

Production-style end-to-end sales forecasting pipeline that ingests raw data, performs scalable feature engineering with PySpark, trains baseline and ML models, and serves forecasts via a Snowflake-backed Streamlit dashboard.

*Problem Statement*

Accurate sales forecasting is critical for inventory planning, revenue forecasting, and operational decision-making.
This project demonstrates how to build a production-ready forecasting pipeline using modern data engineering and MLOps practices.

**Solution Overview**

End-to-end pipeline covering data ingestion → feature engineering → modeling → evaluation → forecasting → monitoring → visualization.

Key ideas:

      Use PySpark for scalable ETL and time-series feature engineering

      Persist features, forecasts, and metrics in Snowflake

      Compare baseline heuristics vs ML models (XGBoost)

      Serve results via an interactive Streamlit dashboard

**Architecture**
Raw CSV Data
   ↓
PySpark ETL
   ↓
Feature Engineering (lags, rolling stats)
   ↓
Snowflake Feature Store (FEATURES)
   ↓
Baseline Model + XGBoost
   ↓
Backtesting (MAE / MAPE)
   ↓
Batch Forecasts (FORECASTS)
   ↓
Monitoring Metrics (METRICS_MONITORING)
   ↓
Streamlit Dashboard

**Models**

      Baseline: Last-week average (strong heuristic benchmark)

      ML Model: XGBoost (industry-standard for tabular forecasting)

      Models are versioned and stored as artifacts (models/xgb_v1.json, etc.).

**Evaluation**

      Rolling backtests

Metrics:

      MAE (Mean Absolute Error)

      MAPE (Mean Absolute Percentage Error)

Metrics written back to Snowflake for monitoring and comparison

**Dashboard**

      Interactive Streamlit app reading directly from Snowflake:

      Actual vs Forecast (Baseline vs XGBoost)

      Future forecast horizon (N days)

      Backtest metric summaries

      Monitoring / drift metrics

-> Live Demo: (https://sales-forecasting-mlops.streamlit.app/)

**Tech Stack**

      Data Engineering: PySpark

      Data Warehouse: Snowflake (free trial)

      Modeling: XGBoost, Pandas

      MLOps: Versioned artifacts, batch pipelines, monitoring

      Visualization: Streamlit

      Language: Python


**Run the Pipeline (Local)**
python run_pipeline.py


This will:

      Ingest and clean raw sales data

      Generate time-series features using PySpark

      Write features to Snowflake

      Train baseline and XGBoost models

      Run rolling backtests

      Generate batch forecasts

      Log monitoring metrics

**Configuration**

Local development: .env file (not committed)

Cloud deployment: Streamlit Community Cloud Secrets

Secrets include:

      SNOWFLAKE_ACCOUNT
      SNOWFLAKE_USER
      SNOWFLAKE_PASSWORD
      SNOWFLAKE_ROLE
      SNOWFLAKE_WAREHOUSE
      SNOWFLAKE_DATABASE
      SNOWFLAKE_SCHEMA

**Why This Project Matters**

Demonstrates real-world MLOps thinking

Bridges data engineering + ML

Uses industry-relevant tools (PySpark, Snowflake, XGBoost)

Shows ability to build end-to-end, deployable systems

**Future Enhancements**

Multi-series forecasting (region / category level)

Automated retraining schedules

Advanced drift detection

CI/CD integration

👤 Author

Shreevikas Bangalore Jagadish
Master’s in Information Technology & Management (Data & Analytics)
Illinois Institute of Technology
GitHub: https://github.com/Shreevikas-BJ



