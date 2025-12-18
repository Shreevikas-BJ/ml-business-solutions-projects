"""
One-command runner for the Sales Forecasting MLOps Pipeline
Cross-platform (Windows / macOS / Linux)

Usage:
    python run_pipeline.py
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

STEPS = [
    ("ETL: Ingest & Clean", ["python", "src/etl/01_ingest_clean.py"]),
    ("ETL: Feature Engineering", ["python", "src/etl/02_feature_engineering.py"]),
    ("Baseline Model Backtest", ["python", "src/train/01_train_baseline.py"]),
    ("Train XGBoost Model", ["python", "src/train/02_train_xgboost.py"]),
    ("XGBoost Backtest", ["python", "src/train/03_backtest.py"]),
    ("Batch Forecast (Next N Days)", ["python", "src/forecast/01_batch_forecast.py"]),
    ("Monitoring Snapshot", ["python", "src/monitoring/01_monitoring_metrics.py"]),
]

def run_step(name, cmd):
    print("\n" + "=" * 70)
    print(f"▶ {name}")
    print("=" * 70)

    # Ensure child Python processes can import `src`
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print(f"\n❌ Pipeline failed at step: {name}")
        sys.exit(1)

def main():
    print("\n🚀 Starting Sales Forecasting MLOps Pipeline\n")
    for name, cmd in STEPS:
        run_step(name, cmd)
    print("\n✅ Pipeline completed successfully\n")

if __name__ == "__main__":
    main()
