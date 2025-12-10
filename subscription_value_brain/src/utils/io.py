# src/utils/io.py
import os
import pandas as pd

# Resolve project root: src/.. = project folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_telco_churn(filename: str = "telco_churn.csv") -> pd.DataFrame:
    """
    Load the Telco churn dataset from data/raw.
    """
    path = os.path.join(RAW_DATA_DIR, filename)
    return pd.read_csv(path)

def load_online_retail(filename: str = "online_retail_ii.csv") -> pd.DataFrame:
    """
    Load the Online Retail II dataset from data/raw.
    """
    path = os.path.join(RAW_DATA_DIR, filename)
    return pd.read_csv(path, encoding="latin1")


def save_processed(df: pd.DataFrame, filename: str) -> str:
    """
    Save a processed DataFrame to data/processed as parquet.
    Returns the full path.
    """
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
