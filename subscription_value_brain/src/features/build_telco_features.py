# src/features/build_telco_features.py
import pandas as pd

from src.utils.io import load_telco_churn, save_processed


def clean_telco_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for Telco dataset.
    - Replace ' ' in TotalCharges with NaN and convert to float
    - Drop customerID if present (just an identifier)
    """
    df = df.copy()

    # Some versions of the dataset have TotalCharges as string with spaces
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            pd.to_numeric(df["TotalCharges"], errors="coerce")
        )

    # Drop obviously useless pure IDs
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Drop rows where TotalCharges became NaN (their tenure is often 0)
    if "TotalCharges" in df.columns:
        df = df.dropna(subset=["TotalCharges"])

    return df


def main():
    df_raw = load_telco_churn()
    df_clean = clean_telco_raw(df_raw)

    output_path = save_processed(df_clean, "telco_clean.parquet")
    print(f"Saved cleaned Telco data to: {output_path}")


if __name__ == "__main__":
    main()
