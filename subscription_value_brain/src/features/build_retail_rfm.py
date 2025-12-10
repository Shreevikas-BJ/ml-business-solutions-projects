# src/features/build_retail_rfm.py

import pandas as pd
from src.utils.io import load_online_retail, save_processed

def clean_retail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Remove canceled orders (InvoiceNo starting with 'C')
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # Keep only positive quantity
    df = df[df["Quantity"] > 0]

    # Calculate total price
    df["TotalPrice"] = df["Quantity"] * df["Price"]

    # Drop rows without customerID
    df = df.dropna(subset=["Customer ID"])

    # Rename for consistency
    df = df.rename(columns={"Customer ID": "customer_id"})

    return df


def main():
    df_raw = load_online_retail()
    df_clean = clean_retail(df_raw)

    rfm = build_rfm_features(df_clean)
    output = save_processed(rfm, "retail_rfm.parquet")

    print("Saved RFM features to:", output)


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Reference date = last date + 1 day
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm



if __name__ == "__main__":
    main()