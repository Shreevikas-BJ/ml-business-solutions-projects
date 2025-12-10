# src/features/build_criteo_features.py

import pandas as pd
import os

from src.utils.io import RAW_DATA_DIR, PROCESSED_DATA_DIR


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a chunk of the Criteo uplift dataset.
    """
    df = chunk.copy()

    # Convert to numeric (Criteo often stores everything as strings)
    for col in df.columns:
        if col.startswith("f"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Only keep relevant columns (optional, keeps RAM low)
    keep_cols = [c for c in df.columns if c.startswith("f")] + [
        "treatment",
        "conversion",
    ]

    df = df[keep_cols]

    return df


def main():
    input_file = os.path.join(RAW_DATA_DIR, "criteo_uplift.csv")  
    output_file = os.path.join(PROCESSED_DATA_DIR, "criteo_small.parquet")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Use chunking for large file
    chunksize = 200_000
    total = 0
    processed_chunks = []

    print("Processing large Criteo file in chunks...")

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        df_clean = preprocess_chunk(chunk)

        processed_chunks.append(df_clean)
        total += len(df_clean)

        print(f"Processed rows: {total:,}")

        # To avoid huge RAM usage, concatenate & write every ~8 million rows
        if total >= 8_000_000:
            big_df = pd.concat(processed_chunks, ignore_index=True)
            big_df.to_parquet(output_file, index=False)
            print(f"Saved ~8M rows to: {output_file}")
            return  # early exit for this example

    print("Finished processing all chunks.")


if __name__ == "__main__":
    main()
