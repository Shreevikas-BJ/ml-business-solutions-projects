# src/pipeline/scoring_pipeline.py

import os
import numpy as np
import pandas as pd

from src.utils.io import PROCESSED_DATA_DIR


def load_churn_scores() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_DIR, "telco_churn_scores.parquet")
    df = pd.read_parquet(path)
    return df[["p_churn"]]


def load_clv_scores() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_DIR, "retail_clv.parquet")
    df = pd.read_parquet(path)
    # Ensure the column is named CLV
    if "CLV" not in df.columns:
        raise ValueError("retail_clv.parquet must contain a 'CLV' column.")
    return df[["CLV"]]


def load_uplift_scores() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_DIR, "criteo_uplift_scores.parquet")
    df = pd.read_parquet(path)
    return df[["uplift", "uplift_segment"]]


def build_master_users() -> pd.DataFrame:
    churn_df = load_churn_scores()
    clv_df = load_clv_scores()
    uplift_df = load_uplift_scores()

    # Determine how many users we can safely align
    n = min(len(churn_df), len(clv_df), len(uplift_df))
    print(f"Building master table with {n} synthetic users.")

    # Randomly sample n rows from each source to avoid any accidental ordering bias
    rng = 42
    churn_sub = churn_df.sample(n, random_state=rng).reset_index(drop=True)
    clv_sub = clv_df.sample(n, random_state=rng + 1).reset_index(drop=True)
    uplift_sub = uplift_df.sample(n, random_state=rng + 2).reset_index(drop=True)

    # Create synthetic user_id
    user_ids = np.arange(1, n + 1)

    master = pd.DataFrame({
        "user_id": user_ids,
        "p_churn": churn_sub["p_churn"].values,
        "clv": clv_sub["CLV"].values,
        "uplift": uplift_sub["uplift"].values,
        "uplift_segment": uplift_sub["uplift_segment"].values,
    })

    # Clip uplift at 0 for scoring (we don't reward negative uplift)
    master["uplift_positive"] = master["uplift"].clip(lower=0.0)

    # Targeting score: high churn risk, high value, and positive uplift
    master["target_score"] = (
        master["p_churn"] * master["clv"] * master["uplift_positive"]
    )

    # Rank customers by target_score (1 = best target)
    master = master.sort_values("target_score", ascending=False).reset_index(drop=True)
    master["target_rank"] = np.arange(1, len(master) + 1)

    return master


def simulate_campaign(master: pd.DataFrame, top_fraction: float = 0.2) -> None:
    """
    Very simple campaign simulation:
    - Target top X% of users by target_score.
    - Approximate expected 'extra revenue' as clv * uplift_positive.
    (This is toy math, but great for interview storytelling.)
    """
    n = len(master)
    k = int(np.ceil(n * top_fraction))

    selected = master.head(k)

    # Toy approximation of "extra revenue"
    expected_extra_revenue = (selected["clv"] * selected["uplift_positive"]).sum()

    persuadables = (selected["uplift_segment"] == "Persuadable").sum()
    dont_disturb = (selected["uplift_segment"] == "Do-Not-Disturb").sum()

    print(f"\n--- Campaign Simulation (top {top_fraction*100:.0f}% users) ---")
    print(f"Total users: {n}")
    print(f"Targeted users: {k}")
    print(f"Expected extra revenue (approx): {expected_extra_revenue:,.2f}")
    print(f"Persuadables in target group: {persuadables}")
    print(f"Do-Not-Disturb in target group: {dont_disturb}")


def main():
    master = build_master_users()

    # Save master table
    output_path = os.path.join(PROCESSED_DATA_DIR, "master_users.parquet")
    master.to_parquet(output_path, index=False)
    print(f"Saved master users table to: {output_path}")

    # Run a simple campaign simulation on top 20%
    simulate_campaign(master, top_fraction=0.2)


if __name__ == "__main__":
    main()
