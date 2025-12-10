# app/streamlit_app.py

import os
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config ----------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_MASTER_PATH = os.path.join(PROCESSED_DIR, "master_users.parquet")

print("DEBUG - Loading from:", DEFAULT_MASTER_PATH)

# "Core" columns we really need
REQUIRED_BASE_COLS = [
    "p_churn",  # must have (for churn-based targeting)
    "clv",      # must have (for value-based targeting)
]

# Recommended / derived columns that we can auto-compute if missing
RECOMMENDED_COLS = [
    "user_id",
    "uplift",
    "uplift_segment",
    "uplift_positive",
    "target_score",
]

ALL_DOC_COLS = REQUIRED_BASE_COLS + RECOMMENDED_COLS


# ---------- Helpers for derived columns ----------

def classify_uplift_value(uplift: float) -> str:
    """
    Classify uplift into marketing-friendly segments.
    """
    if uplift > 0.05:
        return "Persuadable"          # good target
    elif uplift > 0:
        return "Sure Thing"
    elif uplift < -0.05:
        return "Do-Not-Disturb"       # negatively affected
    else:
        return "Lost Cause"


def normalize_master_users(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Ensure that the master_users DataFrame has the columns needed by the app.
    - Enforces required base columns: p_churn, clv
    - Auto-creates: user_id, uplift, uplift_positive, uplift_segment, target_score if missing.
    Returns:
        df_norm, warnings
    """
    warnings = []

    # Check base required columns
    missing_base = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"Uploaded or loaded file is missing required base columns: {missing_base}. "
            f"At minimum, you must provide: {REQUIRED_BASE_COLS}"
        )

    df = df.copy()

    # user_id: if missing, create a simple 1..N synthetic ID
    if "user_id" not in df.columns:
        df["user_id"] = np.arange(1, len(df) + 1)
        warnings.append("user_id was missing → created synthetic IDs 1..N.")

    # uplift: if missing, assume pure churn × CLV targeting (uplift = 1.0)
    if "uplift" not in df.columns:
        df["uplift"] = 1.0
        warnings.append(
            "uplift was missing → set uplift = 1.0 for all users "
            "(behaves like pure churn × CLV targeting)."
        )

    # uplift_positive: max(uplift, 0)
    if "uplift_positive" not in df.columns:
        df["uplift_positive"] = df["uplift"].clip(lower=0.0)
        warnings.append(
            "uplift_positive was missing → computed uplift_positive = max(uplift, 0)."
        )

    # uplift_segment: classify using uplift thresholds
    if "uplift_segment" not in df.columns:
        df["uplift_segment"] = df["uplift"].apply(classify_uplift_value)
        warnings.append(
            "uplift_segment was missing → classified users based on uplift "
            "(Persuadable / Sure Thing / Lost Cause / Do-Not-Disturb)."
        )

    # target_score: p_churn × clv × uplift_positive
    if "target_score" not in df.columns:
        df["target_score"] = df["p_churn"] * df["clv"] * df["uplift_positive"]
        warnings.append(
            "target_score was missing → computed target_score = p_churn × clv × uplift_positive."
        )

    return df, warnings


# ---------- Data loading ----------

@st.cache_data
def load_master_users(path: str = DEFAULT_MASTER_PATH) -> pd.DataFrame:
    """
    Load and normalize the master users table from default path.
    """
    df = pd.read_parquet(path)
    df_norm, _ = normalize_master_users(df)
    return df_norm


def compute_campaign_summary(df: pd.DataFrame, top_pct: float, selected_segments: list):
    """
    Filter by uplift segment, take top pct by target_score, and compute summary stats.
    """
    df_filtered = df.copy()

    if selected_segments:
        df_filtered = df_filtered[df_filtered["uplift_segment"].isin(selected_segments)].copy()

    n_total = len(df_filtered)
    if n_total == 0:
        return df_filtered, {
            "n_total": 0,
            "n_targeted": 0,
            "expected_extra_revenue": 0.0,
            "avg_clv": 0.0,
            "avg_p_churn": 0.0,
            "persuadables": 0,
            "dont_disturb": 0,
            "targeted_pct": 0.0,
        }

    # top_pct is in %, e.g. 20 → top 20%
    k = int(np.ceil(n_total * top_pct / 100.0))
    k = max(1, k)  # ensure at least 1

    df_sorted = df_filtered.sort_values("target_score", ascending=False).reset_index(drop=True)
    targeted = df_sorted.head(k)

    expected_extra_revenue = (targeted["clv"] * targeted["uplift_positive"]).sum()
    avg_clv = targeted["clv"].mean()
    avg_p_churn = targeted["p_churn"].mean()
    persuadables = (targeted["uplift_segment"] == "Persuadable").sum()
    dont_disturb = (targeted["uplift_segment"] == "Do-Not-Disturb").sum()
    targeted_pct = k * 100.0 / n_total

    summary = {
        "n_total": n_total,
        "n_targeted": k,
        "expected_extra_revenue": expected_extra_revenue,
        "avg_clv": avg_clv,
        "avg_p_churn": avg_p_churn,
        "persuadables": persuadables,
        "dont_disturb": dont_disturb,
        "targeted_pct": targeted_pct,
    }

    return targeted, summary


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(
        page_title="Subscription Value Brain",
        layout="wide",
    )

    st.title("📊 Subscription Value Brain")
    st.caption(
        "Churn × CLV × Uplift → Decide who to target with discounts for maximum saved revenue."
    )

    # ----- Sidebar: data + strategy controls -----
    st.sidebar.header("Data")

    use_custom = st.sidebar.checkbox("Upload custom master_users file", value=False)

    uploaded = None
    df = None
    normalization_warnings: list[str] = []

    if use_custom:
        uploaded = st.sidebar.file_uploader(
            "Upload master_users file (.parquet or .csv)",
            type=["parquet", "csv"],
        )

    # Sidebar: Targeting controls (always visible)
    st.sidebar.header("Targeting Strategy")

    top_pct = st.sidebar.slider(
        "Target top % of users by target_score",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.write("**Segments legend:**")
    st.sidebar.write("- 🟢 Persuadable → good marketing targets")
    st.sidebar.write("- 🔵 Sure Thing → will convert anyway")
    st.sidebar.write("- 🟡 Lost Cause → unlikely to convert")
    st.sidebar.write("- 🔴 Do-Not-Disturb → may react negatively")

    # ---------- Data selection + normalization ----------

    if use_custom:
        st.subheader("📥 Custom master_users file requirements")

        st.markdown(
            """
You can upload a **partial or full** master_users file.

At minimum, you must include:

- `p_churn` → predicted churn probability between 0 and 1  
- `clv` → numeric customer lifetime value (e.g. in USD – this app will display `$`)

The app can **auto-compute** the following if missing:

- `user_id` → synthetic IDs 1..N  
- `uplift` → if missing, assumes `uplift = 1.0` (pure churn × CLV targeting)  
- `uplift_positive` → computed as `max(uplift, 0)`  
- `uplift_segment` → classified using uplift into Persuadable / Sure Thing / Lost Cause / Do-Not-Disturb  
- `target_score` → computed as `p_churn × clv × uplift_positive`
"""
        )

        st.markdown("**Recommended columns (if you have them):**")
        st.code("\n".join(ALL_DOC_COLS), language="text")

        st.markdown("---")

        if uploaded is None:
            st.info(
                "⬆️ Upload a custom master_users file in the sidebar to continue, "
                "or uncheck the box to use the default demo data."
            )
            return
        else:
            if uploaded.name.endswith(".parquet"):
                df_raw = pd.read_parquet(uploaded)
            else:
                df_raw = pd.read_csv(uploaded)

            try:
                df, normalization_warnings = normalize_master_users(df_raw)
            except ValueError as e:
                st.error(str(e))
                return
    else:
        if not os.path.exists(DEFAULT_MASTER_PATH):
            st.error(f"Default master_users file not found at: {DEFAULT_MASTER_PATH}")
            return
        df = load_master_users(DEFAULT_MASTER_PATH)
        normalization_warnings = []

    # Show any normalization warnings (for uploaded data only)
    if normalization_warnings:
        with st.expander("⚠️ Data adjustments applied automatically", expanded=True):
            for w in normalization_warnings:
                st.write("- " + w)

    # ---------- Now df is ready ----------

    all_segments = sorted(df["uplift_segment"].dropna().unique())
    selected_segments = st.sidebar.multiselect(
        "Include uplift segments",
        options=all_segments,
        default=all_segments,
    )

    # Recompute campaign summary based on current slider + segment filters
    targeted, summary = compute_campaign_summary(df, top_pct, selected_segments)

    # ---------- Tabs for readability ----------

    tab_overview, tab_campaign, tab_targets = st.tabs(
        ["📌 Overview", "📈 Campaign Simulation", "🎯 Targeted Users"]
    )

    # -------- Overview tab --------
    with tab_overview:
        st.subheader("Universe Overview")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        total_users = len(df)
        avg_clv_all = df["clv"].mean()
        avg_p_churn_all = df["p_churn"].mean()

        col1.metric("Total users in universe", f"{total_users:,}")
        col2.metric("Avg CLV (all users)", f"${avg_clv_all:,.2f}")
        col3.metric("Avg churn risk (all users)", f"{avg_p_churn_all:.2f}")

        seg_counts = df["uplift_segment"].value_counts()
        persuadables_all = int(seg_counts.get("Persuadable", 0))
        dont_disturb_all = int(seg_counts.get("Do-Not-Disturb", 0))

        col4.metric("Persuadables (all)", f"{persuadables_all:,}")
        col5.metric("Do-Not-Disturb (all)", f"{dont_disturb_all:,}")
        col6.metric("Unique uplift segments", len(seg_counts))

        st.caption("These metrics are calculated on the full user universe before any targeting filter.")

    # -------- Campaign Simulation tab --------
    with tab_campaign:
        st.subheader("Campaign Simulation Results")

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Users considered (after segment filter)", f"{summary['n_total']:,}")
        c2.metric("Users targeted", f"{summary['n_targeted']:,}")
        c3.metric("Expected extra revenue", f"${summary['expected_extra_revenue']:,.2f}")

        c4.metric("Avg CLV (targeted)", f"${summary['avg_clv']:,.2f}")
        c5.metric("Avg churn risk (targeted)", f"{summary['avg_p_churn']:.2f}")
        c6.metric("Targeted % of considered users", f"{summary['targeted_pct']:.1f}%")

        st.markdown("---")

        st.subheader("Uplift segments in targeted users")
        if summary["n_targeted"] == 0:
            st.info("No users selected with current filters. Try increasing top % or changing segments.")
        else:
            seg_target_counts = targeted["uplift_segment"].value_counts().reset_index()
            seg_target_counts.columns = ["uplift_segment", "count"]
            st.bar_chart(
                data=seg_target_counts,
                x="uplift_segment",
                y="count",
            )

    # -------- Targeted Users tab --------
    with tab_targets:
        st.subheader("Top targeted users")

        if summary["n_targeted"] == 0:
            st.info("No users selected. Adjust the sliders/segments in the sidebar to see targets.")
        else:
            show_cols = ["user_id", "p_churn", "clv", "uplift", "uplift_segment", "target_score"]

            st.dataframe(
                targeted[show_cols].head(200),
                use_container_width=True,
            )

            csv_data = targeted[show_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download targeted users as CSV",
                data=csv_data,
                file_name="targeted_users.csv",
                mime="text/csv",
            )

        st.caption(
            "This is a demo tool: IDs and scores are from public datasets combined into a synthetic user universe "
            "or from your uploaded master_users file."
        )


if __name__ == "__main__":
    main()
