# app/streamlit_app.py

import os
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Config ----------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DEFAULT_MASTER_PATH = os.path.join(PROCESSED_DIR, "master_users.parquet")

# Required base columns
REQUIRED_BASE_COLS = ["p_churn", "clv"]


# ---------- Helpers ----------

def classify_uplift_value(uplift: float) -> str:
    if uplift > 0.05:
        return "Persuadable"
    elif uplift > 0:
        return "Sure Thing"
    elif uplift < -0.05:
        return "Do-Not-Disturb"
    else:
        return "Lost Cause"


def normalize_master_users(df: pd.DataFrame):
    warnings = []

    missing_base = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")

    df = df.copy()

    if "user_id" not in df.columns:
        df["user_id"] = np.arange(1, len(df) + 1)
        warnings.append("user_id missing → synthetic IDs created.")

    if "uplift" not in df.columns:
        df["uplift"] = 1.0
        warnings.append("uplift missing → assumed uplift = 1.0.")

    if "uplift_positive" not in df.columns:
        df["uplift_positive"] = df["uplift"].clip(lower=0.0)
        warnings.append("uplift_positive computed.")

    if "uplift_segment" not in df.columns:
        df["uplift_segment"] = df["uplift"].apply(classify_uplift_value)
        warnings.append("uplift_segment classified automatically.")

    if "target_score" not in df.columns:
        df["target_score"] = df["p_churn"] * df["clv"] * df["uplift_positive"]
        warnings.append("target_score computed.")

    return df, warnings


@st.cache_data
def load_master_users(path: str):
    df = pd.read_parquet(path)
    df, _ = normalize_master_users(df)
    return df


def compute_campaign_summary(df, top_pct, segments):
    df_f = df[df["uplift_segment"].isin(segments)] if segments else df
    n = len(df_f)

    if n == 0:
        return df_f, {
            "n_total": 0,
            "n_targeted": 0,
            "expected_extra_revenue": 0.0,
            "avg_clv": 0.0,
            "avg_p_churn": 0.0,
            "targeted_pct": 0.0,
        }

    k = max(1, int(np.ceil(n * top_pct / 100)))
    df_t = df_f.sort_values("target_score", ascending=False).head(k)

    return df_t, {
        "n_total": n,
        "n_targeted": k,
        "expected_extra_revenue": (df_t["clv"] * df_t["uplift_positive"]).sum(),
        "avg_clv": df_t["clv"].mean(),
        "avg_p_churn": df_t["p_churn"].mean(),
        "targeted_pct": k * 100 / n,
    }


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="Subscription Value Brain", layout="wide")

    # ---------- Top bar ----------
    col_title, col_btn = st.columns([0.85, 0.15])

    with col_title:
        st.title("📊 Subscription Value Brain")
        st.caption(
            "Churn × CLV × Uplift → Decide who to target with discounts for maximum saved revenue."
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📘 Methodology"):
            st.switch_page("pages/1_Methodology.py")

    # ---------- Sidebar ----------
    st.sidebar.header("Targeting Strategy")

    top_pct = st.sidebar.slider(
        "Target top % of users by target_score",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
    )

    # ---------- Load data ----------
    if not os.path.exists(DEFAULT_MASTER_PATH):
        st.error(f"Default master_users file not found at: {DEFAULT_MASTER_PATH}")
        return

    df = load_master_users(DEFAULT_MASTER_PATH)

    segments = st.sidebar.multiselect(
        "Include uplift segments",
        sorted(df["uplift_segment"].unique()),
        default=list(df["uplift_segment"].unique()),
    )

    targeted, summary = compute_campaign_summary(df, top_pct, segments)

    # ---------- Tabs ----------
    tab_overview, tab_campaign, tab_targets = st.tabs(
        ["📌 Overview", "📈 Campaign Simulation", "🎯 Targeted Users"]
    )

    # ---------- Overview ----------
    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Total users", f"{len(df):,}")
        col2.metric("Avg CLV", f"${df['clv'].mean():,.2f}")
        col3.metric("Avg churn risk", f"{df['p_churn'].mean():.2f}")

        seg_counts = df["uplift_segment"].value_counts()
        col4.metric("Persuadables", int(seg_counts.get("Persuadable", 0)))
        col5.metric("Do-Not-Disturb", int(seg_counts.get("Do-Not-Disturb", 0)))
        col6.metric("Uplift segments", seg_counts.count())

    # ---------- Campaign ----------
    with tab_campaign:
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric("Users considered", summary["n_total"])
        c2.metric("Users targeted", summary["n_targeted"])
        c3.metric("Expected extra revenue", f"${summary['expected_extra_revenue']:,.2f}")

        c4.metric("Avg CLV (targeted)", f"${summary['avg_clv']:,.2f}")
        c5.metric("Avg churn risk (targeted)", f"{summary['avg_p_churn']:.2f}")
        c6.metric("Targeted %", f"{summary['targeted_pct']:.1f}%")

        st.markdown("### Uplift segments in targeted users")
        if summary["n_targeted"] > 0:
            seg_target = targeted["uplift_segment"].value_counts().reset_index()
            seg_target.columns = ["uplift_segment", "count"]
            st.bar_chart(seg_target, x="uplift_segment", y="count")

    # ---------- Targeted Users ----------
    with tab_targets:
        show_cols = ["user_id", "p_churn", "clv", "uplift", "uplift_segment", "target_score"]

        st.dataframe(
            targeted[show_cols].head(200),
            use_container_width=True,
        )

        csv = targeted[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download targeted users",
            csv,
            "targeted_users.csv",
            "text/csv",
        )


if __name__ == "__main__":
    main()