import streamlit as st

st.set_page_config(
    page_title="Methodology | Subscription Value Brain",
    layout="wide",
)

st.title("📘 System Methodology")
st.caption(
    "How churn prediction, CLV, and uplift modeling are combined into one targeting engine."
)

st.markdown("---")

# ---------------- Phase 1 ----------------
st.subheader("🔹 Phase 1: Customer Churn Prediction")
st.markdown("""
**Objective:** Identify customers likely to cancel their subscription.

**Models used:**
- Logistic Regression (baseline, interpretable)
- XGBoost Classifier (final model)

**Model selection:**
- Compared using ROC-AUC, Precision–Recall AUC, and Precision@K  
- Best precision–recall tradeoff selected

**Output:**  
`p_churn` → probability of churn (0–1)

**Why it matters:**  
Focus retention efforts only on customers truly at risk.
""")

# ---------------- Phase 2 ----------------
st.subheader("🔹 Phase 2: Customer Lifetime Value (CLV)")
st.markdown("""
**Objective:** Estimate future revenue per customer.

**Features:**  
RFM metrics — Recency, Frequency, Monetary

**Model used:**  
Random Forest Regressor

**Key transformation:**  
`log(1 + Monetary)` to reduce skew and stabilize learning

**Output:**  
`clv` → predicted future customer value

**Why it matters:**  
Losing high-value customers hurts revenue far more.
""")

# ---------------- Phase 3 ----------------
st.subheader("🔹 Phase 3: Uplift Modeling")
st.markdown("""
**Objective:** Measure the incremental effect of marketing.

**Data handling:**  
Large datasets processed in 200k-row chunks (up to ~8M rows).

**Modeling approach:**  
Two-model uplift strategy:
- Treatment model → `P1 = P(conversion | treatment)`
- Control model → `P0 = P(conversion | no treatment)`

**Uplift equation:**
Uplift = P1 − P0

python
Copy code

**Models used:**  
Random Forest Classifier (both models)

**Customer segments:**
- Persuadable
- Sure Thing
- Lost Cause
- Do-Not-Disturb
""")

# ---------------- Phase 4 ----------------
st.subheader("🔹 Phase 4: Scoring & Targeting Pipeline")
st.markdown("""
**Objective:** Combine all signals into one decision score.

**Final target score:**
target_score = p_churn × clv × uplift_positive

pgsql
Copy code

**Master table fields:**
- user_id
- p_churn
- clv
- uplift
- uplift_positive
- uplift_segment
- target_score

**Why it matters:**  
Prioritizes customers with the highest expected retention ROI.
""")

# ---------------- Phase 5 ----------------
st.subheader("🔹 Phase 5: Campaign Simulation")
st.markdown("""
**Objective:** Simulate real marketing constraints.

- Target only top X% of customers
- Filter by uplift segments
- Estimate expected retained revenue

This mirrors how real marketing teams operate under budget limits.
""")

st.markdown("---")

st.success(
    "This system reflects real-world data science pipelines used in subscription businesses "
    "to optimize retention and marketing spend."
)
