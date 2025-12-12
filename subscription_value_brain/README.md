📊 Subscription Value Brain

End-to-End Churn, CLV & Uplift Modeling System

A production-style data science project that shows how subscription businesses (Netflix/Spotify/SaaS-like) can reduce churn and optimize marketing spend by targeting the right customers with the right offers.

🚀 What This Project Does

  The system answers four key business questions:

  Who is likely to churn? (Churn prediction)

  How valuable is each customer? (CLV)

  Will marketing actually help? (Uplift modeling)

  Who should be targeted first? (Unified scoring)

  The final output is an interactive Streamlit dashboard that simulates real marketing decisions under budget constraints.

🧠 Modeling Approach (High Level)
  1️⃣ Churn Prediction

  Models: Logistic Regression, XGBoost

  Metrics: ROC-AUC, Precision-Recall, Precision@K

  Output: p_churn (probability of churn)

  2️⃣ Customer Lifetime Value (CLV)

  Features: RFM (Recency, Frequency, Monetary)

  Model: Random Forest Regressor

  Log-transformed monetary values to handle skew

  Output: clv (expected future value)

  3️⃣ Uplift Modeling

  Two-model approach:

  Treatment model → P1

  Control model → P0

  Uplift = P1 − P0

  Model: Random Forest Classifier

  Customers segmented into:

  Persuadable, Sure Thing, Lost Cause, Do-Not-Disturb

  4️⃣ Targeting Score
    target_score = p_churn × clv × uplift_positive

  Ranks customers by expected retained revenue.

📊 Streamlit Dashboard

  The app provides:

  Target top X% of customers

  Campaign simulation metrics

  Uplift segment breakdown

  Downloadable target list

  Separate Methodology page explaining model choices

🛠️ Tech Stack

    Python, Pandas, NumPy

    Scikit-learn, XGBoost

    Streamlit

    Joblib

    Streamlit (deployment)

🎯 Why This Project Matters

  This project demonstrates:

  End-to-end data science thinking

  Business-driven modeling decisions

  Scalable data processing

  Clean pipelines & deployment-ready code

  Clear communication of impact

👤 Author

  Shreevikas Jagadish
