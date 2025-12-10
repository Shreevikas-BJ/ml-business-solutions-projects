# src/utils/preprocessing.py
from typing import Tuple, List
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def build_telco_preprocessor(
    df: pd.DataFrame,
    target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, List[str]]:
    """
    Prepare Telco dataset for modeling:
    - Drop rows with missing target
    - Split into X (features) and y (label)
    - Build a ColumnTransformer for numeric + categorical features

    Returns:
        X (raw, before transform)
        y
        preprocessor (ColumnTransformer)
        feature_names (list of original feature columns)
    """
    df = df.copy()

    # Convert target to binary 0/1
    df[target_col] = df[target_col].map({"No": 0, "Yes": 1})

    # Drop rows without target just in case
    df = df.dropna(subset=[target_col])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Identify numeric vs categorical columns
    numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_feats = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Numeric: scale
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    # Categorical: one-hot encode, ignore unseen categories at inference
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
            )
        ]
    )

    # Combine both into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_feats),
            ("cat", categorical_transformer, categorical_feats),
        ]
    )

    return X, y, preprocessor, X.columns.tolist()
