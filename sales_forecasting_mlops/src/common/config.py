import os
from dotenv import load_dotenv

# Local dev: load .env if present
load_dotenv()

def _get_from_env(key: str):
    v = os.getenv(key)
    return v if v and v.strip() else None

def _get_from_streamlit_secrets(key: str):
    # Streamlit Cloud stores secrets in st.secrets
    try:
        import streamlit as st
        return st.secrets.get(key, None)
    except Exception:
        return None

def env(key: str) -> str:
    v = _get_from_env(key)
    if v is None:
        v = _get_from_streamlit_secrets(key)
    if v is None:
        raise ValueError(f"Missing secret/env var: {key}")
    return v

SNOWFLAKE_CFG = {
    "account": env("SNOWFLAKE_ACCOUNT"),
    "user": env("SNOWFLAKE_USER"),
    "password": env("SNOWFLAKE_PASSWORD"),
    "role": env("SNOWFLAKE_ROLE"),
    "warehouse": env("SNOWFLAKE_WAREHOUSE"),
    "database": env("SNOWFLAKE_DATABASE"),
    "schema": env("SNOWFLAKE_SCHEMA"),
}
