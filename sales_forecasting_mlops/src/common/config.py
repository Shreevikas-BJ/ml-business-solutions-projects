import os
from dotenv import load_dotenv

load_dotenv()

def _get_env(key: str):
    v = os.getenv(key)
    return v if v and v.strip() else None

def _get_secrets(key: str):
    try:
        import streamlit as st
        return st.secrets.get(key, None)
    except Exception:
        return None

def get_cfg(key: str, required: bool = True):
    v = _get_env(key) or _get_secrets(key)
    if required and (v is None or str(v).strip() == ""):
        raise ValueError(f"Missing config: {key}")
    return v

SNOWFLAKE_CFG = {
    "account": get_cfg("SNOWFLAKE_ACCOUNT"),
    "user": get_cfg("SNOWFLAKE_USER"),
    "password": get_cfg("SNOWFLAKE_PASSWORD"),
    "role": get_cfg("SNOWFLAKE_ROLE"),
    "warehouse": get_cfg("SNOWFLAKE_WAREHOUSE"),
    "database": get_cfg("SNOWFLAKE_DATABASE"),
    "schema": get_cfg("SNOWFLAKE_SCHEMA"),
    # Optional (recommended for Streamlit Cloud)
    "host": get_cfg("SNOWFLAKE_HOST", required=False),
}
