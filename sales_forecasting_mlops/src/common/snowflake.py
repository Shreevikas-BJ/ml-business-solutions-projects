import snowflake.connector
import pandas as pd
from src.common.config import SNOWFLAKE_CFG

def get_conn():
    kwargs = dict(
        account=SNOWFLAKE_CFG["account"],
        user=SNOWFLAKE_CFG["user"],
        password=SNOWFLAKE_CFG["password"],
        warehouse=SNOWFLAKE_CFG["warehouse"],
        database=SNOWFLAKE_CFG["database"],
        schema=SNOWFLAKE_CFG["schema"],
        role=SNOWFLAKE_CFG["role"],
    )

    # If provided, force the exact host (fixes 404 / wrong endpoint)
    if SNOWFLAKE_CFG.get("host"):
        kwargs["host"] = SNOWFLAKE_CFG["host"]

    return snowflake.connector.connect(**kwargs)

def read_sql_df(sql: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(sql, conn)
