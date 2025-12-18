import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from src.common.config import SNOWFLAKE_CFG
import pandas as pd

def get_conn(set_context: bool = True):
    conn = snowflake.connector.connect(
        account=SNOWFLAKE_CFG["account"],
        user=SNOWFLAKE_CFG["user"],
        password=SNOWFLAKE_CFG["password"],
        role=SNOWFLAKE_CFG["role"],
    )

    if set_context:
        with conn.cursor() as cur:
            # quote identifiers to be safe
            cur.execute(f'USE WAREHOUSE "{SNOWFLAKE_CFG["warehouse"]}"')
            cur.execute(f'USE DATABASE "{SNOWFLAKE_CFG["database"]}"')
            cur.execute(f'USE SCHEMA "{SNOWFLAKE_CFG["schema"]}"')

    return conn

def exec_sql(sql: str, set_context: bool = True):
    with get_conn(set_context=set_context) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

def write_df(df, table_name: str):
    with get_conn(set_context=True) as conn:
        return write_pandas(conn, df, table_name)

def read_sql_df(sql: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql(sql, conn)

