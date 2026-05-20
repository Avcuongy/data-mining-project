import duckdb
import pandas as pd
from pathlib import Path
import streamlit as st

DATAWAREHOUSE_PATH = Path("data_warehouse.duckdb")


@st.cache_resource
def get_conn(path: str | Path = DATAWAREHOUSE_PATH):
    conn = duckdb.connect(database=str(path), read_only=True)
    try:
        conn.execute("SET search_path = 'data_warehouse.DataWarehouse,main'")
    except Exception:
        pass
    return conn


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    conn = get_conn()
    if params:
        return conn.execute(sql, params).df()
    return conn.execute(sql).df()
