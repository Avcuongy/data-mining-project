import argparse
from typing import Optional

import duckdb
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def _get_duckdb_connection(
    db_path: str = "data_warehouse.duckdb",
) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=db_path, read_only=False)
    conn.execute("SET search_path = 'data_warehouse.DataWarehouse,main'")
    return conn


def _parse_optional_int(value: str | None) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"none", "null", ""}:
        return None
    return int(text)


def _validate_min_sup(min_sup: float) -> float:
    value = float(min_sup)
    if not 0.1 <= value <= 0.9:
        raise ValueError("min_sup must be a float percentage between 0.1 and 0.9")
    return value


def _apply_min_sup_percentage(
    df: pd.DataFrame, measure_column: str, min_sup: float
) -> pd.DataFrame:
    threshold = df[measure_column].max() * _validate_min_sup(min_sup)
    return df[df[measure_column] >= threshold]


def build_iceberg_sales_growth_cube(
    min_sup: float, k: Optional[int] = None, db_path: str = "data_warehouse.duckdb"
) -> pd.DataFrame:
    """Build `iceberg_sales_growth_cube`

    Dimensions: year, payment_type, customer_state, product_category_name
    Measures: SUM(total_item_value) as `total_revenue`, COUNT(order_id_item) as `qty_items`

    Args:
            min_sup: iceberg threshold as a float percentage in the range 0.1 to 0.9.
            k: optional top-k filter by average item value (if provided, select top-k by avg_item_value before applying min_sup).
            db_path: path to DuckDB file.

    Returns:
            DataFrame containing the resulting cube (and saved as table `iceberg_sales_growth_cube`).
    """
    conn = _get_duckdb_connection(db_path)

    sql = """
    SELECT
        d.year AS year,
        doi.payment_type AS payment_type,
        dc.customer_state AS customer_state,
        dp.product_category_name AS product_category_name,
        SUM(fs.total_item_value) AS total_revenue,
        COUNT(fs.order_id_item) AS qty_items,
        AVG(fs.total_item_value) AS avg_item_value
    FROM fact_sales AS fs
    LEFT JOIN dim_date AS d USING(date_key)
    LEFT JOIN dim_customer AS dc USING(customer_key)
    LEFT JOIN dim_product AS dp USING(product_key)
    LEFT JOIN dim_order_info AS doi USING(order_info_key)
    GROUP BY d.year, doi.payment_type, dc.customer_state, dp.product_category_name
    """

    df = conn.execute(sql).df()

    if k is not None:
        df = df.sort_values("avg_item_value", ascending=False).head(k)

    if min_sup is not None:
        df = _apply_min_sup_percentage(df, "total_revenue", min_sup)

    conn.register("tmp_sales_cube", df)
    conn.execute(
        "CREATE OR REPLACE TABLE iceberg_sales_growth_cube AS SELECT * FROM tmp_sales_cube"
    )
    conn.close()
    return df


def build_iceberg_logistics_risk_cube(
    min_sup: float, k: Optional[int] = None, db_path: str = "data_warehouse.duckdb"
) -> pd.DataFrame:
    """Build `iceberg_logistics_risk_cube`

    Dimensions: year, seller_state, customer_state, product_category_name
    Measures: COUNT(late_order) as `late_orders`, SUM(freight_value) as `total_freight`

    Args:
            min_sup: iceberg threshold as a float percentage in the range 0.1 to 0.9.
            k: optional top-k filter by average freight per order (if provided, select top-k by avg_freight before applying min_sup).
            db_path: path to DuckDB file.

    Returns:
            DataFrame containing the resulting cube (and saved as table `iceberg_logistics_risk_cube`).
    """
    conn = _get_duckdb_connection(db_path)

    sql = """
    WITH late_fact AS (
        SELECT fs.*
        FROM fact_sales AS fs
        INNER JOIN dim_order_info AS doi USING(order_info_key)
        WHERE doi.is_late_delivery = TRUE
    )
    SELECT
        d.year AS year,
        ds.seller_state AS seller_state,
        dc.customer_state AS customer_state,
        dp.product_category_name AS product_category_name,
        COUNT(DISTINCT lf.order_id) AS late_orders,
        SUM(lf.freight_value) AS total_freight,
        AVG(lf.freight_value) AS avg_freight
    FROM late_fact AS lf
    LEFT JOIN dim_date AS d USING(date_key)
    LEFT JOIN dim_customer AS dc USING(customer_key)
    LEFT JOIN dim_product AS dp USING(product_key)
    LEFT JOIN dim_seller AS ds USING(seller_key)
    GROUP BY d.year, ds.seller_state, dc.customer_state, dp.product_category_name
    """

    df = conn.execute(sql).df()

    if k is not None:
        df = df.sort_values("avg_freight", ascending=False).head(k)

    if min_sup is not None:
        df = _apply_min_sup_percentage(df, "late_orders", min_sup)

    conn.register("tmp_logistics_cube", df)
    conn.execute(
        "CREATE OR REPLACE TABLE iceberg_logistics_risk_cube AS SELECT * FROM tmp_logistics_cube"
    )
    conn.close()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build iceberg cubes")
    parser.add_argument("--db", default="data_warehouse.duckdb")
    parser.add_argument("--min_sup_sales", type=float, default=0.1)
    parser.add_argument("--k_sales", default=None)
    parser.add_argument("--min_sup_logistics", type=float, default=0.1)
    parser.add_argument("--k_logistics", default=None)
    args = parser.parse_args()

    print("[Iceberg cube] Building iceberg_sales_growth_cube")
    df_sales = build_iceberg_sales_growth_cube(
        args.min_sup_sales, _parse_optional_int(args.k_sales), db_path=args.db
    )
    print(f"               Rows in sales cube: {len(df_sales)}")

    print("[Iceberg cube] Building iceberg_logistics_risk_cube")
    df_log = build_iceberg_logistics_risk_cube(
        args.min_sup_logistics, _parse_optional_int(args.k_logistics), db_path=args.db
    )
    print(f"               Rows in logistics cube: {len(df_log)}")
