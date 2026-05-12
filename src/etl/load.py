from pathlib import Path

import pandas as pd
import duckdb
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLETED_DIR = PROJECT_ROOT / "data" / "etl" / "completed"
DUCK_DB_PATH = PROJECT_ROOT / "data_warehouse.duckdb"
# Load dimension tables first, then fact table (to respect foreign key constraints)
SOURCE = [
    "dim_date",
    "dim_customer",
    "dim_seller",
    "dim_product",
    "dim_order_info",
    "fact_sales",
]
TABLE_COLUMNS = {
    "dim_customer": [
        "customer_key",
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
        "is_current",
    ],
    "dim_seller": [
        "seller_key",
        "seller_id",
        "seller_city",
        "seller_state",
        "is_current",
    ],
    "dim_product": [
        "product_key",
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_volume_cm3",
        "product_photos_qty",
    ],
    "dim_order_info": [
        "order_info_key",
        "order_status",
        "payment_type",
        "max_payment_installments",
        "is_late_delivery",
    ],
    "dim_date": [
        "date_key",
        "full_date",
        "day",
        "month",
        "year",
        "quarter",
        "day_of_week",
        "is_weekend",
        "is_holiday",
    ],
    "fact_sales": [
        "order_id",
        "order_id_item",
        "date_key",
        "customer_key",
        "product_key",
        "seller_key",
        "order_info_key",
        "price",
        "freight_value",
        "total_item_value",
    ],
}


def _get_latest_file_in_directory(directory: Path) -> dict[str, Path]:
    latest_files: dict[str, tuple[float, Path]] = {}

    if not directory.exists():
        return {}

    for path in directory.glob("*.parquet"):
        table_name = next(
            (name for name in SOURCE if path.name.startswith(f"{name}_")),
            None,
        )
        if table_name is None:
            continue

        modified_time = path.stat().st_mtime

        current = latest_files.get(table_name)
        if current is None or modified_time > current[0]:
            latest_files[table_name] = (modified_time, path)

    return {table_name: path for table_name, (_, path) in latest_files.items()}


def _insert_data_to_table(
    conn: duckdb.DuckDBPyConnection, table_name: str, file_path: Path
) -> None:
    try:
        df = pd.read_parquet(file_path)
        columns = TABLE_COLUMNS[table_name]

        insert_sql = (
            f"INSERT INTO data_warehouse.{table_name} ({', '.join(columns)}) "
            f"SELECT {', '.join(columns)} FROM df"
        )

        conn.execute(insert_sql)

        row_count = len(df)
        print(f"[Load] Successfully inserted {row_count} rows into {table_name}")

    except Exception as e:
        print(f"[Load] Warning: Error loading {table_name}: {str(e)}")
        raise


def load() -> None:
    print("[Load] Loading...")
    db_path = DUCK_DB_PATH
    conn = duckdb.connect(str(db_path))
    if conn is None:
        print("[Load] Error: Failed to connect to DuckDB")
        return

    try:
        conn.execute("SET search_path = 'data_warehouse.DataWarehouse,main'")

        latest_files = _get_latest_file_in_directory(COMPLETED_DIR)

        if not latest_files:
            print("[Load] Warning: No parquet files found in directory")
            return
        with conn:
            for table_name in SOURCE:
                if table_name in latest_files:
                    file_path = latest_files[table_name]
                    _insert_data_to_table(conn, table_name, file_path)
                else:
                    print(f"[Load] Warning: No file found for {table_name}")

        print("[Load] Completed")

    except Exception as e:
        print(f"[Load] Error during data loading: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    load()
