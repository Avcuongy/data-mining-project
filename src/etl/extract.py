import os
import sys
import datetime
import traceback

import pandas as pd
from sqlalchemy import create_engine
from utils.staging import staging
from utils.config_env import DATABASE_URL
import warnings

warnings.filterwarnings("ignore")

DATABASE_URL = DATABASE_URL


def _get_db_connection():
    """
    Create database connection.
    """
    connection_string = DATABASE_URL

    engine = create_engine(connection_string, echo=False)
    if engine:
        print("[Extract] Success to connect database.")
    else:
        print("[Extract] Failed to connect database.")
        sys.exit(1)

    return engine


def extract(tables: list = None, output_dir: str = "data/etl/staging") -> dict:
    """
    Extract data from source tables and export each table to a Parquet file.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("[Extract] Extracting...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if tables is None:
        tables = [
            "olist_order_customers",
            "olist_order_items",
            "olist_order_payments",
            "olist_orders",
            "olist_products",
            "olist_sellers",
        ]

    try:
        # Get database connection
        engine = _get_db_connection()

        exported_files = {}

        for table in tables:
            print("-" * 60)
            print(f"[Extract] Table: {table}")

            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, engine)

            if df.empty:
                print(
                    f"[Extract] Warning: Table '{table}' returned no data. Skipping Parquet file."
                )
                continue

            # Staging (per table)
            df = staging(df, table, engine)

            # Save as Parquet (per table)
            output_file = os.path.join(output_dir, f"{table}_{timestamp}.parquet")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_parquet(
                output_file,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            print(f"[Extract] Rows exported   : {len(df):,}")
            print(
                f"[Extract] File size (MB)  : {os.path.getsize(output_file) / (1024 * 1024):.2f}"
            )

            exported_files[table] = output_file

        print("-" * 60)
        print("[Extract] Completed")

        if not exported_files:
            print("[Extract] No tables were exported")

        return exported_files

    except Exception as e:
        print(f"[Extract] Error while exporting tables to Parquet: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    extract()
