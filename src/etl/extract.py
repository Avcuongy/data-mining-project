import os
import sys
import datetime
import traceback

import pandas as pd
from sqlalchemy import create_engine
from utils.staging import staging
from utils.config_env import DATABASE_URL

DATABASE_URL = DATABASE_URL


def get_db_connection():
    """
    Create database connection.
    """
    connection_string = DATABASE_URL

    engine = create_engine(connection_string, echo=False)
    if engine:
        print("Success to connect database.")
    else:
        print("Failed to connect database.")
        sys.exit(1)

    return engine


def to_parquet(df: pd.DataFrame, output_path: str):
    """
    Save one dataframe to Parquet format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )


def extract(tables: list = None, output_dir: str = "data/etl/staging") -> dict:
    """
    Extract data from source tables and export each table to a Parquet file.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Exporting data to Parquet format")
    print("Output directory:", output_dir)
    print("=" * 60)
    print()

    # Use current date for timestamping files
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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
        engine = get_db_connection()

        exported_files = {}

        for table in tables:
            print("\n" + "-" * 60)
            print(f"Table: {table}")

            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, engine)

            if df.empty:
                print(
                    f"Warning: Table '{table}' returned no data. Skipping Parquet file."
                )
                continue

            # Staging (per table)
            df = staging(df, table, engine)

            # Save as Parquet (per table)
            output_file = os.path.join(output_dir, f"{table}_{timestamp}.parquet")
            to_parquet(df, output_file)

            print(f"Rows exported   : {len(df):,}")
            print(
                f"File size (MB)  : {os.path.getsize(output_file) / (1024 * 1024):.2f}"
            )

            print("-" * 60)

            exported_files[table] = output_file

        print("\n" + "=" * 60)
        print("Extraction Completed")
        print("=" * 60)

        if not exported_files:
            print("No tables were exported")

        return exported_files

    except Exception as e:
        print(f"\nError while exporting tables to Parquet: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    extract()
