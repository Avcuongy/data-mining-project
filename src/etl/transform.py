import datetime
import os
import re
from pathlib import Path

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = PROJECT_ROOT / "data" / "etl" / "staging"
COMPLETED_DIR = PROJECT_ROOT / "data" / "etl" / "completed"
SOURCE = {
    "olist_order_customers",
    "olist_order_items",
    "olist_order_payments",
    "olist_orders",
    "olist_products",
    "olist_sellers",
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


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _assign_surrogate_key(frame: pd.DataFrame, key_column: str) -> pd.DataFrame:
    frame = frame.drop_duplicates().reset_index(drop=True).copy()
    frame.insert(0, key_column, range(1, len(frame) + 1))
    return frame


def _build_dim_customer(customers: pd.DataFrame) -> pd.DataFrame:
    frame = customers[
        [
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
        ]
    ].copy()
    frame = frame.drop_duplicates().sort_values(
        ["customer_id", "customer_unique_id"], kind="stable"
    )
    frame = _assign_surrogate_key(frame, "customer_key")
    frame["is_current"] = True
    return frame[
        [
            "customer_key",
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state",
            "is_current",
        ]
    ]


def _build_dim_seller(sellers: pd.DataFrame) -> pd.DataFrame:
    frame = sellers[
        [
            "seller_id",
            "seller_city",
            "seller_state",
        ]
    ].copy()
    frame = frame.drop_duplicates().sort_values(["seller_id"], kind="stable")
    frame = _assign_surrogate_key(frame, "seller_key")
    frame["is_current"] = True
    return frame[
        ["seller_key", "seller_id", "seller_city", "seller_state", "is_current"]
    ]


def _build_dim_product(products: pd.DataFrame) -> pd.DataFrame:
    frame = products[
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
            "product_photos_qty",
        ]
    ].copy()

    frame["product_volume_cm3"] = (
        pd.to_numeric(frame["product_width_cm"], errors="coerce")
        * pd.to_numeric(frame["product_height_cm"], errors="coerce")
        * pd.to_numeric(frame["product_length_cm"], errors="coerce")
    )

    frame = frame[
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_volume_cm3",
            "product_photos_qty",
        ]
    ]
    frame = frame.drop_duplicates().sort_values(["product_id"], kind="stable")
    frame = _assign_surrogate_key(frame, "product_key")
    return frame[
        [
            "product_key",
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_volume_cm3",
            "product_photos_qty",
        ]
    ]


def _build_payment_summary(payments: pd.DataFrame) -> pd.DataFrame:
    frame = payments[
        [
            "order_id",
            "payment_sequential",
            "payment_type",
            "payment_installments",
        ]
    ].copy()
    frame["payment_sequential"] = pd.to_numeric(
        frame["payment_sequential"], errors="coerce"
    )
    frame["payment_installments"] = pd.to_numeric(
        frame["payment_installments"], errors="coerce"
    )
    frame = frame.sort_values(["order_id", "payment_sequential"], kind="stable")
    return frame.groupby("order_id", as_index=False).agg(
        payment_type=("payment_type", "first"),
        max_payment_installments=("payment_installments", "max"),
    )


def _build_dim_order_info(orders: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    payment_summary = _build_payment_summary(payments)

    frame = orders[
        [
            "order_id",
            "order_status",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].copy()
    frame["order_delivered_customer_date"] = _safe_to_datetime(
        frame["order_delivered_customer_date"]
    )
    frame["order_estimated_delivery_date"] = _safe_to_datetime(
        frame["order_estimated_delivery_date"]
    )
    frame = frame.merge(payment_summary, on="order_id", how="left")
    frame["is_late_delivery"] = (
        frame["order_delivered_customer_date"] > frame["order_estimated_delivery_date"]
    ).astype("boolean")

    frame = frame[
        [
            "order_status",
            "payment_type",
            "max_payment_installments",
            "is_late_delivery",
        ]
    ]
    frame = frame.drop_duplicates().sort_values(
        ["order_status", "payment_type", "max_payment_installments"],
        kind="stable",
    )
    frame = _assign_surrogate_key(frame, "order_info_key")
    return frame[
        [
            "order_info_key",
            "order_status",
            "payment_type",
            "max_payment_installments",
            "is_late_delivery",
        ]
    ]


def _build_dim_date(orders: pd.DataFrame) -> pd.DataFrame:
    frame = orders[["order_approved_at"]].copy()
    frame["full_date"] = _safe_to_datetime(frame["order_approved_at"]).dt.normalize()
    frame = frame.dropna(subset=["full_date"]).drop_duplicates(subset=["full_date"])
    frame = frame.sort_values(["full_date"], kind="stable")

    frame["date_key"] = frame["full_date"].dt.strftime("%Y%m%d").astype(int)
    frame["day"] = frame["full_date"].dt.day.astype("Int64")
    frame["month"] = frame["full_date"].dt.month.astype("Int64")
    frame["year"] = frame["full_date"].dt.year.astype("Int64")
    frame["quarter"] = frame["full_date"].dt.quarter.astype("Int64")
    frame["day_of_week"] = frame["full_date"].dt.dayofweek.astype("Int64")
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype("boolean")
    frame["is_holiday"] = False

    return frame[
        [
            "date_key",
            "full_date",
            "day",
            "month",
            "year",
            "quarter",
            "day_of_week",
            "is_weekend",
            "is_holiday",
        ]
    ]


def _build_fact_sales(
    items: pd.DataFrame,
    orders: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_product: pd.DataFrame,
    dim_seller: pd.DataFrame,
    dim_order_info: pd.DataFrame,
    dim_date: pd.DataFrame,
    payments: pd.DataFrame,
) -> pd.DataFrame:
    payment_summary = _build_payment_summary(payments)

    order_level = orders[
        [
            "order_id",
            "customer_id",
            "order_status",
            "order_approved_at",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].copy()
    order_level["order_approved_at"] = _safe_to_datetime(
        order_level["order_approved_at"]
    )
    order_level["order_delivered_customer_date"] = _safe_to_datetime(
        order_level["order_delivered_customer_date"]
    )
    order_level["order_estimated_delivery_date"] = _safe_to_datetime(
        order_level["order_estimated_delivery_date"]
    )
    order_level = order_level.merge(payment_summary, on="order_id", how="left")
    order_level["is_late_delivery"] = (
        order_level["order_delivered_customer_date"]
        > order_level["order_estimated_delivery_date"]
    ).astype("boolean")

    fact = items[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ]
    ].copy()
    fact = fact.merge(order_level, on="order_id", how="left")
    fact = fact.merge(
        dim_customer[["customer_key", "customer_id"]],
        on="customer_id",
        how="left",
    )
    fact = fact.merge(
        dim_product[["product_key", "product_id"]],
        on="product_id",
        how="left",
    )
    fact = fact.merge(
        dim_seller[["seller_key", "seller_id"]],
        on="seller_id",
        how="left",
    )

    fact = fact.merge(
        dim_order_info,
        on=[
            "order_status",
            "payment_type",
            "max_payment_installments",
            "is_late_delivery",
        ],
        how="left",
    )

    fact["full_date"] = fact["order_approved_at"].dt.normalize()
    fact = fact.merge(
        dim_date[["date_key", "full_date"]],
        on="full_date",
        how="left",
    )
    fact["price"] = pd.to_numeric(fact["price"], errors="coerce")
    fact["freight_value"] = pd.to_numeric(fact["freight_value"], errors="coerce")
    fact["total_item_value"] = fact["price"] + fact["freight_value"]

    fact = fact.rename(columns={"order_item_id": "order_id_item"})
    return (
        fact[
            [
                "date_key",
                "customer_key",
                "product_key",
                "seller_key",
                "order_info_key",
                "order_id",
                "order_id_item",
                "price",
                "freight_value",
                "total_item_value",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def transform(
    input_dir: str | Path = STAGING_DIR,
    output_dir: str | Path = COMPLETED_DIR,
) -> dict[str, str]:

    print(f"[Transform] Transforming...")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    latest_files = _get_latest_file_in_directory(input_dir)
    missing_tables = sorted(SOURCE - set(latest_files))
    if missing_tables:
        discovered_files = sorted(path.name for path in input_dir.glob("*.parquet"))
        raise ValueError(
            f"[Transform] Warning: Missing staging files for tables: {missing_tables}. "
            f"[Transform] Warning: Found parquet files in {input_dir}: {discovered_files}"
        )

    customers = pd.read_parquet(latest_files["olist_order_customers"])
    items = pd.read_parquet(latest_files["olist_order_items"])
    payments = pd.read_parquet(latest_files["olist_order_payments"])
    orders = pd.read_parquet(latest_files["olist_orders"])
    products = pd.read_parquet(latest_files["olist_products"])
    sellers = pd.read_parquet(latest_files["olist_sellers"])

    dim_customer = _build_dim_customer(customers)
    dim_product = _build_dim_product(products)
    dim_seller = _build_dim_seller(sellers)
    dim_order_info = _build_dim_order_info(orders, payments)
    dim_date = _build_dim_date(orders)
    fact_sales = _build_fact_sales(
        items=items,
        orders=orders,
        dim_customer=dim_customer,
        dim_product=dim_product,
        dim_seller=dim_seller,
        dim_order_info=dim_order_info,
        dim_date=dim_date,
        payments=payments,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs = {
        "dim_date": dim_date,
        "dim_customer": dim_customer,
        "dim_seller": dim_seller,
        "dim_product": dim_product,
        "dim_order_info": dim_order_info,
        "fact_sales": fact_sales,
    }

    written_files: dict[str, str] = {}
    for table_name, frame in outputs.items():
        output_path = output_dir / f"{table_name}_{timestamp}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(
            output_path, engine="pyarrow", compression="snappy", index=False
        )
        written_files[table_name] = str(output_path)
        print(f"[Transform] Saved {table_name}: {output_path}")

    print(f"[Transform] Completed")

    return written_files


if __name__ == "__main__":
    transform()
