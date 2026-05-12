CREATE SCHEMA IF NOT EXISTS DataWarehouse;

SET schema
    = 'DataWarehouse';

-- DIM_CUSTOMER
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key INTEGER PRIMARY KEY,
    -- Surrogate key
    customer_id VARCHAR,
    customer_unique_id VARCHAR,
    customer_city VARCHAR,
    customer_state VARCHAR,
    -- SCD2
    is_current BOOLEAN
);

-- DIM_SELLER
CREATE TABLE IF NOT EXISTS dim_seller (
    seller_key INTEGER PRIMARY KEY,
    -- Surrogate key
    seller_id VARCHAR,
    seller_city VARCHAR,
    seller_state VARCHAR,
    -- SCD2
    is_current BOOLEAN
);

-- DIM_PRODUCT
CREATE TABLE IF NOT EXISTS dim_product (
    product_key INTEGER PRIMARY KEY,
    -- Surrogate key
    product_id VARCHAR,
    product_category_name VARCHAR,
    product_weight_g FLOAT,
    product_volume_cm3 FLOAT,
    product_photos_qty INTEGER
);

-- DIM_ORDER_INFO
CREATE TABLE IF NOT EXISTS dim_order_info (
    order_info_key INTEGER PRIMARY KEY,
    -- Surrogate key
    order_status VARCHAR,
    payment_type VARCHAR,
    max_payment_installments INTEGER,
    is_late_delivery BOOLEAN
);

-- DIM_DATE
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    -- Surrogate key
    full_date DATE,
    day INTEGER,
    month INTEGER,
    year INTEGER,
    quarter INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN
);

-- FACT_SALES
CREATE TABLE IF NOT EXISTS fact_sales (
    -- Degenerate Dimensions
    order_id VARCHAR,
    order_id_item INTEGER,
    -- Foreign Keys
    date_key INTEGER,
    customer_key INTEGER,
    product_key INTEGER,
    seller_key INTEGER,
    order_info_key INTEGER,
    -- Measures
    price FLOAT,
    freight_value FLOAT,
    total_item_value FLOAT,
    -- Foreign Keys constraints
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
    FOREIGN KEY (seller_key) REFERENCES dim_seller(seller_key),
    FOREIGN KEY (order_info_key) REFERENCES dim_order_info(order_info_key)
);