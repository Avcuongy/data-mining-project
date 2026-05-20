import streamlit as st
from app.utils.db import query
import pandas as pd


def page():
    st.title("Overview — Sales Performance")

    # KPIs
    sql_kpis = """
    SELECT
      SUM(total_item_value) AS revenue,
      COUNT(*) AS qty
    FROM DataWarehouse.fact_sales
    """
    kpis = query(sql_kpis).iloc[0]

    # approximate profit placeholder (20% margin)
    revenue = float(kpis["revenue"] or 0)
    qty = int(kpis["qty"] or 0)
    profit = revenue * 0.2

    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", f"{revenue:,.0f}")
    col2.metric("Quantity", f"{qty:,}")
    col3.metric("Estimated Profit", f"{profit:,.0f}")

    # Time series by month
    sql_ts = """
    SELECT d.year, d.month, SUM(fs.total_item_value) AS revenue, COUNT(*) AS qty
    FROM DataWarehouse.fact_sales AS fs
    LEFT JOIN DataWarehouse.dim_date AS d USING(date_key)
    GROUP BY d.year, d.month
    ORDER BY d.year, d.month
    """
    ts = query(sql_ts)
    if not ts.empty:
        ts["date"] = pd.to_datetime(ts["year"].astype(str) + "-" + ts["month"].astype(str) + "-01")
        st.line_chart(ts.set_index("date")["revenue"], height=300)
        st.bar_chart(ts.set_index("date")["qty"], height=200)

    st.markdown("---")
    st.subheader("Top Product Categories by Revenue")
    sql_cat = """
    SELECT dp.product_category_name, SUM(fs.total_item_value) AS revenue
    FROM DataWarehouse.fact_sales AS fs
    LEFT JOIN DataWarehouse.dim_product AS dp USING(product_key)
    GROUP BY dp.product_category_name
    ORDER BY revenue DESC
    LIMIT 20
    """
    cat = query(sql_cat)
    st.dataframe(cat)
