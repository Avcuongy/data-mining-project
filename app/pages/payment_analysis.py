import streamlit as st
from app.utils.db import query


def page():
    st.title("Payment Channels — Analysis")

    st.write("Analyze revenue and counts by payment type.")

    sql = """
    SELECT doi.payment_type, SUM(fs.total_item_value) AS revenue, COUNT(*) AS orders
    FROM DataWarehouse.fact_sales AS fs
    LEFT JOIN DataWarehouse.dim_order_info AS doi USING(order_info_key)
    GROUP BY doi.payment_type
    ORDER BY revenue DESC
    """
    df = query(sql)

    if df.empty:
        st.info("No data available")
        return

    st.dataframe(df)
    st.altair_chart(
        (  # simple bar chart using altair
            (df.loc[:, ["payment_type", "revenue"]]
             .rename(columns={"payment_type": "Payment", "revenue": "Revenue"}))
        ).pipe(lambda d: d), use_container_width=True)
