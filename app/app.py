import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="Olist Customer Segmentation", layout="wide")

# ==========================================
# KHỞI TẠO ĐƯỜNG DẪN & TẢI MÔ HÌNH/ DATA
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data_warehouse.duckdb"

import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.utils.hcubing import build_iceberg_sales_growth_cube, build_iceberg_logistics_risk_cube
except ImportError:
    st.error("Không nạp được thư viện xây dựng Iceberg Cube")


@st.cache_resource
def load_models():
    kmeans = joblib.load(MODEL_DIR / 'kmeans_model.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    pca = joblib.load(MODEL_DIR / 'pca.pkl')
    return kmeans, scaler, pca

try:
    kmeans_model, scaler, pca = load_models()
    is_model_loaded = True
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    is_model_loaded = False

@st.cache_data
def load_data():
    if DB_PATH.exists():
        import duckdb
        con = duckdb.connect(str(DB_PATH))
        df = con.execute("SELECT * FROM olist_customers").df()
        con.close()
        return df
    return None

filtered_df = load_data()

numerical_features = [
    'frequency', 'recency', 'monetary', 'total_item',
    'unique_category', 'payment_type_diversity',
    'avg_delivery_days', 'avg_shipping_delay',
    'late_delivery_ratio', 'avg_freight_ratio',
    'avg_freight_value', 'avg_estimated_gap'
]

def get_business_recommendation(cluster_id):
    recommendations = {
        0: {
            "name": "Khách hàng Phổ thông / Trung thành (Moderate Value)",
            "desc": "Mua sắm đặn, giá trị vừa phải, ổn định.",
            "actions": [
                "Chăm sóc định kỳ qua Email/SMS (Newsletter).",
                "Cross-selling: Giới thiệu các sản phẩm thuộc danh mục mới.",
                "Tặng mã freeship sau N đơn hàng."
            ]
        },
        1: {
            "name": "Khách hàng Rời bỏ / Ít giá trị (Low Value / Churn)",
            "desc": "Recency cao (đã lâu không mua), Monetary thấp. Nguy cơ rời bỏ rất cao hoặc ít tương tác.",
            "actions": [
                "Gửi email re-engagement với mã giảm giá sốc.",
                "Khảo sát để tìm hiểu nguyên nhân không quay lại.",
                "Đẩy mạnh quảng cáo các sản phẩm thiết yếu."
            ]
        },
        2: {
            "name": "Khách hàng VIP (High Value / Champions)",
            "desc": "Giá trị giỏ hàng rất cao, tần suất cao, là nhóm mang lại doanh thu chính.",
            "actions": [
                "Khởi tạo chương trình Loyalty Program riêng (VIP Tier).",
                "Chăm sóc cá nhân hoá 1-1, chúc mừng sinh nhật.",
                "Ưu tiên xử lý đơn và đóng gói đặc biệt.",
                "Mời tham gia trải nghiệm sản phẩm cao cấp sớm."
            ]
        }
    }
    return recommendations.get(cluster_id, {"name": "Không xác định", "desc": "", "actions": []})

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
# Giao diện Tabs
st.title("Nền tảng Quản lý Khách Hàng")

tab_bi, tab_cube, tab3, tab4 = st.tabs([
    "Business Intelligence (BI)", 
    "OLAP & Iceberg Cube",
    "Mô Hình Clustering", 
    "Ứng Dụng Khuyến Nghị"
])

# == MENU 1: BI DASHBOARD ==
with tab_bi:
    st.title("1. Business Intelligence (BI) Dashboard")
    st.markdown("Tổng quan các chỉ số quan trọng của khách hàng.")
    if filtered_df is not None:
        # Row 1: RFM Metrics
        st.subheader("Chỉ số RFM & Tổng quan")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng số Khách hàng", f"{filtered_df.shape[0]:,}")
        col2.metric("Trung bình Chi tiêu (M)", f"${filtered_df['monetary'].mean():,.2f}")
        col3.metric("Trung bình Tần suất (F)", f"{filtered_df['frequency'].mean():,.2f} lần")
        col4.metric("Trung bình Recency (R)", f"{filtered_df['recency'].mean():,.0f} ngày")
        
        # Row 2: Logistics Metrics
        st.subheader("Chỉ số Vận hành & Giao hàng")
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Trung bình Phí vận chuyển", f"${filtered_df['avg_freight_value'].mean():,.2f}")
        col6.metric("Thời gian Giao hàng (Ngày)", f"{filtered_df['avg_delivery_days'].mean():,.1f}")
        col7.metric("Độ trễ Giao hàng TB", f"{filtered_df['avg_shipping_delay'].mean():,.1f} ngày")
        col8.metric("Tỷ lệ Trễ hẹn", f"{filtered_df['late_delivery_ratio'].mean()*100:,.1f}%")
        
        st.write("---")
        st.subheader("Nhận định Dữ liệu Tổng quan (Key Insights)")
        st.info("""
        - **Hành vi Mua sắm:** Khách hàng chủ yếu là người mua một lần (one-time buyer), giỏ hàng thường chỉ có 1-2 sản phẩm và tập trung mua ở 1 danh mục duy nhất. Tỷ lệ giữ chân (retention) thấp.
        - **Địa lý:** Tệp khách hàng tập trung rất lớn ở khu vực Đông Nam Bộ của Brazil, đặc biệt là bang São Paulo (SP) và Rio de Janeiro (RJ).
        - **Thanh toán:** Thẻ tín dụng (Credit Card) là phương thức thanh toán thống trị tuyệt đối, theo sau là Boleto. Rất hiếm khách hàng sử dụng đa dạng phương thức thanh toán.
        - **Logistics:** Mặc dù trung bình mất khoảng 12 ngày để nhận hàng, đội ngũ giao hàng có hiệu suất khá tốt vì phần lớn đơn hàng được giao sớm hơn hoặc đúng so với thời gian dự kiến (shipping delay thường mang giá trị âm).
        """)

        st.write("---")
        st.subheader("Phân bố Dữ liệu (Distributions)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Chi tiêu (Monetary)**")
            fig, ax = plt.subplots(figsize=(6,4))
            skewness = filtered_df['monetary'].skew()
            sns.histplot(filtered_df['monetary'], bins=30, kde=True, color='skyblue', ax=ax)
            ax.set_xlim(0, filtered_df['monetary'].quantile(0.95))
            ax.set_title(f'monetary\nSkewness: {skewness:.2f}')
            ax.set_xlabel("Monetary")
            ax.set_ylabel("Số lượng Khách hàng")
            st.pyplot(fig)
        with c2:
            st.write("**Ngày từ lần mua cuối (Recency)**")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            skewness = filtered_df['recency'].skew()
            sns.histplot(filtered_df['recency'], bins=30, kde=True, color='salmon', ax=ax2)
            ax2.set_title(f'recency\nSkewness: {skewness:.2f}')
            ax2.set_xlabel("Recency")
            ax2.set_ylabel("")
            st.pyplot(fig2)
        with c3:
            st.write("**Thời gian Giao hàng (Days)**")
            fig_del, ax_del = plt.subplots(figsize=(6,4))
            skewness = filtered_df['avg_delivery_days'].skew()
            sns.histplot(filtered_df['avg_delivery_days'], bins=30, kde=True, color='lightgreen', ax=ax_del)
            ax_del.set_xlim(0, filtered_df['avg_delivery_days'].quantile(0.95))
            ax_del.set_title(f'avg_delivery_days\nSkewness: {skewness:.2f}')
            ax_del.set_xlabel("Average Delivery Days")
            ax_del.set_ylabel("")
            st.pyplot(fig_del)
            
        st.write("---")
        st.subheader("Phân tích Mối quan hệ (Relationships)")
        c4, c5 = st.columns(2)
        with c4:
            st.write("**Monetary vs Frequency**")
            fig4, ax4 = plt.subplots(figsize=(6,4))
            sample_df = filtered_df.sample(min(5000, len(filtered_df)), random_state=42)
            sns.scatterplot(data=sample_df, x='frequency', y='monetary', alpha=0.5, color='purple', ax=ax4)
            ax4.set_ylim(0, sample_df['monetary'].quantile(0.99))
            ax4.set_xlim(0, sample_df['frequency'].quantile(0.99))
            st.pyplot(fig4)
            
        with c5:
            st.write("**Recency vs Monetary**")
            fig5, ax5 = plt.subplots(figsize=(6,4))
            sns.scatterplot(data=sample_df, x='recency', y='monetary', alpha=0.5, color='orange', ax=ax5)
            ax5.set_ylim(0, sample_df['monetary'].quantile(0.99))
            st.pyplot(fig5)

        st.write("---")
        st.subheader("Phân tích Dữ liệu Phân loại (Categorical Data)")
        cat1, cat2, cat3 = st.columns(3)
        with cat1:
            st.write("**Top 15 Bang (State) có lượng KH nhiều nhất**")
            fig_state, ax_state = plt.subplots(figsize=(6,5))
            value_counts = filtered_df['customer_state'].value_counts().head(15).sort_values()
            value_counts.plot(kind='barh', color='steelblue', ax=ax_state)
            for container in ax_state.containers:
                ax_state.bar_label(container, fmt='%d', padding=3)
            ax_state.set_title('Distribution of customer_state')
            ax_state.set_xlabel('Count')
            ax_state.set_ylabel('customer_state')
            st.pyplot(fig_state)
        with cat2:
            st.write("**Top 15 Danh mục Yêu thích nhất**")
            fig_cat, ax_cat = plt.subplots(figsize=(6,5))
            value_counts = filtered_df['favorite_category'].value_counts().head(15).sort_values()
            value_counts.plot(kind='barh', color='steelblue', ax=ax_cat)
            for container in ax_cat.containers:
                ax_cat.bar_label(container, fmt='%d', padding=3)
            ax_cat.set_title('Distribution of favorite_category')
            ax_cat.set_xlabel('Count')
            ax_cat.set_ylabel('favorite_category')
            st.pyplot(fig_cat)
        with cat3:
            st.write("**Phương thức Thanh toán Chính**")
            fig_pay, ax_pay = plt.subplots(figsize=(6,5))
            value_counts = filtered_df['dominant_payment_type'].value_counts().head(15).sort_values()
            value_counts.plot(kind='barh', color='steelblue', ax=ax_pay)
            for container in ax_pay.containers:
                ax_pay.bar_label(container, fmt='%d', padding=3)
            ax_pay.set_title('Distribution of dominant_payment_type')
            ax_pay.set_xlabel('Count')
            ax_pay.set_ylabel('dominant_payment_type')
            st.pyplot(fig_pay)

        st.write("---")
        st.subheader("Phân tích Chuyên sâu (Advanced)")
        c6, c7 = st.columns([2, 1])
        with c6:
            st.write("**Ma trận tương quan (Correlation)**")
            fig3, ax3 = plt.subplots(figsize=(10,6))
            corr_matrix = filtered_df[numerical_features].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3, annot_kws={"size": 8})
            st.pyplot(fig3)
        with c7:
            st.write("**Thống kê Mô tả Đặc trưng**")
            st.dataframe(filtered_df[numerical_features].describe().T[['mean', 'std', 'min', 'max']].style.format("{:.2f}"))
            
    else:
        st.error("Không tìm thấy dữ liệu. Hệ thống yêu cầu file Dữ liệu đã làm sạch.")

# == MENU 2: OLAP CUBE ==
with tab_cube:
    st.title("2. Phân Tích OLAP & Iceberg Cubing")
    st.markdown("Truy vấn trực tiếp lên Data Warehouse (DuckDB) để xây dựng các khối dữ liệu Iceberg Cube đa chiều xử lý những bài toán kinh doanh lớn. (Chạy hàm cấu trúc `src.utils.hcubing`)")
    
    col_cube1, col_cube2 = st.columns(2)
    with col_cube1:
        st.subheader("🧊 Sales Growth Iceberg Cube")
        st.markdown("**Dimensions:** year, payment_type, customer_state, product_category_name")
        st.markdown("**Measures:** total_revenue, qty_items")
        
        min_sup_sales = st.slider("Ngưỡng Min Support (Doanh thu)", min_value=0.1, max_value=0.9, value=0.1, step=0.1, key='sup_sales')
        top_k_sales = st.number_input("Top K giao dịch cao nhất", min_value=0, value=0, step=1, key='k_sales')
        
        if st.button("Xây dựng Sales Cube"):
            with st.spinner("Đang tính toán Iceberg Cube..."):
                k_val = top_k_sales if top_k_sales > 0 else None
                try:
                    df_sales_cube = build_iceberg_sales_growth_cube(min_sup_sales, k_val, str(DB_PATH))
                    st.success(f"Cube đã tạo: {df_sales_cube.shape[0]} luật thoả mãn min_sup.")
                    st.dataframe(df_sales_cube.style.background_gradient(subset=['total_revenue'], cmap='Greens'))
                except Exception as e:
                    st.error(f"Lỗi truy vấn: {e}")

    with col_cube2:
        st.subheader("🧊 Logistics Risk Iceberg Cube")
        st.markdown("**Dimensions:** year, seller_state, customer_state, product_category_name")
        st.markdown("**Measures:** late_orders, total_freight")
        
        min_sup_log = st.slider("Ngưỡng Min Support (Tổng trễ)", min_value=0.1, max_value=0.9, value=0.1, step=0.1, key='sup_log')
        top_k_log = st.number_input("Top K phí ship cao nhất", min_value=0, value=0, step=1, key='k_log')

        if st.button("Xây dựng Logistics Cube"):
            with st.spinner("Đang tính toán Iceberg Cube..."):
                k_val = top_k_log if top_k_log > 0 else None
                try:
                    df_log_cube = build_iceberg_logistics_risk_cube(min_sup_log, k_val, str(DB_PATH))
                    st.success(f"Cube đã tạo: {df_log_cube.shape[0]} luật thoả mãn min_sup.")
                    st.dataframe(df_log_cube.style.background_gradient(subset=['late_orders'], cmap='Reds'))
                except Exception as e:
                    st.error(f"Lỗi truy vấn: {e}")

# == MENU 3: MACHINE LEARNING ==
with tab3:
    st.title("3. Triển Khai Mô Hình (Clustering)")
    st.markdown("*Kết xuất từ Notebook `x_ml` - Sử dụng K-Means và trực quan hóa qua PCA.*")
    
    if is_model_loaded and filtered_df is not None:
        st.write("---")
        st.subheader("Cập nhật Cluster theo K-Means (K=3)")
        
        with st.spinner('Đang dự đoán lại trên toàn bộ khối dữ liệu...'):
            X = filtered_df[numerical_features].fillna(0)
            X_scaled = scaler.transform(X)
            preds = kmeans_model.predict(X_scaled)
            X_pca = pca.transform(X_scaled)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
            
            # Scatter Plot 
            scatter = ax1.scatter(X_pca[:,0], X_pca[:,1], c=preds, cmap='tab10', alpha=0.5, s=10)
            ax1.set_title("Trực quan không gian 2D bằng PCA")
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            plt.colorbar(scatter, ax=ax1, label='Cluster')
            
            # Cỡ cụm
            clust_counts = pd.Series(preds).value_counts().sort_index()
            sns.barplot(x=clust_counts.index, y=clust_counts.values, palette='tab10', ax=ax2)
            ax2.set_title("Kích thước tệp Khách hàng mỗi Cụm")
            st.pyplot(fig)
            
            st.subheader("Bảng Mô Tả Trung Bình Chân Dung Cụm (Cluster Profile)")
            filtered_df['Cluster'] = preds
            cluster_grouped = filtered_df.groupby('Cluster')[numerical_features].mean()
            st.dataframe(cluster_grouped.style.background_gradient(cmap='viridis'))
    else:
        st.warning("Hệ thống Machine Learning hiện không khả dụng (Thiếu mô hình).")

# == MENU 4: INFERENCE / ỨNG DỤNG ==
with tab4:
    st.title("4. Nhập Liệu Mô Phỏng Khách Hàng (Inference)")
    st.markdown("Cung cấp một mẫu đối tượng Khách hàng giả định để hệ thống tự động suy luận ra tệp đối tượng.")
    
    if is_model_loaded:
        with st.form("single_pred_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                frequency = st.number_input("Tần suất mua hàng (frequency)", min_value=1, value=1)
                recency = st.number_input("Số này kể từ lần mua cuối (recency)", min_value=0, value=30)
                monetary = st.number_input("Tổng tiền chi tiêu (monetary)", min_value=0.0, value=100.0)
                total_item = st.number_input("Tổng sản phẩm đã mua (total_item)", min_value=1, value=2)
            with col2:
                unique_category = st.number_input("Số danh mục đã mua (unique)", min_value=1, value=1)
                payment_type_diversity = st.number_input("Đa dạng thanh toán", min_value=1, value=1)
                avg_delivery_days = st.number_input("Ngày giao hàng (avg_delivery)", min_value=0.0, value=10.0)
                avg_shipping_delay = st.number_input("Độ trễ TB (avg_delay)", min_value=-50.0, value=-2.0)
            with col3:
                late_delivery_ratio = st.number_input("Tỷ lệ trễ hẹn (late_ratio)", min_value=0.0, max_value=1.0, value=0.0)
                avg_freight_ratio = st.number_input("Tỷ lệ phí ship (avg_freight)", min_value=0.0, value=0.15)
                avg_freight_value = st.number_input("Phí ship trung bình", min_value=0.0, value=15.0)
                avg_estimated_gap = st.number_input("Sai số ước tính (avg_gap)", min_value=0.0, value=5.0)
            
            submit = st.form_submit_button("Dự đoán Chân dung Khách Hàng")
            
        if submit:
            input_data = pd.DataFrame([
                [frequency, recency, monetary, total_item, unique_category, 
                 payment_type_diversity, avg_delivery_days, avg_shipping_delay, late_delivery_ratio, 
                 avg_freight_ratio, avg_freight_value, avg_estimated_gap]
            ], columns=numerical_features)
            
            X_scaled = scaler.transform(input_data)
            cluster_id = kmeans_model.predict(X_scaled)[0]
            
            st.success(f"Dự báo hệ thống: Thể loại Khách hàng CỤM {cluster_id}")
            biz_rec = get_business_recommendation(cluster_id)
            st.markdown(f"**Chân dung & Hướng đi Marketing:** {biz_rec['name']}")
            st.markdown(f"_{biz_rec['desc']}_")
            for act in biz_rec['actions']:
                st.markdown(f"- {act}")


