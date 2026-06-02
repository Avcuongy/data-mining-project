from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

plt.style.use("dark_background")


st.set_page_config(
    page_title="Olist Data Command Center",
    layout="wide",
    page_icon="📊",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "etl" / "staging"
COMPLETED_DIR = DATA_DIR / "etl" / "completed"
CLEANED_DIR = DATA_DIR / "cleaned"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data_warehouse.duckdb"
WAREHOUSE_SCHEMA = "DataWarehouse"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from src.utils.hcubing import (
        build_iceberg_logistics_risk_cube,
        build_iceberg_sales_growth_cube,
    )
except ImportError as exc:
    st.error(f"Không nạp được thư viện iceberg cube: {exc}")
    raise


CUSTOMER_FILE = CLEANED_DIR / "olist_customers.csv"

NUMERICAL_FEATURES = [
    "frequency",
    "recency",
    "monetary",
    "total_item",
    "unique_category",
    "payment_type_diversity",
    "avg_delivery_days",
    "avg_shipping_delay",
    "late_delivery_ratio",
    "avg_freight_ratio",
    "avg_freight_value",
    "avg_estimated_gap",
]

CLUSTER_MODEL_FILES = {
    "K-Means": "kmeans_model.pkl",
    "BIRCH": "birch_model.pkl",
    "MiniBatchKMeans": "minibatch_kmeans_model.pkl",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 28%),
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.14), transparent 26%),
                    linear-gradient(180deg, #060816 0%, #0a1020 52%, #05070f 100%);
                color: #e5eefc;
            }
            .hero-card {
                padding: 1.5rem 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #0b1220 0%, #12264a 48%, #064e3b 100%);
                color: white;
                box-shadow: 0 18px 48px rgba(0, 0, 0, 0.42);
                margin-bottom: 1rem;
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
            .hero-eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.72rem;
                opacity: 0.78;
                margin-bottom: 0.4rem;
            }
            .hero-title {
                font-size: 2.2rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 0.5rem;
            }
            .hero-copy {
                font-size: 0.98rem;
                opacity: 0.92;
                max-width: 70rem;
            }
            .section-shell {
                background: rgba(9, 14, 28, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 20px;
                padding: 1rem 1.1rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                color: #e5eefc;
            }
            div[data-testid="stMetric"] {
                background: rgba(9, 14, 28, 0.95);
                border-radius: 16px;
                border: 1px solid rgba(148, 163, 184, 0.16);
                padding: 0.8rem 0.9rem;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.24);
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 0.55rem 1rem;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(37, 99, 235, 0.28), rgba(16, 185, 129, 0.22));
            }
            div[data-testid="stDataFrame"] {
                background: rgba(8, 12, 24, 0.96);
                border: 1px solid rgba(148, 163, 184, 0.14);
                border-radius: 14px;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #070b16 0%, #090f1d 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.14);
            }
            .stMarkdown, .stCaption, p, label, span {
                color: #e5eefc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_bytes(size_bytes: float) -> str:
    if size_bytes <= 0:
        return "0 KB"
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:,.1f} KB"
    return f"{size_kb / 1024:,.2f} MB"


def _list_files(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _directory_summary(directory: Path, suffixes: tuple[str, ...]) -> dict[str, Any]:
    files = _list_files(directory, suffixes)
    total_size = sum(path.stat().st_size for path in files)
    latest = max((path.stat().st_mtime for path in files), default=None)
    return {
        "directory": directory,
        "files": files,
        "count": len(files),
        "total_size": total_size,
        "latest": latest,
    }


def _build_inventory_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage_name, directory, suffixes in [
        ("Raw", RAW_DIR, (".csv", ".parquet")),
        ("Staging", STAGING_DIR, (".parquet",)),
        ("Completed", COMPLETED_DIR, (".parquet",)),
        ("Cleaned", CLEANED_DIR, (".csv", ".parquet")),
    ]:
        summary = _directory_summary(directory, suffixes)
        rows.append(
            {
                "stage": stage_name,
                "path": str(directory.relative_to(BASE_DIR)),
                "files": summary["count"],
                "size": _format_bytes(summary["total_size"]),
                "latest": pd.to_datetime(summary["latest"], unit="s") if summary["latest"] else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_customer_dataset() -> pd.DataFrame:
    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file cleaned khách hàng: {CUSTOMER_FILE}")
    return pd.read_csv(CUSTOMER_FILE)


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "pca": joblib.load(MODEL_DIR / "pca.pkl"),
    }
    for model_name, file_name in CLUSTER_MODEL_FILES.items():
        artifacts[model_name] = joblib.load(MODEL_DIR / file_name)
    return artifacts


def _connect_warehouse() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute(f"SET schema = '{WAREHOUSE_SCHEMA}'")
    return conn


@st.cache_data(show_spinner=False)
def load_warehouse_table_counts() -> pd.DataFrame:
    conn = _connect_warehouse()
    try:
        table_names = [
            row[0]
            for row in conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = ?
                ORDER BY table_name
                """,
                [WAREHOUSE_SCHEMA],
            ).fetchall()
        ]
        rows = []
        for table_name in table_names:
            count = conn.execute(
                f"SELECT COUNT(*) FROM {WAREHOUSE_SCHEMA}.{table_name}"
            ).fetchone()[0]
            rows.append({"table": table_name, "rows": int(count)})
        return pd.DataFrame(rows)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_warehouse_preview(table_name: str, limit: int = 8) -> pd.DataFrame:
    conn = _connect_warehouse()
    try:
        return conn.execute(
            f"SELECT * FROM {WAREHOUSE_SCHEMA}.{table_name} LIMIT {int(limit)}"
        ).df()
    finally:
        conn.close()


def _apply_customer_filters(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    filtered = frame.copy()
    with st.sidebar:
        st.markdown("### Bộ lọc phân tích")
        state_options = sorted(filtered["customer_state"].dropna().unique().tolist())
        payment_options = sorted(filtered["dominant_payment_type"].dropna().unique().tolist())
        category_options = sorted(filtered["favorite_category"].dropna().unique().tolist())

        selected_states = st.multiselect(
            "Bang / State",
            state_options,
            default=state_options,
        )
        selected_payments = st.multiselect(
            "Phương thức thanh toán",
            payment_options,
            default=payment_options,
        )
        default_categories = category_options[: min(10, len(category_options))]
        selected_categories = st.multiselect(
            "Danh mục yêu thích",
            category_options,
            default=default_categories,
        )

        frequency_min, frequency_max = int(filtered["frequency"].min()), int(filtered["frequency"].max())
        recency_min, recency_max = int(filtered["recency"].min()), int(filtered["recency"].max())
        monetary_min, monetary_max = float(filtered["monetary"].min()), float(filtered["monetary"].max())

        frequency_range = st.slider(
            "Frequency",
            min_value=frequency_min,
            max_value=frequency_max,
            value=(frequency_min, frequency_max),
        )
        recency_range = st.slider(
            "Recency",
            min_value=recency_min,
            max_value=recency_max,
            value=(recency_min, recency_max),
        )
        monetary_range = st.slider(
            "Monetary",
            min_value=float(np.floor(monetary_min)),
            max_value=float(np.ceil(monetary_max)),
            value=(float(np.floor(monetary_min)), float(np.ceil(monetary_max))),
        )

        keep_high_value = st.checkbox("Ưu tiên khách hàng giá trị cao", value=False)

    if selected_states:
        filtered = filtered[filtered["customer_state"].isin(selected_states)]
    if selected_payments:
        filtered = filtered[filtered["dominant_payment_type"].isin(selected_payments)]
    if selected_categories:
        filtered = filtered[filtered["favorite_category"].isin(selected_categories)]

    filtered = filtered[
        filtered["frequency"].between(*frequency_range)
        & filtered["recency"].between(*recency_range)
        & filtered["monetary"].between(*monetary_range)
    ]

    if keep_high_value and not filtered.empty:
        threshold = filtered["monetary"].quantile(0.75)
        filtered = filtered[filtered["monetary"] >= threshold]

    filters = {
        "selected_states": selected_states,
        "selected_payments": selected_payments,
        "selected_categories": selected_categories,
        "frequency_range": frequency_range,
        "recency_range": recency_range,
        "monetary_range": monetary_range,
        "keep_high_value": keep_high_value,
    }
    return filtered, filters


def _safe_stat(frame: pd.DataFrame, column: str, reducer: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    series = pd.to_numeric(frame[column], errors="coerce")
    if reducer == "mean":
        return float(series.mean())
    if reducer == "median":
        return float(series.median())
    return float("nan")


def _persona_card(cluster_id: int, persona: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b;">Cụm {cluster_id}</div>
            <h3 style="margin-top:0.2rem;">{persona['name']}</h3>
            <p style="margin-bottom:0.5rem;">{persona['desc']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for action in persona["actions"]:
        st.markdown(f"- {action}")


def _build_persona_map(cluster_profile: pd.DataFrame) -> dict[int, dict[str, Any]]:
    normalized = cluster_profile.copy()
    for column in ["frequency", "recency", "monetary"]:
        series = normalized[column].astype(float)
        if series.std(ddof=0) == 0:
            normalized[f"{column}_z"] = 0.0
        else:
            normalized[f"{column}_z"] = (series - series.mean()) / series.std(ddof=0)

    normalized["vip_score"] = normalized["monetary_z"] + normalized["frequency_z"] - normalized["recency_z"]
    normalized["risk_score"] = normalized["recency_z"] - normalized["monetary_z"] - normalized["frequency_z"]

    vip_cluster = int(normalized["vip_score"].idxmax())
    risk_cluster = int(normalized["risk_score"].idxmax())
    remaining = [int(idx) for idx in normalized.index.tolist() if idx not in {vip_cluster, risk_cluster}]
    core_cluster = remaining[0] if remaining else vip_cluster

    personas = {
        vip_cluster: {
            "name": "Khách hàng VIP / Champions",
            "desc": "Giá trị cao, mua thường xuyên và đóng góp doanh thu tốt nhất.",
            "actions": [
                "Triển khai loyalty tier riêng và ưu tiên chăm sóc 1-1.",
                "Đề xuất sản phẩm cao cấp, bundle và ưu đãi độc quyền.",
                "Ưu tiên xử lý đơn và tặng đặc quyền vận chuyển.",
            ],
        },
        risk_cluster: {
            "name": "Khách hàng cần tái kích hoạt",
            "desc": "Recency cao, giá trị thấp hoặc tần suất mua giảm rõ rệt.",
            "actions": [
                "Gửi chiến dịch win-back với ưu đãi giới hạn thời gian.",
                "Tạo workflow khảo sát để tìm nguyên nhân rời bỏ.",
                "Cá nhân hoá nội dung theo danh mục quan tâm gần nhất.",
            ],
        },
        core_cluster: {
            "name": "Khách hàng lõi / ổn định",
            "desc": "Nhóm ở giữa, có tiềm năng nâng giá trị bằng cross-sell và upsell.",
            "actions": [
                "Cross-sell theo danh mục yêu thích và phương thức thanh toán.",
                "Khuyến khích mua lặp lại bằng voucher nhẹ.",
                "Theo dõi chuyển dịch hành vi để đẩy lên VIP.",
            ],
        },
    }
    return personas


def _render_kpis(frame: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Khách hàng", f"{len(frame):,}")
    col2.metric("Monetary TB", f"{_safe_stat(frame, 'monetary', 'mean'):,.2f}")
    col3.metric("Recency TB", f"{_safe_stat(frame, 'recency', 'mean'):,.0f} ngày")
    col4.metric("Frequency TB", f"{_safe_stat(frame, 'frequency', 'mean'):,.2f}")
    col5.metric("Shipping delay TB", f"{_safe_stat(frame, 'avg_shipping_delay', 'mean'):,.1f} ngày")


def _render_overview(frame: pd.DataFrame) -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("Tổng quan dữ liệu khách hàng")
    st.caption("Nguồn chính: file cleaned của ETL. Các bộ lọc ở sidebar đang điều chỉnh toàn bộ dashboard.")
    _render_kpis(frame)

    if frame.empty:
        st.warning("Không còn bản ghi nào sau khi áp bộ lọc.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("#### Phân bố theo bang")
        state_counts = frame["customer_state"].value_counts().head(12).sort_values()
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        sns.barplot(x=state_counts.values, y=state_counts.index, palette="crest", ax=ax)
        ax.set_xlabel("Số khách hàng")
        ax.set_ylabel("Bang")
        ax.set_title("Top bang theo số lượng khách hàng")
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.markdown("#### Phân bố thanh toán")
        payment_counts = frame["dominant_payment_type"].value_counts()
        pie_fig, pie_ax = plt.subplots(figsize=(6.2, 4.8))
        payment_counts.plot(kind="pie", autopct="%.1f%%", startangle=90, ax=pie_ax, legend=False)
        pie_ax.set_ylabel("")
        pie_ax.set_title("Phương thức thanh toán chủ đạo")
        st.pyplot(pie_fig, clear_figure=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Monetary vs Frequency")
        sample_df = frame.sample(min(4000, len(frame)), random_state=42)
        fig_scatter, ax_scatter = plt.subplots(figsize=(7.4, 4.8))
        sns.scatterplot(
            data=sample_df,
            x="frequency",
            y="monetary",
            hue="dominant_payment_type",
            alpha=0.55,
            s=28,
            ax=ax_scatter,
        )
        ax_scatter.set_title("Mối quan hệ tần suất và chi tiêu")
        ax_scatter.legend(loc="best", fontsize=8)
        st.pyplot(fig_scatter, clear_figure=True)
    with c4:
        st.markdown("#### Recency vs Monetary")
        fig_scatter2, ax_scatter2 = plt.subplots(figsize=(7.4, 4.8))
        sns.scatterplot(
            data=sample_df,
            x="recency",
            y="monetary",
            hue="customer_state",
            alpha=0.5,
            s=24,
            legend=False,
            ax=ax_scatter2,
        )
        ax_scatter2.set_title("Khách hàng lâu ngày vs giá trị đơn hàng")
        st.pyplot(fig_scatter2, clear_figure=True)

    with st.expander("Xem mô tả thống kê và dữ liệu lọc", expanded=False):
        st.dataframe(
            frame[NUMERICAL_FEATURES + ["customer_state", "favorite_category", "dominant_payment_type"]]
            .describe(include="all")
            .T,
            use_container_width=True,
        )
        st.dataframe(frame.head(200), use_container_width=True)
        st.download_button(
            "Tải CSV của tập dữ liệu đã lọc",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customers.csv",
            mime="text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_etl_tab() -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("ETL lineage và trạng thái dữ liệu")
    st.caption("Tab này cho thấy nguồn raw, staging, completed và cleaned đang được dashboard tiêu thụ.")

    inventory = _build_inventory_frame()
    left, right = st.columns([0.95, 1.05])
    with left:
        st.dataframe(inventory, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Warehouse tables")
        try:
            table_counts = load_warehouse_table_counts()
            st.dataframe(table_counts, use_container_width=True, hide_index=True)
            warehouse_tables = table_counts["table"].tolist()
            if warehouse_tables:
                preview_table = st.selectbox(
                    "Chọn bảng warehouse để xem mẫu",
                    warehouse_tables,
                    key="warehouse_preview_table",
                )
                st.dataframe(
                    load_warehouse_preview(preview_table, limit=10),
                    use_container_width=True,
                )
        except Exception as exc:
            st.warning(f"Không đọc được warehouse: {exc}")

    st.markdown("#### Xem mẫu dữ liệu từ từng giai đoạn")
    stage_map = {
        "Raw": RAW_DIR,
        "Staging": STAGING_DIR,
        "Completed": COMPLETED_DIR,
        "Cleaned": CLEANED_DIR,
    }
    stage_name = st.selectbox("Chọn giai đoạn", list(stage_map.keys()))
    directory = stage_map[stage_name]
    suffixes = (".csv", ".parquet") if stage_name in {"Raw", "Cleaned"} else (".parquet",)
    files = _list_files(directory, suffixes)
    if not files:
        st.info("Không có file nào trong thư mục này.")
    else:
        file_choice = st.selectbox("Chọn file", [path.name for path in files])
        file_path = directory / file_choice
        try:
            if file_path.suffix.lower() == ".csv":
                preview = pd.read_csv(file_path)
            else:
                preview = pd.read_parquet(file_path)
            st.dataframe(preview.head(20), use_container_width=True)
        except Exception as exc:
            st.error(f"Không đọc được file {file_choice}: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def _display_cube_result(frame: pd.DataFrame, measure_column: str, title: str) -> None:
    if frame is None or frame.empty:
        st.info("Chưa có kết quả cube để hiển thị.")
        return

    st.success(f"Đã tạo {len(frame):,} dòng cube.")
    axis_candidates = [
        column
        for column in frame.columns
        if column not in {measure_column, "avg_item_value", "avg_freight"}
        and (frame[column].dtype == object or column == "year")
    ]
    if not axis_candidates:
        axis_candidates = [frame.columns[0]]
    axis_name = st.selectbox(f"Trục phân tích cho {title}", axis_candidates, key=f"axis_{title}")
    grouped = frame.groupby(axis_name, dropna=False)[measure_column].sum().sort_values(ascending=False)

    chart_fig, chart_ax = plt.subplots(figsize=(7.5, 4.5))
    grouped.head(12).sort_values().plot(kind="barh", color="#0f766e", ax=chart_ax)
    chart_ax.set_title(f"{title} - top theo {axis_name}")
    chart_ax.set_xlabel(measure_column)
    st.pyplot(chart_fig, clear_figure=True)

    st.dataframe(frame, use_container_width=True)
    st.download_button(
        f"Tải {title}",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{title.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def _render_cube_tab() -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("OLAP & Iceberg cube")
    st.caption("Tạo cube trực tiếp từ warehouse. Mỗi cube có thể tái tạo lại theo ngưỡng min support và top-k.")

    sales_col, logistics_col = st.columns(2)
    if "sales_cube" not in st.session_state:
        st.session_state.sales_cube = None
    if "logistics_cube" not in st.session_state:
        st.session_state.logistics_cube = None

    with sales_col:
        st.markdown("#### Sales Growth Iceberg Cube")
        with st.form("sales_cube_form"):
            min_sup_sales = st.slider("Min support doanh thu", 0.1, 0.9, 0.2, 0.1, key="sales_min_sup")
            top_k_sales = st.number_input("Top-k giao dịch", min_value=0, value=0, step=1, key="sales_top_k")
            build_sales = st.form_submit_button("Xây dựng Sales Cube")
        if build_sales:
            with st.spinner("Đang xây dựng sales cube..."):
                try:
                    st.session_state.sales_cube = build_iceberg_sales_growth_cube(
                        min_sup_sales,
                        top_k_sales or None,
                        str(DB_PATH),
                    )
                except Exception as exc:
                    st.session_state.sales_cube = None
                    st.error(f"Không tạo được sales cube: {exc}")
        _display_cube_result(
            st.session_state.sales_cube,
            "total_revenue",
            "Sales Cube",
        )

    with logistics_col:
        st.markdown("#### Logistics Risk Iceberg Cube")
        with st.form("logistics_cube_form"):
            min_sup_logistics = st.slider("Min support trễ giao hàng", 0.1, 0.9, 0.2, 0.1, key="log_min_sup")
            top_k_logistics = st.number_input("Top-k phí ship", min_value=0, value=0, step=1, key="log_top_k")
            build_logistics = st.form_submit_button("Xây dựng Logistics Cube")
        if build_logistics:
            with st.spinner("Đang xây dựng logistics cube..."):
                try:
                    st.session_state.logistics_cube = build_iceberg_logistics_risk_cube(
                        min_sup_logistics,
                        top_k_logistics or None,
                        str(DB_PATH),
                    )
                except Exception as exc:
                    st.session_state.logistics_cube = None
                    st.error(f"Không tạo được logistics cube: {exc}")
        _display_cube_result(
            st.session_state.logistics_cube,
            "late_orders",
            "Logistics Cube",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_cluster_tab(frame: pd.DataFrame, artifacts: dict[str, Any]) -> None:
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("Mô hình phân cụm")
    st.caption("Chọn model, xem profile cụm, rồi suy luận một khách hàng mẫu ngay trên dashboard.")

    if frame.empty:
        st.warning("Không có dữ liệu sau khi lọc để chạy clustering.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    model_name = st.selectbox("Chọn model", list(CLUSTER_MODEL_FILES.keys()))
    model = artifacts.get(model_name)
    scaler = artifacts.get("scaler")
    pca = artifacts.get("pca")

    if model is None or scaler is None or pca is None:
        st.error("Thiếu model, scaler hoặc PCA.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    sample_size = st.slider(
        "Kích thước mẫu để vẽ",
        min_value=500,
        max_value=max(500, len(frame)),
        value=min(3000, len(frame)),
        step=100,
    )
    working_frame = frame.sample(min(sample_size, len(frame)), random_state=42).copy() if len(frame) > sample_size else frame.copy()

    X = working_frame[NUMERICAL_FEATURES].fillna(0)
    X_scaled = scaler.transform(X)
    cluster_ids = model.predict(X_scaled)
    X_pca = pca.transform(X_scaled)

    cluster_frame = working_frame.copy()
    cluster_frame["cluster"] = cluster_ids
    cluster_profile = cluster_frame.groupby("cluster")[NUMERICAL_FEATURES].mean().round(2)
    personas = _build_persona_map(cluster_profile)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số cụm", f"{len(np.unique(cluster_ids))}")
    c2.metric("Mẫu phân tích", f"{len(cluster_frame):,}")
    c3.metric("Model", model_name)
    c4.metric("PCA dims", "2")

    plot_left, plot_right = st.columns([1.15, 0.85])
    with plot_left:
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_ids, cmap="tab10", s=18, alpha=0.65)
        ax.set_title("PCA projection theo cụm")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig, clear_figure=True)
    with plot_right:
        st.markdown("#### Kích thước cụm")
        cluster_sizes = pd.Series(cluster_ids).value_counts().sort_index()
        size_fig, size_ax = plt.subplots(figsize=(5.5, 5.2))
        sns.barplot(x=cluster_sizes.index.astype(str), y=cluster_sizes.values, palette="tab10", ax=size_ax)
        size_ax.set_xlabel("Cluster")
        size_ax.set_ylabel("Số khách hàng")
        size_ax.set_title("Phân bố kích thước cụm")
        st.pyplot(size_fig, clear_figure=True)

    st.markdown("#### Profile trung bình của từng cụm")
    st.dataframe(cluster_profile, use_container_width=True)

    with st.expander("Diễn giải từng cụm", expanded=True):
        for cluster_id in cluster_profile.index.tolist():
            st.markdown(f"##### Cụm {cluster_id}")
            _persona_card(int(cluster_id), personas[int(cluster_id)])

    st.markdown("#### Dự đoán một khách hàng mẫu")
    sample_index = st.selectbox(
        "Chọn một khách hàng từ dữ liệu đã lọc",
        cluster_frame.index.tolist(),
        format_func=lambda idx: f"Customer #{idx}",
    )
    sample_row = cluster_frame.loc[sample_index]
    manual_tab, sample_tab = st.tabs(["Nhập tay", "Dùng khách hàng mẫu"])

    with manual_tab:
        with st.form("manual_cluster_form"):
            cols = st.columns(3)
            input_values: dict[str, float] = {}
            for position, column_name in enumerate(NUMERICAL_FEATURES):
                target_col = cols[position % 3]
                with target_col:
                    input_values[column_name] = st.number_input(
                        column_name,
                        value=float(sample_row[column_name]),
                        step=1.0,
                        format="%.4f",
                    )
            submit_manual = st.form_submit_button("Dự đoán cụm")

        if submit_manual:
            input_frame = pd.DataFrame([input_values], columns=NUMERICAL_FEATURES)
            predicted_cluster = int(model.predict(scaler.transform(input_frame))[0])
            st.success(f"Khách hàng này thuộc cụm {predicted_cluster}")
            _persona_card(predicted_cluster, personas[predicted_cluster])

    with sample_tab:
        sample_payload = pd.DataFrame([sample_row[NUMERICAL_FEATURES].to_dict()])
        predicted_cluster = int(model.predict(scaler.transform(sample_payload))[0])
        st.info(f"Khách hàng mẫu được gán vào cụm {predicted_cluster}")
        _persona_card(predicted_cluster, personas[predicted_cluster])

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    _inject_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Olist Data Platform</div>
            <div class="hero-title">Dashboard tương tác cho ETL, Iceberg Cube và Clustering</div>
            <div class="hero-copy">
                Dữ liệu đi từ raw -> staging -> cleaned -> warehouse. Dashboard này dùng trực tiếp file cleaned của ETL
                cho phân tích khách hàng, và dùng warehouse để xây Iceberg cube, xem lineage và vận hành mô hình phân cụm.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        customer_frame = load_customer_dataset()
        artifacts = load_model_artifacts()
    except Exception as exc:
        st.error(f"Không tải được dữ liệu hoặc model: {exc}")
        st.stop()

    filtered_frame, filters = _apply_customer_filters(customer_frame)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Trạng thái tải")
    st.sidebar.success(f"Customers: {len(filtered_frame):,} / {len(customer_frame):,}")
    st.sidebar.caption(
        f"State: {len(filters['selected_states'])} | Payment: {len(filters['selected_payments'])} | Category: {len(filters['selected_categories'])}"
    )

    tab_overview, tab_etl, tab_cube, tab_cluster = st.tabs(
        ["Tổng quan", "ETL & Nguồn", "OLAP / Iceberg", "Clustering & Dự đoán"]
    )

    with tab_overview:
        _render_overview(filtered_frame)

    with tab_etl:
        _render_etl_tab()

    with tab_cube:
        _render_cube_tab()

    with tab_cluster:
        _render_cluster_tab(filtered_frame, artifacts)


if __name__ == "__main__":
    main()
