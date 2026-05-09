import duckdb
import os

# Đường dẫn tới file DuckDB
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
database_path = os.path.join(BASE_DIR, "data_warehouse.duckdb")

# Xóa file cơ sở dữ liệu nếu tồn tại
if os.path.exists(database_path):
    os.remove(database_path)

# Kết nối đến DuckDB, tạo hoặc mở cơ sở dữ liệu
conn = duckdb.connect(database=database_path)

# Đọc nội dung của file SQL (tương đối theo vị trí script này)
sql_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_dw.sql"
)
with open(sql_file_path, "r", encoding="utf-8") as file:
    sql_script = file.read()

# Chạy các câu lệnh SQL từ file
conn.execute(sql_script)

# Đóng kết nối
conn.close()

print(f"Database has been created and saved to {database_path}")
