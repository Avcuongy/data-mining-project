import duckdb
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.join(BASE_DIR, "data_warehouse.duckdb")

if os.path.exists(database_path):
    os.remove(database_path)

conn = duckdb.connect(database=database_path)

sql_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_dw.sql"
)
with open(sql_file_path, "r", encoding="utf-8") as file:
    sql_script = file.read()

conn.execute(sql_script)

conn.close()
