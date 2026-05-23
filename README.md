# Project setup

## Enviroment

- **Python Version:** Python >= 3.9

## Project setup

Run the following commands in your terminal:

```bash
git clone https://github.com/Avcuongy/data-mining-project.git

cd data-mining-project

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install -e .

python scripts/config.py   # make config
python scripts/etl.py      # make etl
```

# Duckdb

**Setup ODBC**: [ODBC API on Windows](https://duckdb.org/docs/current/clients/odbc/windows)

**Configure connection string in PowerBI:** `Driver=DuckDB Driver;Database=YOUR_ABSOLUTE_PATH\data-mining-project\data_warehouse.duckdb access_mode=read_only;`. Replace `YOUR_ABSOLUTE_PATH` with actual full local directory path.
