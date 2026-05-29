# Project setup

## Enviroment

**Python Version:** Python >= 3.9

## Project setup

Run the following commands in your terminal:

```bash
git clone https://github.com/Avcuongy/data-mining-project.git

cd data-mining-project

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

pip install -e .

# Run all by `make all`
python scripts/config.py
python scripts/etl.py
python src/utils/hcubing.py --db data_warehouse.duckdb --min_sup_sales 0.2 --k_sales 500 --min_sup_logistics 0.1 --k_logistics 500
```

# ELT

## Data source

![ERD](/assets/erd.png)

## ETL

![ETL](/assets/etl.png)
