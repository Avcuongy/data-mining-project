# Phony targets
.PHONY: help etl extract transform load hcubing

# Variables (can be overridden via command line)
DUCKDB_PATH ?= data_warehouse.duckdb
MIN_SUP_SALES ?= 0.0
K_SALES ?= None
MIN_SUP_LOGISTICS ?= 0.0
K_LOGISTICS ?= None

# Python interpreter (venv)
PYTHON ?= .venv/scripts/python.exe

# Rules
etl: extract transform load
	$(PYTHON) scripts/etl.py

extract:
	$(PYTHON) src/etl/extract.py

transform:
	$(PYTHON) src/etl/transform.py

load:
	$(PYTHON) src/etl/load.py

hcubing:
	$(PYTHON) src/utils/h_cubing.py --db $(DUCKDB_PATH) --min_sup_sales $(MIN_SUP_SALES) --k_sales $(K_SALES) --min_sup_logistics $(MIN_SUP_LOGISTICS) --k_logistics $(K_LOGISTICS)

## Display available targets
help:
	@echo "Targets:"
	@echo "  help       - Display this help message"
	@echo "  etl        - Run the ETL process"
	@echo "  extract    - Run the extract step"
	@echo "  transform  - Run the transform step"
	@echo "  load       - Run the load step"
	@echo "  hcubing    - Run iceberg cubing"