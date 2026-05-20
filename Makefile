# Phony targets
.PHONY: help etl extract transform load config hcubing

# Variables
DUCKDB_PATH ?= data_warehouse.duckdb
MIN_SUP_SALES ?= 0.2
K_SALES ?= 500
MIN_SUP_LOGISTICS ?= 0.1
K_LOGISTICS ?= 500

# Python interpreter (venv)
PYTHON ?= .venv/scripts/python.exe

# Rules
config:
	$(PYTHON) scripts/config.py

etl:
	$(PYTHON) scripts/etl.py

extract:
	$(PYTHON) src/etl/extract.py

transform:
	$(PYTHON) src/etl/transform.py

load:
	$(PYTHON) src/etl/load.py

hcubing:
	$(PYTHON) src/utils/hcubing.py --db $(DUCKDB_PATH) --min_sup_sales $(MIN_SUP_SALES) --k_sales $(K_SALES) --min_sup_logistics $(MIN_SUP_LOGISTICS) --k_logistics $(K_LOGISTICS)

## Display available targets
help:
	@echo "Targets:"
	@echo "  help       - Display this help message"
	@echo "  config     - Set up configuration"
	@echo "  etl        - Run the ETL process"
	@echo "  extract    - Run the extract step"
	@echo "  transform  - Run the transform step"
	@echo "  load       - Run the load step"
	@echo "  hcubing    - Run iceberg cubing"