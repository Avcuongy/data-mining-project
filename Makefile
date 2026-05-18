# Declaration of phony targets
.PHONY: help etl extract transform load config hcubing

# Python interpreter (venv)
PYTHON ?= .venv/scripts/python.exe

# Rules
config:
	$(PYTHON) scripts/config.py

etl: extract transform load
	$(PYTHON) scripts/etl.py

extract:
	$(PYTHON) src/etl/extract.py

transform:
	$(PYTHON) src/etl/transform.py

load:
	$(PYTHON) src/etl/load.py

## Display available targets
help:
	@echo "Targets:"
	@echo "  help  - Display this help message"
	@echo "  etl   - Run the ETL process"
	@echo "  extract - Run the extract step"
	@echo "  transform - Run the transform step"
	@echo "  load - Run the load step"