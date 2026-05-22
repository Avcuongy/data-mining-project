from pathlib import Path
import runpy
import logging

import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ETL_STAGING = PROJECT_ROOT / "data" / "etl" / "staging"
DATA_ETL_COMPLETED = PROJECT_ROOT / "data" / "etl" / "completed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"
TEST_DIR = PROJECT_ROOT / "test"
ENV_FILE = PROJECT_ROOT / ".env"


def _ensure_directories() -> None:
    for path in (DATA_ETL_STAGING, DATA_ETL_COMPLETED, DATA_RAW, LOGS_DIR, TEST_DIR):
        path.mkdir(parents=True, exist_ok=True)
        logging.info("Configuration: Initializing folder at %s", path)


def _create_data_warehouse() -> None:
    config_script = PROJECT_ROOT / "config" / "config_dw.py"
    if not config_script.exists():
        raise FileNotFoundError(f"Config script not found: {config_script}")
    logging.info("Configuration: Initializing data warehouse at %s", config_script)
    runpy.run_path(str(config_script), run_name="__main__")


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        filemode="a",
        filename=LOGS_DIR / "config.log",
    )

    logging.info("Configuration: Starting")
    _ensure_directories()
    _create_data_warehouse()
    logging.info("Configuration: Complete")


if __name__ == "__main__":
    main()
