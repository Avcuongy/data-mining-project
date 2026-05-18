from pathlib import Path
import argparse
import logging
import runpy
import sys

import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "data" / "etl" / "staging"
COMPLETED_DIR = PROJECT_ROOT / "data" / "etl" / "completed"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _run_script(script_path: Path) -> None:
    runpy.run_path(str(script_path), run_name="__main__")


def run_pipeline(
    skip_extract: bool = False, skip_transform: bool = False, skip_load: bool = False
) -> None:
    logging.info("ETL pipeline")

    try:
        if not skip_extract:
            logging.info("Step 1/3: Extracting data to %s", STAGING_DIR)
            _run_script(PROJECT_ROOT / "src" / "etl" / "extract.py")
        else:
            logging.info("Skipping extract step")

        if not skip_transform:
            logging.info(
                "Step 2/3: Transforming data from %s to %s", STAGING_DIR, COMPLETED_DIR
            )
            _run_script(PROJECT_ROOT / "src" / "etl" / "transform.py")
        else:
            logging.info("Skipping transform step")

        if not skip_load:
            logging.info("Step 3/3: Loading data from %s into DuckDB", COMPLETED_DIR)
            _run_script(PROJECT_ROOT / "src" / "etl" / "load.py")
        else:
            logging.info("Skipping load step")

        logging.info("ETL pipeline completed")

    except Exception:
        logging.exception("ETL pipeline failed")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ETL pipeline: extract -> transform -> load"
    )
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract step")
    parser.add_argument(
        "--skip-transform", action="store_true", help="Skip transform step"
    )
    parser.add_argument("--skip-load", action="store_true", help="Skip load step")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        filemode="a",
        filename=PROJECT_ROOT / "logs" / "etl.log",
    )

    run_pipeline(
        skip_extract=args.skip_extract,
        skip_transform=args.skip_transform,
        skip_load=args.skip_load,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
