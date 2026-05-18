from pathlib import Path
import sys
import argparse
import logging

from src.etl.extract import extract as extract_data
from src.etl.transform import transform as transform_data
from src.etl.load import load as load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / "data" / "etl" / "staging"
COMPLETED_DIR = PROJECT_ROOT / "data" / "etl" / "completed"


def run_pipeline(
    skip_extract: bool = False, skip_transform: bool = False, skip_load: bool = False
) -> None:
    logging.info("ETL pipeline")

    try:
        if not skip_extract:
            logging.info("Step 1/3: Extracting data to %s", STAGING_DIR)
            exported = extract_data(output_dir=str(STAGING_DIR))
            logging.info(
                "Extract exported tables: %s",
                list(exported.keys()) if exported else "none",
            )
        else:
            logging.info("Skipping extract step")

        if not skip_transform:
            logging.info(
                "Step 2/3: Transforming data from %s to %s", STAGING_DIR, COMPLETED_DIR
            )
            transformed = transform_data(
                input_dir=STAGING_DIR, output_dir=COMPLETED_DIR
            )
            logging.info(
                "Transform produced tables: %s",
                list(transformed.keys()) if transformed else "none",
            )
        else:
            logging.info("Skipping transform step")

        if not skip_load:
            logging.info("Step 3/3: Loading data from %s into DuckDB", COMPLETED_DIR)
            load_data()
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
