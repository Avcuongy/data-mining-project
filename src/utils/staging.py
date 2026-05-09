import re

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy.sql import sqltypes


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        column.strip() if isinstance(column, str) else column
        for column in frame.columns
    ]
    return frame


def _coerce_strings(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()


def _coerce_integers(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "Int64"
            )


def _coerce_floats(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                "Float64"
            )


def _coerce_datetimes(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)


def _stringify_type(column_type: object) -> str:
    return column_type.__class__.__name__.lower()


def _type_group(column_type: object) -> str:
    if isinstance(column_type, (sqltypes.DateTime, sqltypes.TIMESTAMP, sqltypes.Date)):
        return "datetime"
    if isinstance(
        column_type,
        (sqltypes.Integer, sqltypes.SmallInteger, sqltypes.BigInteger),
    ):
        return "integer"
    if isinstance(
        column_type,
        (sqltypes.Numeric, sqltypes.Float, sqltypes.DECIMAL, sqltypes.REAL),
    ):
        return "float"
    if isinstance(column_type, (sqltypes.Boolean,)):
        return "boolean"
    return "string"


def _load_table_metadata(engine, table_name: str) -> dict[str, object]:
    """Load metadata from database

    Args:
        engine: SQLAlchemy engine to connect to the database.
        table_name (str): Name of the table to load metadata.

    Returns:
        dict[str, object]:
    """
    columns = inspect(engine).get_columns(table_name)
    normalized_columns = []
    for column in columns:
        normalized_columns.append(
            {
                "name": column["name"],
                "nullable": column.get("nullable", True),
                "type_group": _type_group(column["type"]),
                "type_name": _stringify_type(column["type"]),
            }
        )

    return {
        "columns": normalized_columns,
        "required": [
            column["name"] for column in normalized_columns if not column["nullable"]
        ],
    }


def staging(
    df: pd.DataFrame,
    table_name: str | None = None,
    engine=None,
) -> pd.DataFrame:
    """Perform staging transformations on the given DataFrame based on the table metadata.

    Args:
        df (pd.DataFrame): Input DataFrame to be staged.
        table_name (str | None, optional): Name of the table to load metadata. Defaults to None.
        engine: SQLAlchemy engine to connect to the database. Defaults to None.

    Returns:
        pd.DataFrame:
    """
    if df is None:
        raise ValueError("staging() expects a pandas DataFrame.")

    frame = _normalize_columns(df)
    if not table_name:
        raise ValueError("staging() requires a table_name to read database metadata.")
    if engine is None:
        raise ValueError(
            "staging() requires a SQLAlchemy engine for metadata inspection."
        )

    schema = _load_table_metadata(engine, table_name)
    required_columns = schema.get("required", [])
    missing_required = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_required:
        raise ValueError(
            f"Missing required columns for {table_name or 'unknown table'}: {missing_required}"
        )

    expected_columns = {column["name"] for column in schema.get("columns", [])}
    unexpected_columns = [
        column for column in frame.columns if column not in expected_columns
    ]
    if unexpected_columns:
        print(
            f"[{table_name}] Keeping unexpected columns during staging: {unexpected_columns}"
        )

    original_rows = len(frame)

    datetime_columns: list[str] = []
    integer_columns: list[str] = []
    float_columns: list[str] = []
    string_columns: list[str] = []

    for column in schema.get("columns", []):
        column_name = column["name"]
        type_group = column["type_group"]
        if type_group == "datetime":
            datetime_columns.append(column_name)
        elif type_group == "integer":
            integer_columns.append(column_name)
        elif type_group == "float":
            float_columns.append(column_name)
        else:
            string_columns.append(column_name)

    _coerce_strings(frame, string_columns)
    _coerce_integers(frame, integer_columns)
    _coerce_floats(frame, float_columns)
    _coerce_datetimes(frame, datetime_columns)

    inferred_datetime_columns = [
        column
        for column in frame.columns
        if column not in datetime_columns
        and re.search(r"(date|time|timestamp)", str(column), re.IGNORECASE)
    ]
    _coerce_datetimes(frame, inferred_datetime_columns)

    for column in string_columns:
        if column in frame.columns:
            frame[column] = frame[column].replace("", pd.NA)

    if required_columns:
        frame = frame.dropna()  # Option

    frame = frame.drop_duplicates().reset_index(drop=True)

    removed_rows = original_rows - len(frame)
    print(f"Staging complete: {len(frame):,} rows kept, {removed_rows:,} rows removed")

    return frame
