from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import types as T

from genpm.utils.utils import SparkDataManager

sdm = SparkDataManager()
spark = sdm.spark


# =============================================================================
# 1. SPARK SCHEMAS — NULLS, NOT NaNS
# =============================================================================

single_schema = T.StructType(
    [
        T.StructField("ts", T.TimestampType(), nullable=False),
        T.StructField("value", T.DoubleType(), nullable=True),
    ]
)

multi_schema = T.StructType(
    [
        T.StructField("ts", T.TimestampType(), nullable=False),
        T.StructField("kpi_a", T.DoubleType(), nullable=True),
        T.StructField("kpi_b", T.DoubleType(), nullable=True),
        T.StructField("kpi_c", T.DoubleType(), nullable=True),
    ]
)


def _is_missing_scalar(value) -> bool:
    """
    Returns True for None, NaN, NaT, and pandas.NA.

    Important: this check must be done before isinstance(value, datetime),
    because pd.NaT can sometimes behave like a datetime-like object.
    """
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_python_datetime(value):
    """
    Converts a value to datetime.datetime for Spark TimestampType.
    """
    if _is_missing_scalar(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()

    if isinstance(value, datetime):
        return value

    return pd.to_datetime(value).to_pydatetime()


def _to_spark_scalar(value):
    """
    Cleans scalar values before creating a Spark DataFrame:
    - None / NaN / NaT / pandas.NA -> None, which becomes Spark NULL,
    - numpy float/int -> Python float/int,
    - pandas Timestamp -> datetime.datetime.
    """
    if _is_missing_scalar(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.integer):
        return int(value)

    return value


def pdf_to_spark_with_schema(pdf: pd.DataFrame, schema: T.StructType) -> DataFrame:
    """
    Creates a Spark DataFrame from a pandas DataFrame using an explicit schema.

    Critical detail:
    Do not use iterrows(), because it can convert a missing value in a double
    column into NaT if the same row also contains a datetime column.
    """
    field_names = [field.name for field in schema.fields]

    missing_cols = [c for c in field_names if c not in pdf.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in pandas DataFrame: {missing_cols}")

    records = []

    n_rows = len(pdf)

    for i in range(n_rows):
        clean_row = {}

        for field in schema.fields:
            name = field.name

            # Important: read from a specific column, not from the whole row via iterrows().
            value = pdf[name].iloc[i]

            if isinstance(field.dataType, T.TimestampType):
                clean_row[name] = _to_python_datetime(value)
            else:
                clean_row[name] = _to_spark_scalar(value)

        records.append(clean_row)

    return spark.createDataFrame(records, schema=schema)


def nullable_float_list(values: np.ndarray, observed_mask: np.ndarray) -> list[float | None]:
    """
    Converts a float array into a list of float/None values.
    None will become Spark NULL.
    """
    return [
        float(v) if bool(is_observed) else None
        for v, is_observed in zip(values, observed_mask, strict=False)
    ]


def to_numeric_array_with_nan(values: pd.Series | list | np.ndarray) -> np.ndarray:
    """
    Converts None/NULL values into np.nan for numpy/pandas computations.

    This does not change the Spark representation.
    It is only used in the local computation layer.
    """
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


# =============================================================================
# 2. DEMO DATA GENERATION — GAPS AS NULLS
# =============================================================================


def make_series(
    n_days: int,
    period_hours: float,
    noise_std: float,
    gap_prob: float,
    gap_len_range: tuple[int, int] = (6, 48),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generates a noisy sinusoidal time series with intentional gaps.

    In pandas, gaps are stored as None so that they become Spark NULL
    after conversion to a Spark DataFrame.

    Sampling interval: 1 hour.
    """
    rng_ = np.random.default_rng(seed)

    n = n_days * 24
    t = np.arange(n)

    daily = np.sin(2 * np.pi * t / period_hours)
    weekly = 0.3 * np.sin(2 * np.pi * t / (24 * 7))

    base = 10 + 3 * daily + 1.5 * weekly
    values = base + rng_.normal(0, noise_std, size=n)

    observed_mask = np.ones(n, dtype=bool)

    i = 0
    while i < n:
        if rng_.random() < gap_prob:
            gap_len = int(rng_.integers(gap_len_range[0], gap_len_range[1]))
            observed_mask[i : i + gap_len] = False
            i += gap_len
        else:
            i += 24

    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")

    return pd.DataFrame(
        {
            "ts": timestamps,
            "value": nullable_float_list(values, observed_mask),
        }
    )


def make_multi(start_date, n_days: int, noise_std: float, seed: int) -> pd.DataFrame:
    """
    Generates three correlated KPI time series with missing values stored as None.

    None values will become Spark NULL after conversion.
    """
    rng_ = np.random.default_rng(seed)

    n = n_days * 24
    t = np.arange(n)

    common = np.sin(2 * np.pi * t / 24)
    week = np.sin(2 * np.pi * t / (24 * 7))

    kpi_a = 10 + 3 * common + 1.0 * week + rng_.normal(0, noise_std, n)
    kpi_b = 20 + 2 * common - 1.5 * week + rng_.normal(0, noise_std, n)
    kpi_c = 5 + 1.0 * common + 0.5 * kpi_a / 10 + rng_.normal(0, noise_std, n)

    masks = []

    for _ in range(3):
        observed_mask = np.ones(n, dtype=bool)
        missing_idx = rng_.choice(n, size=int(0.05 * n), replace=False)
        observed_mask[missing_idx] = False
        masks.append(observed_mask)

    timestamps = pd.date_range(start_date, periods=n, freq="h")

    return pd.DataFrame(
        {
            "ts": timestamps,
            "kpi_a": nullable_float_list(kpi_a, masks[0]),
            "kpi_b": nullable_float_list(kpi_b, masks[1]),
            "kpi_c": nullable_float_list(kpi_c, masks[2]),
        }
    )


real_pdf = make_series(
    n_days=90,
    period_hours=24,
    noise_std=0.3,
    gap_prob=0.05,
    seed=1,
)

synth_pdf = make_series(
    n_days=14,
    period_hours=24,
    noise_std=1.2,
    gap_prob=0.08,
    seed=2,
)
