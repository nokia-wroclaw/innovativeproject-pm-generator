from collections.abc import Callable

import holidays
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import DateType

# Registry of all available date features
DATE_FEATURE_REGISTRY: dict[str, Callable] = {
    "year": lambda col: f.year(col),
    "month": lambda col: f.month(col),
    "day_of_week": lambda col: f.dayofweek(col),
    "day_of_year": lambda col: f.dayofyear(col),
    "week_of_year": lambda col: f.weekofyear(col),
    "quarter": lambda col: f.quarter(col),
    "hour": lambda col: f.hour(col),
    "is_weekend": lambda col: (f.dayofweek(col).isin([1, 7])).cast("int"),
    "is_month_end": lambda col: (f.dayofmonth(col) == f.last_day(col)).cast("int"),
    "is_month_start": lambda col: (f.dayofmonth(col) == 1).cast("int"),
}


def add_date_features(
    df: DataFrame,
    time_col: str,
    features: list[str] | None = None,
) -> DataFrame:
    """
    Adds date-based features to a Spark DataFrame.

    Args:
        df:       Input Spark DataFrame.
        time_col: Name of the timestamp/date column to extract features from.
        features: List of feature names to add. If None, all available features
                  are added. Available features: year, month, day_of_week,
                  day_of_year, week_of_year, quarter, hour, is_weekend,
                  is_month_end, is_month_start.

    Returns:
        DataFrame with new date feature columns appended.

    Raises:
        ValueError: If an unknown feature name is requested.

    Example:
        >>> df = add_date_features(df, "timestamp", features=["year", "month", "is_weekend"])
    """
    # Fall back to all registered features when the caller provides no list.
    requested = features if features is not None else list(DATE_FEATURE_REGISTRY.keys())

    # Fail fast with a descriptive message before any DataFrame work.
    unknown = set(requested) - DATE_FEATURE_REGISTRY.keys()
    if unknown:
        raise ValueError(
            f"Unknown features: {unknown}. " f"Available: {set(DATE_FEATURE_REGISTRY.keys())}"
        )

    # Append each feature column via its registered transformation lambda.
    for feature in requested:
        df = df.withColumn(feature, DATE_FEATURE_REGISTRY[feature](time_col))

    return df


HOLIDAY_FEATURE_REGISTRY = {
    "is_holiday",
    "holiday_name",
    "is_holiday_eve",
    "days_to_next_holiday",
    "days_since_last_holiday",
    "is_long_weekend",
}


def add_holiday_features(
    df: DataFrame,
    time_col: str,
    spark: SparkSession,
    country_code: str = "US",
    features: list[str] | None = None,
) -> DataFrame:
    """Enriches df with public-holiday-based features.

    Uses the ``holidays`` library to derive the holiday calendar for the date
    range present in ``df``, then computes each requested feature on distinct
    dates before joining the results back to the full DataFrame- for better peformance.

    Args:
        df: Input Spark DataFrame containing at least one date/timestamp column.
        time_col: Name of the timestamp
        spark: Active ``SparkSession`` used to create intermediate DataFrames.
        country_code: ISO 3166-1 alpha-2 country code. Defaults to ``"US"``.
        features: List of feature names to compute. If ``None``, all features
            in ``HOLIDAY_FEATURE_REGISTRY`` are computed. Supported values:
            ``is_holiday``, ``holiday_name``, ``is_holiday_eve``,
            ``days_to_next_holiday``, ``days_since_last_holiday``,
            ``is_long_weekend``.

    Returns:
        A new DataFrame with the requested holiday feature columns appended.
        Each row is matched by its calendar date (time-of-day is ignored).

    Raises:
        ValueError: If any element of ``features`` is not present in
            ``HOLIDAY_FEATURE_REGISTRY``.

    Example:
        >>> df = add_holiday_features(df, "event_date", spark, country_code="PL",
        ...                           features=["is_holiday", "days_to_next_holiday"])
    """
    # Default to computing every registered holiday feature
    requested = set(features) if features is not None else HOLIDAY_FEATURE_REGISTRY

    # Validate all requested feature names
    unknown = requested - HOLIDAY_FEATURE_REGISTRY
    if unknown:
        raise ValueError(f"Unknown features: {unknown}. Available: {HOLIDAY_FEATURE_REGISTRY}")

    # Determine the year span covered by the dataset
    min_year = df.agg(f.min(time_col)).collect()[0][0].year
    max_year = df.agg(f.max(time_col)).collect()[0][0].year
    years = list(range(min_year, max_year + 1))

    hols = holidays.country_holidays(country_code, years=years)
    holiday_pdf = pd.DataFrame(
        [{"date": str(d), "holiday_name": name, "is_holiday": 1} for d, name in hols.items()]
    )

    holiday_sdf = spark.createDataFrame(holiday_pdf).withColumn(
        "date", f.col("date").cast(DateType())
    )

    # Keep a sorted list of holiday date strings for efficient isin() checks
    holiday_dates = sorted([str(d) for d in hols.keys()])

    # work on distinct dates only
    df = df.withColumn("date_only", f.to_date(f.col(time_col)))
    date_df = df.select("date_only").distinct()

    # Left-join so non-holiday dates are preserved; fill nulls with sensible
    # defaults so downstream feature logic always sees 0 / "None".
    date_df = (
        date_df.join(holiday_sdf, date_df.date_only == holiday_sdf.date, how="left")
        .drop("date")
        .fillna({"is_holiday": 0, "holiday_name": "None"})
    )

    # A date is a "holiday eve" if the *next* calendar day is a public holiday.
    if "is_holiday_eve" in requested:
        date_df = date_df.withColumn(
            "is_holiday_eve", f.date_add(f.col("date_only"), 1).isin(holiday_dates).cast("int")
        )

    if requested & {"days_to_next_holiday", "days_since_last_holiday"}:
        holiday_df = spark.createDataFrame(
            [(d,) for d in holiday_dates], ["holiday_date"]
        ).withColumn("holiday_date", f.col("holiday_date").cast(DateType()))

        # Partition window lets us aggregate across all holiday distances
        # independently for each distinct date.
        window = Window.partitionBy("date_only")

        # Cross-join produces one row per (date, holiday) pair so datediff can
        # be computed; rows are later collapsed via windowed aggregation.
        date_df = date_df.join(holiday_df, how="cross").withColumn(
            "days_diff", f.datediff("holiday_date", "date_only")
        )

        if "days_to_next_holiday" in requested:
            # Take the smallest non-negative diff → nearest future holiday
            date_df = date_df.withColumn(
                "days_to_next_holiday",
                f.min(f.when(f.col("days_diff") >= 0, f.col("days_diff"))).over(window),
            )

        if "days_since_last_holiday" in requested:
            # Take the smallest non-positive diff (negated) → nearest past holiday.
            date_df = date_df.withColumn(
                "days_since_last_holiday",
                f.min(f.when(f.col("days_diff") <= 0, -f.col("days_diff"))).over(window),
            )

        # Remove the exploded holiday rows; keep one result row per distinct date
        date_df = date_df.drop("holiday_date", "days_diff").dropDuplicates()

    # A "long weekend" occurs when a public holiday falls on a Friday (6) or
    # Monday (2), effectively extending the standard two-day weekend.
    if "is_long_weekend" in requested:
        date_df = date_df.withColumn(
            "is_long_weekend",
            ((f.col("is_holiday") == 1) & f.dayofweek(f.col("date_only")).isin([2, 6])).cast("int"),
        )

    # single join back to full df
    df = df.join(date_df, on="date_only", how="left")

    return df
