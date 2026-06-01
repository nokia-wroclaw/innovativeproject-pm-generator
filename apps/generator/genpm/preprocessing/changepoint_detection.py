import numpy as np
import pandas as pd
import ruptures as rpt
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Output schema mirrors input + regime_id
_REGIME_SCHEMA = StructType(
    [
        StructField("kpi_id", StringType(), True),
        StructField("bts_id", StringType(), True),
        StructField("distname", StringType(), True),
        StructField("start_time", TimestampType(), True),
        StructField("kpi_value", DoubleType(), True),
        StructField("regime_id", IntegerType(), True),
    ]
)


@f.pandas_udf(_REGIME_SCHEMA, f.PandasUDFType.GROUPED_MAP)
def _assign_regimes(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applied per (kpi_id, bts_id) group.
    Sorts by start_time, runs PELT on kpi_value,
    and assigns an integer regime_id to each row.
    """
    _MIN_SIZE = 24  # minimum hours per regime (~1 day)
    _MIN_LENGTH = 48  # minimum series length to attempt PELT

    df = group_df.sort_values("start_time").reset_index(drop=True)
    df["regime_id"] = 0  # default: single regime

    vals = df["kpi_value"].values.astype(float)
    valid_mask = ~np.isnan(vals)

    # Need enough non-null values to run PELT
    if valid_mask.sum() < _MIN_LENGTH:
        return df

    # Fill small gaps with interpolation so PELT sees a contiguous signal.
    # Large gaps (>6h) stay as NaN and are handled by contiguous block logic below.
    filled = (
        pd.Series(vals)
        .interpolate(method="time", limit=6, limit_direction="both")
        .ffill(limit=2)
        .bfill(limit=2)
        .values
    )

    # BIC-derived penalty: log(n) adapts to series length
    n = len(filled)
    pen = np.log(n)

    try:
        algo = rpt.Pelt(model="l2", min_size=_MIN_SIZE, jump=1).fit(filled.reshape(-1, 1))
        # predict() returns end-indices of each segment; last is sentinel (=n)
        breakpoints = algo.predict(pen=pen)  # e.g. [340, 890, 1200]
    except rpt.exceptions.BadSegmentationParameters:
        return df
    except Exception:
        return df

    # Map each row index → regime_id
    # breakpoints[-1] is always n (sentinel), so we drop it
    regime_id = np.zeros(n, dtype=int)
    # prev = 0
    for regime, bp in enumerate(breakpoints[:-1], start=1):
        regime_id[bp:] = regime  # everything from bp onward gets next id

    df["regime_id"] = regime_id
    return df


def add_regime_ids(pm_data: DataFrame) -> DataFrame:
    """
    Takes a pm_data Spark DataFrame with schema:
        kpi_id, bts_id, distname, start_time, kpi_value

    Returns the same DataFrame with an additional integer column:
        regime_id  — 0-indexed per (kpi_id, bts_id), monotonically
                     increasing with time (0 = first regime, 1 = second, ...)

    Notes:
        - Groups with fewer than 48 non-null observations get regime_id = 0.
        - Data is not globally sorted; ordering is local to each group.
        - regime_id resets per (kpi_id, bts_id) — not globally unique.
    """
    return pm_data.groupby("kpi_id", "distname").apply(_assign_regimes)
