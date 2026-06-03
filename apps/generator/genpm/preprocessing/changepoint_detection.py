import numpy as np
import pandas as pd
import ruptures as rpt
from pyspark.sql import DataFrame
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


# No decorator — plain function, (key, data) signature
def _assign_regimes(key: tuple, group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applied per (kpi_id, bts_id, distname) group.
    key is unpacked for clarity but not needed inside since
    the columns are already present in group_df.
    """
    _MIN_SIZE = 24
    _MIN_LENGTH = 48

    df = group_df.sort_values("start_time").reset_index(drop=True)
    df["regime_id"] = 0

    vals = df["kpi_value"].values.astype(float)
    valid_mask = ~np.isnan(vals)

    if valid_mask.sum() < _MIN_LENGTH:
        return df

    filled = (
        pd.Series(vals)
        .interpolate(method="time", limit=6, limit_direction="both")
        .ffill(limit=2)
        .bfill(limit=2)
        .values
    )

    n = len(filled)
    pen = np.log(n)

    try:
        algo = rpt.Pelt(model="l2", min_size=_MIN_SIZE, jump=1).fit(filled.reshape(-1, 1))
        breakpoints = algo.predict(pen=pen)
    except rpt.exceptions.BadSegmentationParameters:
        return df
    except Exception:
        return df

    regime_id = np.zeros(n, dtype=int)
    for regime, bp in enumerate(breakpoints[:-1], start=1):
        regime_id[bp:] = regime

    df["regime_id"] = regime_id
    return df


def add_regime_ids(pm_data: DataFrame) -> DataFrame:
    return pm_data.groupby("kpi_id", "bts_id", "distname").applyInPandas(
        _assign_regimes, schema=_REGIME_SCHEMA
    )
