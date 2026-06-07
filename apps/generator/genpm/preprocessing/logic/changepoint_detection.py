import numpy as np
import pandas as pd
import ruptures as rpt
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
)


# Output schema mirrors input + regime_id
def build_regime_schema(pm_data: DataFrame) -> StructType:
    return StructType(pm_data.schema.fields + [StructField("regime_id", IntegerType(), False)])


# No decorator — plain function, (key, data) signature
def _assign_regimes(key: tuple, group_df: pd.DataFrame) -> pd.DataFrame:
    _MIN_SIZE = 168
    _MIN_LENGTH = 48

    df = group_df.sort_values("start_time").reset_index(drop=True)
    df["regime_id"] = 0

    vals = df["kpi_value"].values.astype(float)
    valid_mask = ~np.isnan(vals)

    if valid_mask.sum() < _MIN_LENGTH:
        return df

    filled = (
        pd.Series(vals, index=df["start_time"])  # <-- DatetimeIndex
        .interpolate(method="time", limit=6, limit_direction="both")
        .ffill(limit=2)
        .bfill(limit=2)
        .values  # back to ndarray
    )

    n = len(filled)
    pen = np.log(n) * np.var(filled) * 20

    try:
        algo = rpt.Pelt(model="rbf", min_size=_MIN_SIZE, jump=6).fit(filled.reshape(-1, 1))
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
    new_schema = build_regime_schema(pm_data)
    # .repartition(200, "kpi_id", "distname")
    return pm_data.groupby("kpi_id", "distname").applyInPandas(_assign_regimes, schema=new_schema)
