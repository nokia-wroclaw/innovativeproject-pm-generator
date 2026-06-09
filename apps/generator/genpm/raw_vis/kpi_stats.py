"""KPI statistics helpers used by timeline plots (extracted for Spark jobs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def pull_kpi(df: DataFrame, kpi: str) -> pd.DataFrame:
    return (
        df.filter(F.col("kpi_id") == kpi)
        .groupBy("start_time")
        .agg(
            F.mean("kpi_value").alias("mean_value"),
            F.stddev("kpi_value").alias("std_value"),
            F.count("kpi_value").alias("n"),
        )
        .orderBy("start_time")
        .toPandas()
        .assign(start_time=lambda d: pd.to_datetime(d["start_time"]))
    )


def fit_distributions(values: np.ndarray) -> list[dict]:
    candidates = {
        "norm": stats.norm,
        "lognorm": stats.lognorm,
        "gamma": stats.gamma,
        "expon": stats.expon,
        "weibull_min": stats.weibull_min,
        "beta": stats.beta,
        "cauchy": stats.cauchy,
        "laplace": stats.laplace,
    }
    results = []
    for name, dist in candidates.items():
        try:
            params = dist.fit(values)
            d_stat, _p = stats.kstest(values, name, args=params)
            log_l = np.sum(dist.logpdf(values, *params))
            aic = 2 * len(params) - 2 * log_l
            results.append(dict(distribution=name, ks_stat=d_stat, aic=aic, params=params))
        except Exception:
            pass
    return sorted(results, key=lambda r: r["aic"])


def pettitt_test(series: np.ndarray) -> tuple[int, float]:
    n = len(series)
    u = np.zeros(n, dtype=float)
    for t in range(1, n):
        u[t] = u[t - 1] + np.sum(np.sign(series[t] - series[:t]))
    k = int(np.argmax(np.abs(u)))
    t_max = np.max(np.abs(u))
    p = 2.0 * np.exp(-6.0 * t_max**2 / (n**3 + n**2))
    return k, float(p)
