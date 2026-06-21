"""PySpark unit tests for simple_logic and SimpleMinMaxScaler."""

from datetime import datetime

import pytest
from pyspark.sql import Row
from pyspark.sql import functions as f

from genpm.preprocessing.logic.scaling import SimpleMinMaxScaler
from genpm.preprocessing.logic.simple_logic import raw_pm_preperation, simple_reports_pivot

# ---------------------------------------------------------------------------
# raw_pm_preperation
# ---------------------------------------------------------------------------


def test_raw_pm_preperation_drops_duplicates(spark):
    rows = [
        Row(
            start_time=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            kpi_id="k1",
            kpi_value=1.0,
        ),
        Row(
            start_time=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            kpi_id="k1",
            kpi_value=1.0,
        ),
        Row(
            start_time=datetime(2024, 1, 1, 1),
            bts_id="b1",
            distname="d1",
            kpi_id="k1",
            kpi_value=2.0,
        ),
    ]
    df = spark.createDataFrame(rows)
    result = raw_pm_preperation(df)
    assert result.count() == 2


def test_raw_pm_preperation_drops_null_start_time(spark):
    rows = [
        Row(start_time=None, bts_id="b1", distname="d1", kpi_id="k1", kpi_value=1.0),
        Row(
            start_time=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            kpi_id="k1",
            kpi_value=2.0,
        ),
    ]
    df = spark.createDataFrame(rows)
    result = raw_pm_preperation(df)
    assert result.count() == 1


def test_raw_pm_preperation_drops_null_distname(spark):
    rows = [
        Row(
            start_time=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname=None,
            kpi_id="k1",
            kpi_value=1.0,
        ),
        Row(
            start_time=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            kpi_id="k1",
            kpi_value=2.0,
        ),
    ]
    df = spark.createDataFrame(rows)
    result = raw_pm_preperation(df)
    assert result.count() == 1


def test_raw_pm_preperation_keeps_valid_rows(spark):
    rows = [
        Row(
            start_time=datetime(2024, 1, 1, t),
            bts_id="b1",
            distname="d1",
            kpi_id="k1",
            kpi_value=float(t),
        )
        for t in range(5)
    ]
    df = spark.createDataFrame(rows)
    result = raw_pm_preperation(df)
    assert result.count() == 5


# ---------------------------------------------------------------------------
# simple_reports_pivot
# ---------------------------------------------------------------------------


def test_simple_reports_pivot_pivots_report_names(spark):
    rows = [
        Row(
            datetime=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            report_name="cfg_a",
            report_result="val_a",
        ),
        Row(
            datetime=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            report_name="cfg_b",
            report_result="val_b",
        ),
    ]
    df = spark.createDataFrame(rows)
    result = simple_reports_pivot(df)
    assert "cfg_a" in result.columns
    assert "cfg_b" in result.columns


def test_simple_reports_pivot_one_row_per_datetime_distname(spark):
    rows = [
        Row(
            datetime=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            report_name="cfg_a",
            report_result="v1",
        ),
        Row(
            datetime=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            report_name="cfg_b",
            report_result="v2",
        ),
        Row(
            datetime=datetime(2024, 1, 1, 1),
            bts_id="b1",
            distname="d1",
            report_name="cfg_a",
            report_result="v3",
        ),
    ]
    df = spark.createDataFrame(rows)
    result = simple_reports_pivot(df)
    assert result.count() == 2


def test_simple_reports_pivot_correct_value(spark):
    rows = [
        Row(
            datetime=datetime(2024, 1, 1, 0),
            bts_id="b1",
            distname="d1",
            report_name="band",
            report_result="NR",
        ),
    ]
    df = spark.createDataFrame(rows)
    result = simple_reports_pivot(df)
    row = result.collect()[0]
    assert row["band"] == "NR"


# ---------------------------------------------------------------------------
# SimpleMinMaxScaler
# ---------------------------------------------------------------------------


def _pm_df(spark, values: list[float], kpi_id: str = "k1"):
    rows = [Row(kpi_id=kpi_id, bts_id="b1", kpi_value=v) for v in values]
    return spark.createDataFrame(rows)


def test_simple_minmax_scaler_fit_returns_min_max(spark):
    df = _pm_df(spark, [0.0, 5.0, 10.0])
    scaler = SimpleMinMaxScaler(value_col="kpi_value", group_cols=["kpi_id"])
    params = scaler.fit(df)
    row = params.filter(f.col("kpi_id") == "k1").collect()[0]
    assert row["mm_min"] == pytest.approx(0.0)
    assert row["mm_max"] == pytest.approx(10.0)


def test_simple_minmax_scaler_transform_range_zero_to_one(spark):
    df = _pm_df(spark, [0.0, 5.0, 10.0])
    scaler = SimpleMinMaxScaler(value_col="kpi_value", group_cols=["kpi_id"])
    scaler.fit(df)
    result = scaler.transform(df)
    values = [r["kpi_value"] for r in result.collect()]
    assert min(values) == pytest.approx(0.0, abs=1e-6)
    assert max(values) == pytest.approx(1.0, abs=1e-5)


def test_simple_minmax_scaler_inverse_transform_restores_originals(spark):
    originals = [0.0, 2.5, 10.0]
    df = _pm_df(spark, originals)
    scaler = SimpleMinMaxScaler(value_col="kpi_value", group_cols=["kpi_id"])
    scaler.fit(df)
    scaled = scaler.transform(df)
    restored = scaler.inverse_transform(scaled)
    values = sorted(r["kpi_value"] for r in restored.collect())
    for orig, val in zip(sorted(originals), values, strict=False):
        assert val == pytest.approx(orig, abs=1e-4)


def test_simple_minmax_scaler_constant_series_no_error(spark):
    df = _pm_df(spark, [5.0, 5.0, 5.0])
    scaler = SimpleMinMaxScaler(value_col="kpi_value", group_cols=["kpi_id"])
    scaler.fit(df)
    result = scaler.transform(df)
    # When min == max the denominator is ε, so scaled values are near zero
    assert result.count() == 3


def test_simple_minmax_scaler_multiple_kpis_scaled_independently(spark):
    rows = [Row(kpi_id="k1", bts_id="b1", kpi_value=v) for v in [0.0, 100.0]] + [
        Row(kpi_id="k2", bts_id="b1", kpi_value=v) for v in [0.0, 1.0]
    ]
    df = spark.createDataFrame(rows)
    scaler = SimpleMinMaxScaler(value_col="kpi_value", group_cols=["kpi_id"])
    scaler.fit(df)
    result = scaler.transform(df)
    # Both KPIs should have a max scaled value of 1.0
    max_per_kpi = {
        r["kpi_id"]: r["max_val"]
        for r in result.groupBy("kpi_id").agg(f.max("kpi_value").alias("max_val")).collect()
    }
    assert max_per_kpi["k1"] == pytest.approx(1.0, abs=1e-5)
    assert max_per_kpi["k2"] == pytest.approx(1.0, abs=1e-5)
