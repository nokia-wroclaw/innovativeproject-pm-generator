import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from genpm.utils.utils import when_chained


def make_pattern(words):
    """Build a word-boundary regex that matches any word in the list, case-insensitive."""
    return r"(?i)(^|[^a-z0-9])(" + "|".join(re.escape(w) for w in words) + r")([^a-z0-9]|$)"


def classify_kpis(
    df: DataFrame,
    avg_keywords: list[str],
    max_keywords: list[str],
    min_keywords: list[str],
    mean_like_keywords: list[str],
    ratio_keywords: list[str],
    volume_keywords: list[str],
    mean_like_units: list[str],
    volume_units: list[str],
) -> DataFrame:
    """Classify each KPI's aggregation character (avg/sum/max/min) from name, unit, and value range."""
    ratio_pattern = make_pattern(ratio_keywords)
    mean_like_pattern = make_pattern(mean_like_keywords)
    volume_pattern = make_pattern(volume_keywords)
    avg_pattern = make_pattern(avg_keywords)
    max_pattern = make_pattern(max_keywords)
    min_pattern = make_pattern(min_keywords)

    kpi_classified = (
        df.withColumn(
            "stat_keyword_match",
            when_chained(
                [
                    (f.col("kpi_name").rlike(avg_pattern), "avg"),
                    (f.col("kpi_name").rlike(max_pattern), "max"),
                    (f.col("kpi_name").rlike(min_pattern), "min"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "unit_match",
            when_chained(
                [
                    (f.col("unit").isin(*mean_like_units, "mean_like"), "mean_like"),
                    (f.col("unit").isin(*volume_units, "volume"), "volume"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "keyword_match",
            when_chained(
                [
                    (f.col("kpi_name").rlike(ratio_pattern), "ratio"),
                    (f.col("kpi_name").rlike(mean_like_pattern), "mean_like"),
                    (f.col("kpi_name").rlike(volume_pattern), "volume"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "kpi_character",
            when_chained(
                [
                    (f.col("stat_keyword_match") != "unknown", f.col("stat_keyword_match")),
                    (f.col("unit_match") != "unknown", f.col("unit_match")),
                    (f.col("keyword_match") != "unknown", f.col("keyword_match")),
                    (f.col("kpi_min") > 0, "mean_like"),
                    ((f.col("kpi_min") >= 0) & (f.col("kpi_max") <= 100), "ratio"),
                ],
                otherwise="unknown",
            ),
        )
        .withColumn(
            "classification_source",
            when_chained(
                [
                    (f.col("stat_keyword_match") != "unknown", "stat_keyword"),
                    (f.col("unit_match").isin("mean_like", "volume"), "unit"),
                    (f.col("keyword_match") != "unknown", "keyword"),
                ],
                otherwise="value_range_fallback",
            ),
        )
        .withColumn(
            "agg_method",
            when_chained(
                [
                    (
                        f.col("kpi_character").isin("mean_like", "ratio", "avg"),
                        "avg",
                    ),
                    (f.col("kpi_character") == "max", "max"),
                    (f.col("kpi_character") == "min", "min"),
                ],
                otherwise="sum",
            ),
        )
        .drop("keyword_match")
    )
    return kpi_classified
