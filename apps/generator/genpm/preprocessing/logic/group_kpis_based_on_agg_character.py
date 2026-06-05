import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

mean_like_units = ["%", "bit/s", "kbit/s", "Mbit/s", "ms", "#/s", "#/h"]
volume_units = ["#"]

min_keywords = ["min", "minimal", "minimum"]

max_keywords = ["max", "maximal", "maximum"]

avg_keywords = ["avg", "average"]

# RATIOS: telecom KPI acronyms / categories that are usually percentages
ratio_keywords = [
    "cssr",  # Call Setup Success Rate
    "hosr",  # Handover Success Rate
    "asr",  # Answer Seizure Ratio
    "ccr",  # Call Completion Ratio / related completion KPIs
    "dcr",  # Drop Call Ratio
    "bler",  # Block Error Rate
    "fer",  # Frame Error Rate
    "per",  # Packet Error Rate
    "availability",
    "accessibility",
    "retainability",
    "integrity",
    "utilization",
    "Average Time",
    "Average Duration",
]

# MEAN-LIKE: speed / quality / radio level / delay measurements
mean_like_keywords = [
    "throughput",
    "latency",
    "jitter",
    "rtt",
    "rssi",
    "rsrp",
    "rsrq",
    "sinr",
    "snr",
    "mos",
]

# VOLUME: additive traffic / count style telecom nouns
volume_keywords = [
    "erlang",
    "mou",
    "bytes",
    "octets",
    "attempts",
    "packets",
    "Total Time",
    "Total Duration",
    "volume",
]


def make_pattern(words):
    return r"(?i)(^|[^a-z0-9])(" + "|".join(re.escape(w) for w in words) + r")([^a-z0-9]|$)"


# ratio_pattern   = "(?i)(" + "|".join(ratio_keywords) + ")"

ratio_pattern = make_pattern(ratio_keywords)
mean_like_pattern = make_pattern(mean_like_keywords)
volume_pattern = make_pattern(volume_keywords)
avg_pattern = make_pattern(avg_keywords)
max_pattern = make_pattern(max_keywords)
min_pattern = make_pattern(min_keywords)


def classify_kpis(
    df: DataFrame,
) -> DataFrame:
    kpi_classified = (
        df.withColumn(
            "stat_keyword_match",
            f.when(f.col("kpi_name").rlike(avg_pattern), "avg")
            .when(f.col("kpi_name").rlike(max_pattern), "max")
            .when(f.col("kpi_name").rlike(min_pattern), "min")
            .otherwise("unknown"),
        )
        .withColumn(
            "unit_match",
            f.when(f.col("unit").isin(*mean_like_units, "mean_like"), "mean_like")
            .when(f.col("unit").isin(*volume_units, "volume"), "volume")
            .otherwise("unknown"),
        )
        .withColumn(
            "keyword_match",
            f.when(f.col("kpi_name").rlike(ratio_pattern), "ratio")
            .when(f.col("kpi_name").rlike(mean_like_pattern), "mean_like")
            .when(f.col("kpi_name").rlike(volume_pattern), "volume")
            .otherwise("unknown"),
        )
        .withColumn(
            "kpi_character",
            # Check stat_keyword first. If it's not "unknown", use it.
            f.when(f.col("stat_keyword_match") != "unknown", f.col("stat_keyword_match"))
            # check units
            .when(f.col("unit_match") != "unknown", f.col("unit_match"))
            # check keywords
            .when(f.col("keyword_match") != "unknown", f.col("keyword_match"))
            # value range fallback
            .when(
                f.col("kpi_min") > 0,
                "mean_like",  # when minimal value is smaller than 0 => 'mean_like' category
            )
            .when(
                (f.col("kpi_min") >= 0) & (f.col("kpi_max") <= 100),
                "ratio",  # when minimal and maximal values are between 0 and 100 => 'ratio' cat
            )
            .otherwise("unknown"),
        )
        .withColumn(
            "classification_source",
            f.when(f.col("stat_keyword_match") != "unknown", "stat_keyword")
            .when(f.col("unit_match").isin("mean_like", "volume"), "unit")
            .when(f.col("keyword_match") != "unknown", "keyword")
            .otherwise("value_range_fallback"),
        )
        .withColumn(
            "agg_method",
            f.when(
                (f.col("kpi_character") == "mean_like")
                | (f.col("kpi_character") == "ratio")
                | (f.col("kpi_character") == "avg"),
                "avg",
            )
            .when((f.col("kpi_character") == "max"), "max")
            .when((f.col("kpi_character") == "min"), "min")
            .otherwise("sum"),
        )
        .drop("keyword_match")
    )
    return kpi_classified
