from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame

from utils import SparkDataManager
from utils.consts import SHARED_DIR_PATH

sdm = SparkDataManager()

PM_DATA_PATH = [SHARED_DIR_PATH / "raw_data" / f"pm_kpis_part{i}.parquet" for i in range(1, 6)]

KPI_DEFINITIONS_PATH = SHARED_DIR_PATH / "raw_data" / "kpis_definitions.parquet"

SIMPLE_REPORTS_PATH = SHARED_DIR_PATH / "raw_data" / "simple_reports.parquet"


def load_clean_pm_data():
    list_of_pm_dfs = [sdm.read_parquet(dp) for dp in PM_DATA_PATH]

    pm_df: DataFrame = reduce(lambda df1, df2: df1.unionByName(df2), list_of_pm_dfs)

    # tbc


kpis_definitions_df = spark.read.parquet(DATA_PATHS["kpis_definitions"])
simple_reports_df = spark.read.parquet(DATA_PATHS["simple_reports"])


pm_df_all.printSchema()


pm_df_all.show()
pm_df_all.printSchema()


# FULL data cleaning
print(f"PM df count with duplicates: {pm_df_all.count()}")
pm_df_all = pm_df_all.dropDuplicates()
print(f"PM df count withOUT duplicates: {pm_df_all.count()}")

# Missing values in pm:
pm_df_all.select([f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in pm_df_all.columns]).show()

pm_df_all = pm_df_all.dropna(subset=("start_time", "bts_anon", "distname_anon"))


# PIVOT ATTEMPT
def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


kpis = [r.kpi_id for r in pm_df_all.select("kpi_id").distinct().collect()]
batches = list(chunk(kpis, 30))  # mniejsze batch = mniej RAM
print(f"{len(batches)=}")

pm_df_all = pm_df_all.repartition("kpi_id").persist()

pm_df_all.count()

final_df = None


for i, batch in enumerate(batches):
    print(f"Batch {i}")

    df_batch = pm_df_all.filter(f.col("kpi_id").isin(batch))

    df_pivot = (
        df_batch.groupBy("bts_anon", "distname_anon", "start_time")
        .pivot("kpi_id")
        .agg(f.first("kpi_value"))
    )

    if i % 5 == 0 and i != 0:
        final_df = final_df.checkpoint()

    if final_df is None:
        final_df = df_pivot
    else:
        final_df = final_df.join(df_pivot, ["bts_anon", "distname_anon", "start_time"], "outer")

print("Pivot complete!")


eda_data_path = Path("/home/sparkuser/app/apps/data/shared_dir/eda_data")
raw_pm_path = eda_data_path / "raw_pm_data"
pm_pivot = eda_data_path / "pm_data_pivot"
sample_path = eda_data_path / "sample"
pm_stats_path = eda_data_path / "stats"
pm_agg_path = eda_data_path / "agg"
other_data = eda_data_path / "pm_metadata"


pm_df_all = pm_df_all.withColumn("start_date", f.to_date("start_time"))
pm_df_all = pm_df_all.withColumn("start_hour", f.hour("start_time"))
pm_df_all = pm_df_all.withColumnsRenamed({"bts_anon": "bts_id", "distname_anon": "distname"})
final_df = final_df.withColumnsRenamed({"bts_anon": "bts_id", "distname_anon": "distname"})


final_df.count()
final_df.printSchema()
final_df.write.parquet(str(pm_pivot), mode="overwrite")


kpis_definitions_df.select(
    [f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in kpis_definitions_df.columns]
).show()

simple_reports_df.select(
    [f.count(f.when(f.col(c).isNull(), c)).alias(c) for c in simple_reports_df.columns]
).show()


pm_df_all.write.partitionBy("start_date").parquet(str(raw_pm_path), mode="overwrite")
kpis_definitions_df.write.parquet(str(other_data / "kpis_definitions"), mode="overwrite")
simple_reports_df.write.parquet(str(other_data / "simple_reports"), mode="overwrite")


# calculate stats df
pm_df_stats = pm_df_all.groupBy("kpi_id").agg(
    f.count("*").alias("count"),
    f.avg("kpi_value").alias("mean"),
    f.stddev("kpi_value").alias("std"),
    f.min("kpi_value").alias("min"),
    f.max("kpi_value").alias("max"),
    f.expr("percentile_approx(kpi_value, array(0.25, 0.5, 0.75))").alias("quantiles"),
)

# pm aggregated per day
pm_df_grouped_date = pm_df_all.groupBy("bts_id", "distname", "kpi_id", "start_date").agg(
    f.avg("kpi_value").alias("kpi_mean"),
    f.min("kpi_value").alias("kpi_min"),
    f.max("kpi_value").alias("kpi_max"),
    f.count("*").alias("kpi_count"),
    # add other aggregations in need
)

# pm aggregated per hour
pm_df_grouped_hours = pm_df_all.groupBy("bts_id", "distname", "kpi_id", "start_hour").agg(
    f.avg("kpi_value").alias("kpi_mean"),
    f.min("kpi_value").alias("kpi_min"),
    f.max("kpi_value").alias("kpi_max"),
    f.count("*").alias("kpi_count"),
    # add other aggregations in need
)

# sample df
kpi_list = pm_df_all.select("kpi_id").distinct().rdd.flatMap(lambda x: x).collect()
kpi_fractions = {k: 0.01 for k in kpi_list}
pm_sample = pm_df_all.sampleBy("kpi_id", kpi_fractions, seed=42).limit(6_000_000)


# pm grouped by bts and cell
pm_df_grouped_distname = pm_df_grouped_hours = pm_df_all.groupBy(
    "bts_id", "distname", "kpi_id"
).agg(
    f.avg("kpi_value").alias("kpi_mean"),
    f.min("kpi_value").alias("kpi_min"),
    f.max("kpi_value").alias("kpi_max"),
    f.count("*").alias("kpi_count"),
    # add other aggregations in need
)


pm_df_stats.write.parquet(str(pm_stats_path / "kpi_stats"), mode="overwrite")
pm_df_grouped_date.write.parquet(
    str(pm_agg_path / "pm_date_agg"), mode="overwrite", partitionBy="start_date"
)
pm_df_grouped_hours.write.parquet(str(pm_agg_path / "pm_hour_agg"), mode="overwrite")
pm_sample.write.parquet(str(sample_path / "pm_sample"), mode="overwrite")
pm_df_grouped_distname.write.parquet(str(pm_agg_path / "pm_distname_agg"), mode="overwrite")


pm_df_all.repartition("kpi_id")
