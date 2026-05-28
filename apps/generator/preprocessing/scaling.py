from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS

cfg = SPARK_CONFIGS["WINDOW_HEAVY"]

spark = (
    SparkSession.builder.appName("GenPM-fe")
    .config(map=cfg)
    .config("spark.log.level", "ERROR")
    .getOrCreate()
)

EDA_DATA_PATH = SHARED_DIR_PATH / "eda_data"
raw_pm_path = EDA_DATA_PATH / "raw_pm_data"
pm_kpi_pivot = EDA_DATA_PATH / "pm_data_pivot"
sample_path = EDA_DATA_PATH / "sample"
pm_stats_path = EDA_DATA_PATH / "stats"
pm_agg_path = EDA_DATA_PATH / "agg"
pm_metadata = EDA_DATA_PATH / "pm_metadata"
PREPROCESSED_DATASET_PATH = SHARED_DIR_PATH / "preprocessed_dataset"

# imputed_path = PREPROCESSED_DATASET_PATH / "pm_imputed"

# pm_imputed = spark.read.parquet(str(imputed_path))


class GroupedKPIScaler:
    """
    Scales one value column per group.

    Design goals:
    - no imputation,
    - fit ignores nulls,
    - transform preserves nulls,
    - transform replaces value_col with scaled values,
    - transform adds scaled_flag,
    - inverse_transform restores value_col using scaled_flag,
    - fully Spark-native transform via join.
    """

    def __init__(
        self,
        value_col: str,
        group_cols: list[str],
        min_valid_points: int = 4,
        percentile_accuracy: int = 10000,
        broadcast_params: bool = False,
        scaled_flag_col: str = "scaled_flag",
    ):
        self.value_col = value_col
        self.group_cols = group_cols
        self.min_valid_points = min_valid_points
        self.percentile_accuracy = percentile_accuracy
        self.broadcast_params = broadcast_params
        self.scaled_flag_col = scaled_flag_col

        self.params_df: DataFrame
        self.audit_df: DataFrame

    def _stats_df(self, df: DataFrame) -> DataFrame:
        v = f.col(self.value_col).cast("double")

        return (
            df.groupBy(*self.group_cols)
            .agg(
                f.count(v).alias("n_valid"),
                f.mean(v).alias("mean_raw"),
                f.stddev_pop(v).alias("std_raw"),
                f.min(v).alias("min_raw"),
                f.max(v).alias("max_raw"),
                f.skewness(v).alias("skew_raw"),
                f.kurtosis(v).alias("kurt_raw"),
                f.percentile_approx(
                    v,
                    f.array(
                        f.lit(0.01),
                        f.lit(0.25),
                        f.lit(0.50),
                        f.lit(0.75),
                        f.lit(0.99),
                    ),
                    f.lit(self.percentile_accuracy),
                ).alias("pct"),
                (
                    f.sum(f.when(f.col(self.value_col).isNull(), 1).otherwise(0))
                    / f.count(f.lit(1))
                ).alias("null_pct"),
            )
            .select(
                *self.group_cols,
                "n_valid",
                "mean_raw",
                "std_raw",
                "min_raw",
                "max_raw",
                "skew_raw",
                "kurt_raw",
                f.col("pct")[0].alias("q01"),
                f.col("pct")[1].alias("q25"),
                f.col("pct")[2].alias("q50"),
                f.col("pct")[3].alias("q75"),
                f.col("pct")[4].alias("q99"),
                "null_pct",
            )
        )

    def _with_scaler_choice(self, stats_df: DataFrame) -> DataFrame:
        iqr = (f.col("q75") - f.col("q25")) + f.lit(1e-8)

        outlier_score = f.greatest(
            (f.col("q99") - f.col("q75")) / iqr,
            (f.col("q25") - f.col("q01")) / iqr,
        )

        return (
            stats_df.withColumn("iqr_raw", f.col("q75") - f.col("q25"))
            .withColumn("range_raw", f.col("max_raw") - f.col("min_raw"))
            .withColumn("outlier_score", outlier_score)
            .withColumn(
                "scaler",
                f.when(f.col("n_valid") < f.lit(self.min_valid_points), f.lit("SKIP"))
                .when(f.col("outlier_score") > f.lit(3.0), f.lit("robust"))
                .when(
                    (f.abs(f.col("skew_raw")) > f.lit(2.0)) & (f.col("min_raw") > f.lit(0)),
                    f.lit("log_standard"),
                )
                .when(f.col("kurt_raw") < f.lit(-0.8), f.lit("minmax"))
                .otherwise(f.lit("standard")),
            )
        )

    def fit(self, df: DataFrame) -> DataFrame:
        stats_df = self._stats_df(df)
        choice_df = self._with_scaler_choice(stats_df)

        log_stats_df = (
            df.join(
                choice_df.filter(f.col("scaler") == "log_standard").select(*self.group_cols),
                on=self.group_cols,
                how="inner",
            )
            .where(f.col(self.value_col).isNotNull())
            .groupBy(*self.group_cols)
            .agg(
                f.mean(f.log1p(f.col(self.value_col).cast("double"))).alias("log_mean"),
                f.stddev_pop(f.log1p(f.col(self.value_col).cast("double"))).alias("log_std"),
            )
        )

        params_df = (
            choice_df.join(log_stats_df, on=self.group_cols, how="left")
            .withColumn(
                "reason",
                f.when(
                    f.col("n_valid") < f.lit(self.min_valid_points), f.lit("too_few_valid_points")
                )
                .when(
                    (f.col("scaler") == "standard")
                    & (f.col("std_raw").isNull() | (f.col("std_raw") < f.lit(1e-8))),
                    f.lit("zero_or_invalid_std"),
                )
                .when(
                    (f.col("scaler") == "robust")
                    & (f.col("iqr_raw").isNull() | (f.col("iqr_raw") < f.lit(1e-8))),
                    f.lit("zero_or_invalid_iqr"),
                )
                .when(
                    (f.col("scaler") == "minmax")
                    & (f.col("range_raw").isNull() | (f.col("range_raw") < f.lit(1e-8))),
                    f.lit("zero_or_invalid_range"),
                )
                .when(
                    (f.col("scaler") == "log_standard")
                    & (f.col("log_std").isNull() | (f.col("log_std") < f.lit(1e-8))),
                    f.lit("zero_or_invalid_log_std"),
                )
                .otherwise(f.lit("ok")),
            )
            .withColumn(
                "scaler",
                f.when(f.col("reason") != f.lit("ok"), f.lit("SKIP")).otherwise(f.col("scaler")),
            )
            .withColumn(
                "param_a",
                f.when(f.col("scaler") == "standard", f.col("mean_raw"))
                .when(f.col("scaler") == "robust", f.col("q50"))
                .when(f.col("scaler") == "minmax", f.col("min_raw"))
                .when(f.col("scaler") == "log_standard", f.col("log_mean")),
            )
            .withColumn(
                "param_b",
                f.when(f.col("scaler") == "standard", f.col("std_raw") + f.lit(1e-8))
                .when(f.col("scaler") == "robust", f.col("iqr_raw") + f.lit(1e-8))
                .when(f.col("scaler") == "minmax", f.col("range_raw") + f.lit(1e-8))
                .when(f.col("scaler") == "log_standard", f.col("log_std") + f.lit(1e-8)),
            )
            .select(
                *self.group_cols,
                "null_pct",
                "n_valid",
                "scaler",
                "reason",
                "param_a",
                "param_b",
            )
        )

        self.params_df = params_df.cache()
        self.audit_df = params_df.select(
            *self.group_cols, "null_pct", "n_valid", "scaler", "reason"
        )

        return self.audit_df

    def transform(
        self,
        df: DataFrame,
        keep_params: bool = False,
    ) -> DataFrame:
        """
        Replace value_col with scaled values and add scaled_flag_col.

        scaled_flag_col:
        - true  -> row was scaled
        - false -> row was not scaled (null value, missing params, or SKIP group)
        """
        if self.params_df is None:
            raise ValueError("Call fit() before transform().")

        params_df = f.broadcast(self.params_df) if self.broadcast_params else self.params_df
        joined = df.join(params_df, on=self.group_cols, how="left")

        x = f.col(self.value_col).cast("double")

        can_scale = f.col("scaler").isNotNull() & (f.col("scaler") != f.lit("SKIP")) & x.isNotNull()

        scaled_value = (
            f.when(
                can_scale & (f.col("scaler") == "standard"),
                (x - f.col("param_a")) / f.col("param_b"),
            )
            .when(
                can_scale & (f.col("scaler") == "robust"), (x - f.col("param_a")) / f.col("param_b")
            )
            .when(
                can_scale & (f.col("scaler") == "minmax"), (x - f.col("param_a")) / f.col("param_b")
            )
            .when(
                can_scale & (f.col("scaler") == "log_standard"),
                (f.log1p(x) - f.col("param_a")) / f.col("param_b"),
            )
            .otherwise(x)
        )

        out = joined.withColumn(self.scaled_flag_col, can_scale).withColumn(
            self.value_col, scaled_value
        )

        if not keep_params:
            out = out.drop("null_pct", "n_valid", "scaler", "reason", "param_a", "param_b")

        return out

    def inverse_transform(
        self,
        df: DataFrame,
        keep_params: bool = False,
    ) -> DataFrame:
        """
        Replace value_col with restored original-scale values using scaled_flag_col.
        Only rows with scaled_flag_col == true are inverse-transformed.
        """
        if self.params_df is None:
            raise ValueError("Call fit() before inverse_transform().")

        params_df = f.broadcast(self.params_df) if self.broadcast_params else self.params_df
        joined = df.join(params_df, on=self.group_cols, how="left")

        x = f.col(self.value_col).cast("double")

        can_restore = (
            f.col(self.scaled_flag_col).eqNullSafe(f.lit(True))
            & f.col("scaler").isNotNull()
            & (f.col("scaler") != f.lit("SKIP"))
            & x.isNotNull()
        )

        restored_value = (
            f.when(
                can_restore & (f.col("scaler") == "standard"),
                x * f.col("param_b") + f.col("param_a"),
            )
            .when(
                can_restore & (f.col("scaler") == "robust"), x * f.col("param_b") + f.col("param_a")
            )
            .when(
                can_restore & (f.col("scaler") == "minmax"), x * f.col("param_b") + f.col("param_a")
            )
            .when(
                can_restore & (f.col("scaler") == "log_standard"),
                f.expm1(x * f.col("param_b") + f.col("param_a")),
            )
            .otherwise(x)
        )

        out = joined.withColumn(self.value_col, restored_value)

        if not keep_params:
            out = out.drop("null_pct", "n_valid", "scaler", "reason", "param_a", "param_b")

        return out

    def summary(self) -> DataFrame:
        if self.audit_df is None:
            raise ValueError("Call fit() first.")
        return self.audit_df


# scaler = GroupedKPIScaler(
#     value_col="kpi_value",
#     group_cols=["kpi_id", "bts_id", "distname"],
#     min_valid_points=4,
#     percentile_accuracy=5000,
#     broadcast_params=False,
# )

# audit_df = scaler.fit(imputed_pm)
# df_scaled = scaler.transform(imputed_pm)
# df_restored = scaler.inverse_transform(df_scaled)
