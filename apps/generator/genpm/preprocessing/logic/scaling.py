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
    Scale one numeric value column per group.

    Principles:
    - fit() computes per-group scaling parameters
    - parameters can be persisted to parquet/S3
    - transform() replaces value_col with scaled values
    - inverse_transform() restores original scale by rejoining persisted params
    - no scaled_flag column; scaler choice is the source of truth
    - output fact dataframe stays lean by default
    """

    PARAM_COLUMNS = [
        "null_pct",
        "n_valid",
        "scaler",
        "reason",
        "param_a",
        "param_b",
    ]

    AUDIT_COLUMNS = [
        "null_pct",
        "n_valid",
        "scaler",
        "reason",
    ]

    def __init__(
        self,
        value_col: str,
        group_cols: list[str],
        min_valid_points: int = 4,
        percentile_accuracy: int = 10000,
        broadcast_params: bool = False,
        params_path: str | None = None,
    ):
        self.value_col = value_col
        self.group_cols = group_cols
        self.min_valid_points = min_valid_points
        self.percentile_accuracy = percentile_accuracy
        self.broadcast_params = broadcast_params
        self.params_path = params_path

        self.params_df: DataFrame | None = None
        self.audit_df: DataFrame | None = None

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
        q01 = f.col("q01")
        q25 = f.col("q25")
        q75 = f.col("q75")
        q99 = f.col("q99")
        iqr = (q75 - q25) + f.lit(1e-8)

        outlier_score = f.greatest(
            (q99 - q75) / iqr,
            (q25 - q01) / iqr,
        )

        return stats_df.withColumns(
            {
                "iqr_raw": q75 - q25,
                "range_raw": f.col("max_raw") - f.col("min_raw"),
                "outlier_score": outlier_score,
                "scaler": (
                    f.when(f.col("n_valid") < f.lit(self.min_valid_points), f.lit("SKIP"))
                    .when(outlier_score > f.lit(3.0), f.lit("robust"))
                    .when(
                        (f.abs(f.col("skew_raw")) > f.lit(2.0)) & (f.col("min_raw") > f.lit(0)),
                        f.lit("log_standard"),
                    )
                    .when(f.col("kurt_raw") < f.lit(-0.8), f.lit("minmax"))
                    .otherwise(f.lit("standard"))
                ),
            }
        )

    def _build_params_df(self, df: DataFrame) -> DataFrame:
        stats_df = self._stats_df(df)
        choice_df = self._with_scaler_choice(stats_df)

        log_groups_df = choice_df.filter(f.col("scaler") == "log_standard").select(*self.group_cols)

        log_stats_df = (
            df.join(log_groups_df, on=self.group_cols, how="inner")
            .where(f.col(self.value_col).isNotNull())
            .groupBy(*self.group_cols)
            .agg(
                f.mean(f.log1p(f.col(self.value_col).cast("double"))).alias("log_mean"),
                f.stddev_pop(f.log1p(f.col(self.value_col).cast("double"))).alias("log_std"),
            )
        )

        base_df = choice_df.join(log_stats_df, on=self.group_cols, how="left")

        scaler = f.col("scaler")
        is_standard = scaler == "standard"
        is_robust = scaler == "robust"
        is_minmax = scaler == "minmax"
        is_log_standard = scaler == "log_standard"

        params_df = (
            base_df.withColumns(
                {
                    "reason": (
                        f.when(
                            f.col("n_valid") < f.lit(self.min_valid_points),
                            f.lit("too_few_valid_points"),
                        )
                        .when(
                            is_standard
                            & (f.col("std_raw").isNull() | (f.col("std_raw") < f.lit(1e-8))),
                            f.lit("zero_or_invalid_std"),
                        )
                        .when(
                            is_robust
                            & (f.col("iqr_raw").isNull() | (f.col("iqr_raw") < f.lit(1e-8))),
                            f.lit("zero_or_invalid_iqr"),
                        )
                        .when(
                            is_minmax
                            & (f.col("range_raw").isNull() | (f.col("range_raw") < f.lit(1e-8))),
                            f.lit("zero_or_invalid_range"),
                        )
                        .when(
                            is_log_standard
                            & (f.col("log_std").isNull() | (f.col("log_std") < f.lit(1e-8))),
                            f.lit("zero_or_invalid_log_std"),
                        )
                        .otherwise(f.lit("ok"))
                    ),
                }
            )
            .withColumns(
                {
                    "scaler": f.when(f.col("reason") != "ok", f.lit("SKIP")).otherwise(scaler),
                    "param_a": (
                        f.when(f.col("scaler") == "standard", f.col("mean_raw"))
                        .when(f.col("scaler") == "robust", f.col("q50"))
                        .when(f.col("scaler") == "minmax", f.col("min_raw"))
                        .when(f.col("scaler") == "log_standard", f.col("log_mean"))
                    ),
                    "param_b": (
                        f.when(f.col("scaler") == "standard", f.col("std_raw") + f.lit(1e-8))
                        .when(f.col("scaler") == "robust", f.col("iqr_raw") + f.lit(1e-8))
                        .when(f.col("scaler") == "minmax", f.col("range_raw") + f.lit(1e-8))
                        .when(f.col("scaler") == "log_standard", f.col("log_std") + f.lit(1e-8))
                    ),
                }
            )
            .select(*self.group_cols, *self.PARAM_COLUMNS)
        )

        return params_df

    def fit(
        self,
        df: DataFrame,
        params_path: str | None = None,
        write_mode: str = "overwrite",
        cache_params: bool = False,
    ) -> DataFrame:
        """
        Fit scaling parameters per group.

        If params_path is provided, writes params_df to parquet immediately.
        Returns a lean audit dataframe.
        """
        resolved_path = params_path or self.params_path

        self.params_df = self._build_params_df(df)

        if cache_params:
            self.params_df = self.params_df.cache()

        self.audit_df = self.params_df.select(*self.group_cols, *self.AUDIT_COLUMNS)

        if resolved_path:
            self.params_df.write.mode(write_mode).parquet(resolved_path)

        return self.audit_df

    def save_params(self, path: str | None = None, mode: str = "overwrite") -> None:
        """
        Persist fitted params to parquet.
        """
        if self.params_df is None:
            raise ValueError("Call fit() before save_params().")

        resolved_path = path or self.params_path
        if not resolved_path:
            raise ValueError("No params path provided.")

        self.params_df.write.mode(mode).parquet(resolved_path)

    @classmethod
    def load_params_parquet(
        cls,
        spark: SparkSession,
        value_col: str,
        group_cols: list[str],
        path: str,
        min_valid_points: int = 4,
        percentile_accuracy: int = 10000,
        broadcast_params: bool = False,
    ) -> GroupedKPIScaler:
        instance = cls(
            value_col=value_col,
            group_cols=group_cols,
            min_valid_points=min_valid_points,
            percentile_accuracy=percentile_accuracy,
            broadcast_params=broadcast_params,
            params_path=path,
        )
        instance.params_df = spark.read.parquet(path)
        instance.audit_df = instance.params_df.select(*instance.group_cols, *instance.AUDIT_COLUMNS)
        return instance

    def _get_params_df(self) -> DataFrame:
        if self.params_df is None:
            raise ValueError("Params are not available. Call fit() or load_params() first.")

        return f.broadcast(self.params_df) if self.broadcast_params else self.params_df

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Scale value_col using fitted group params.

        By default drops param columns from output to keep fact data lean.
        """
        params_df = self._get_params_df()
        joined = df.join(params_df, on=self.group_cols, how="left")

        x = f.col(self.value_col).cast("double")
        scaler = f.col("scaler")

        can_scale = scaler.isNotNull() & (scaler != f.lit("SKIP")) & x.isNotNull()

        scaled_value = (
            f.when(
                can_scale & (scaler == "standard"),
                (x - f.col("param_a")) / f.col("param_b"),
            )
            .when(
                can_scale & (scaler == "robust"),
                (x - f.col("param_a")) / f.col("param_b"),
            )
            .when(
                can_scale & (scaler == "minmax"),
                (x - f.col("param_a")) / f.col("param_b"),
            )
            .when(
                can_scale & (scaler == "log_standard"),
                (f.log1p(x) - f.col("param_a")) / f.col("param_b"),
            )
            .otherwise(x)
        )

        out = joined.withColumns({self.value_col: scaled_value})

        out = out.drop(*self.PARAM_COLUMNS)

        return out

    def inverse_transform(self, df: DataFrame, keep_params: bool = False) -> DataFrame:
        """
        Restore original-scale value_col using fitted group params.

        This assumes the incoming df was previously transformed by this scaler
        design, or otherwise contains values already in scaled space.
        """
        params_df = self._get_params_df()
        joined = df.join(params_df, on=self.group_cols, how="left")

        x = f.col(self.value_col).cast("double")
        scaler = f.col("scaler")

        can_restore = scaler.isNotNull() & (scaler != f.lit("SKIP")) & x.isNotNull()

        restored_value = (
            f.when(
                can_restore & (scaler == "standard"),
                x * f.col("param_b") + f.col("param_a"),
            )
            .when(
                can_restore & (scaler == "robust"),
                x * f.col("param_b") + f.col("param_a"),
            )
            .when(
                can_restore & (scaler == "minmax"),
                x * f.col("param_b") + f.col("param_a"),
            )
            .when(
                can_restore & (scaler == "log_standard"),
                f.expm1(x * f.col("param_b") + f.col("param_a")),
            )
            .otherwise(x)
        )

        out = joined.withColumns({self.value_col: restored_value})

        if not keep_params:
            out = out.drop(*self.PARAM_COLUMNS)

        return out

    def summary(self) -> DataFrame:
        """
        Return fitted audit dataframe.
        """
        if self.audit_df is None:
            raise ValueError("Call fit() or load_params() first.")
        return self.audit_df


# scaler = GroupedKPIScaler(
#     value_col="kpi_value",
#     group_cols=["kpi_id", "bts_id", "distname"],
#     min_valid_points=4,
#     percentile_accuracy=5000,
#     broadcast_params=True,
#     params_path= PREPROCESSED_DATASET_PATH / "scaling_params_df"
# )

# audit_df = scaler.fit(imputed_df)
# df_scaled = scaler.transform(imputed_df)

# # later, in another job
# scaler_reloaded = GroupedKPIScaler(
#     value_col="kpi_value",
#     group_cols=["kpi_id", "bts_id", "distname"],
#     broadcast_params=True,
#     params_path= PREPROCESSED_DATASET_PATH / "scaling_params_df"
# )

# scaler_reloaded.load_params(spark)
# df_restored = scaler_reloaded.inverse_transform(df_scaled)
