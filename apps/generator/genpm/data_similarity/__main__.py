"""genpm.data_similarity — runnable as `python -m genpm.data_similarity`."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import boto3
import joblib
from botocore.config import Config

from genpm.data_similarity.configs import DataSimilarityConfig
from genpm.data_similarity.run import run_data_similarity
from genpm.utils.consts import SPARK_CONFIGS
from genpm.utils.spark_session import SparkDataManager


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Data similarity validation: real vs synthetic time series.",
    )

    # Airflow run conf
    parser.add_argument(
        "--conf-json",
        default=None,
        help="Dag run conf JSON string. If provided, S3 paths and params are extracted automatically.",
    )

    # RAW paths - input
    parser.add_argument(
        "--real-data-path",
        default=None,
        help="Path to real data parquet. If omitted, fake data is generated.",
    )
    parser.add_argument(
        "--synth-data-path",
        default=None,
        help="Path to synth data parquet. If omitted, fake data is generated.",
    )

    # Output
    parser.add_argument(
        "--output-path-prefix",
        default=None,
        help="Local directory or S3 prefix for figures and summary JSON",
    )

    # KPIs to validate
    parser.add_argument(
        "--single-kpi-cols",
        nargs="+",
        default=[],
        help="KPI columns to validate individually (one figure per KPI)",
    )
    parser.add_argument(
        "--multi-kpi-cols",
        nargs="+",
        default=[],
        help="KPI columns to validate jointly (one multivariate figure). Needs >= 2.",
    )

    # Time column
    parser.add_argument("--ts-col", default="ts")

    # Cell config filtering for real data
    parser.add_argument(
        "--cell-config-cols",
        nargs="+",
        default=None,
        help=(
            "Ordered [CELL] column names in the real data to combine into config_id "
            "(must match the order used during training, e.g. from cell_config_map['config_cols'])."
        ),
    )
    parser.add_argument(
        "--cell-configs",
        nargs="+",
        default=None,
        help=(
            "Target config values in the same order as --cell-config-cols. "
            "Real data is filtered to rows whose joined config_id matches these values."
        ),
    )

    # Single-KPI metric parameters
    parser.add_argument("--acf-max-lag", type=int, default=24 * 8)
    parser.add_argument("--ls-min-period-h", type=float, default=2.0)
    parser.add_argument("--ls-max-period-h", type=float, default=24 * 14)
    parser.add_argument("--ls-n-freq", type=int, default=2000)
    parser.add_argument("--kde-n-grid", type=int, default=300)

    # Multi-KPI metric parameters
    parser.add_argument("--n-projections", type=int, default=200)

    # Plot dimensions
    parser.add_argument("--plot-width", type=int, default=1500)
    parser.add_argument("--plot-height-single", type=int, default=850)
    parser.add_argument("--plot-height-multi", type=int, default=750)

    args = parser.parse_args(argv)

    if args.conf_json:
        conf = json.loads(args.conf_json)
        bucket = os.getenv("S3_BUCKET", "datasets")

        comp_path = conf.get("comparison_dataset_path")
        if not comp_path:
            raise ValueError("comparison_dataset_path is missing from conf, cannot run similarity.")

        out_prefix = conf["dag_args"]["output_path_prefix"]

        args.real_data_path = f"s3a://{bucket}/{comp_path}/pm_df_wide_indexed_winds"
        args.synth_data_path = f"s3a://{bucket}/{out_prefix}/*.parquet"
        args.output_path_prefix = f"s3a://{bucket}/{out_prefix}"
        args.single_kpi_cols = conf["dag_args"]["kpi_list"]
        args.multi_kpi_cols = conf["dag_args"]["kpi_list"]
        args.real_ts_col = "start_time"
        args.synth_ts_col = "timestamp"

        cell_id = conf["dag_args"].get("cell_id")
        if cell_id:
            config_key = conf.get("config_path")
            if config_key:
                print(f"Loading cell configs for {cell_id} from s3://{bucket}/{config_key}")
                s3 = boto3.client(
                    "s3",
                    endpoint_url=os.getenv("S3_URL", "http://minio:9000"),
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / "cell_config_map.pkl"
                    s3.download_file(bucket, config_key, str(tmp_path))
                    cmap = joblib.load(tmp_path)
                    args.cell_config_cols = cmap["config_cols"]
                    args.cell_configs = cmap["map"][cell_id]

    if not args.output_path_prefix:
        parser.error("--output-path-prefix is required when not using --conf-json")

    cfg = DataSimilarityConfig(
        real_data_path=args.real_data_path,
        synth_data_path=args.synth_data_path,
        output_path_prefix=args.output_path_prefix,
        single_kpi_cols=args.single_kpi_cols,
        multi_kpi_cols=args.multi_kpi_cols,
        ts_col=args.ts_col,
        real_ts_col=getattr(args, "real_ts_col", None),
        synth_ts_col=getattr(args, "synth_ts_col", None),
        acf_max_lag=args.acf_max_lag,
        ls_min_period_h=args.ls_min_period_h,
        ls_max_period_h=args.ls_max_period_h,
        ls_n_freq=args.ls_n_freq,
        kde_n_grid=args.kde_n_grid,
        n_projections=args.n_projections,
        cell_config_cols=args.cell_config_cols,
        cell_configs=args.cell_configs,
    )

    sdm = SparkDataManager(additional_conf=SPARK_CONFIGS["HALF_SAFE"])

    run_data_similarity(sdm, cfg)


if __name__ == "__main__":
    main()
