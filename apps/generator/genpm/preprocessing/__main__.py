"""genpm.preprocessing — runnable as `python -m genpm.preprocessing` or a SparkSubmit target.

Receives the finalized Airflow `dag_run.conf` as a single JSON argument (`--conf-json`) instead of
~20 individual flags. The job owns its schema and defaults via `PreprocessingConfig.from_conf`.
"""

import argparse
import json
import os

from genpm.utils.spark_bootstrap import bootstrap_spark_submit_driver

bootstrap_spark_submit_driver()

from genpm.preprocessing.configs import PreprocessingConfig  # noqa: E402
from genpm.preprocessing.run import run_preprocessing  # noqa: E402
from genpm.utils.spark_session import SparkDataManager  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for PM synthetic data generation",
    )
    parser.add_argument(
        "--conf-json",
        required=True,
        help="Finalized dag_run.conf as a JSON string (keys: s3_key, dag_args, genpm_run_id, ...).",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET", "datasets"),
        help="S3 bucket for relative keys (defaults to $S3_BUCKET).",
    )
    args = parser.parse_args(argv)

    conf = json.loads(args.conf_json)
    cfg = PreprocessingConfig.from_conf(conf, bucket=args.bucket)

    dataset_id = conf.get("dataset_id")
    print(f"Preprocessing starting (dataset_id={dataset_id}, output={cfg.output_path_prefix})")

    with SparkDataManager(app_name="PreprocessingSparkJob") as sdm:
        run_preprocessing(sdm, cfg)


if __name__ == "__main__":
    main()
