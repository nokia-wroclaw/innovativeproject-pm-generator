import argparse
import json
import os

from genpm.utils.spark_bootstrap import bootstrap_spark_submit_driver

bootstrap_spark_submit_driver()

from genpm.raw_vis.configs import RawVisConfig  # noqa: E402
from genpm.raw_vis.data_visualisation import make_kpi_analysis, make_summary_json  # noqa: E402
from genpm.raw_vis.dataset_job import run_dataset_visualization  # noqa: E402
from genpm.utils.spark_session import SparkDataManager  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw PM data visualisation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_dataset = sub.add_parser(
        "dataset",
        help="Airflow job: read RAW parquet from S3 and write visualization JSON artifacts",
    )
    p_dataset.add_argument(
        "--conf-json",
        required=True,
        help="Finalized dag_run.conf as a JSON string (keys: dataset_id, s3_key, ...).",
    )
    p_dataset.add_argument(
        "--bucket",
        default=os.environ.get("S3_BUCKET", "datasets"),
        help="S3 bucket for relative keys (defaults to $S3_BUCKET).",
    )

    p_summary = sub.add_parser(
        "summary",
        help="Generate a dataset summary JSON (schema, basic info, KPI catalog, BTS coverage heatmap)",
    )
    p_summary.add_argument("--raw-pm-data-path", required=True, help="Path to raw PM parquet data")
    p_summary.add_argument(
        "--output-path",
        default="data_summary.json",
        help="Output JSON file (default: data_summary.json)",
    )

    p_kpi = sub.add_parser(
        "kpi-analysis",
        help="Generate per-KPI timeline + distribution JSON for a given set of KPI IDs",
    )
    p_kpi.add_argument("--raw-pm-data-path", required=True, help="Path to raw PM parquet data")
    p_kpi.add_argument(
        "--output-path",
        default="kpi_analysis.json",
        help="Output JSON file (default: kpi_analysis.json)",
    )
    p_kpi.add_argument(
        "--kpi-list", required=True, nargs="+", metavar="KPI_ID", help="KPI IDs to analyse"
    )

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset":
        conf = json.loads(args.conf_json)
        s3_key = str(conf.get("s3_key") or "").strip()
        if not s3_key:
            raise ValueError("dag_run.conf missing required key: s3_key")
        run_dataset_visualization(
            dataset_id=str(conf.get("dataset_id") or ""),
            s3_key=s3_key,
            bucket=args.bucket,
        )
        return

    cfg = RawVisConfig(
        raw_pm_data_path=args.raw_pm_data_path,
        output_path=args.output_path,
    )

    with SparkDataManager() as sdm:
        raw_df = sdm.read_parquet(cfg.raw_pm_data_path)

        if args.command == "summary":
            make_summary_json(raw_df, cfg.output_path)
        elif args.command == "kpi-analysis":
            make_kpi_analysis(raw_df, args.kpi_list, cfg.output_path)


if __name__ == "__main__":
    main()
