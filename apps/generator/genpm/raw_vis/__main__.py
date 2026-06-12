"""genpm.raw_vis — runnable as `python -m genpm.raw_vis <summary|kpi-analysis>`."""

import argparse

from genpm.raw_vis.configs import RawVisConfig
from genpm.raw_vis.data_visualisation import make_kpi_analysis, make_summary_json
from genpm.utils.consts import SPARK_CONFIGS
from genpm.utils.spark_session import SparkDataManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw PM data visualisation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- summary (run once) ---
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

    # --- kpi-analysis (can be called multiple times with different KPI sets) ---
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

    cfg = RawVisConfig(
        raw_pm_data_path=args.raw_pm_data_path,
        output_path=args.output_path,
    )

    sdm = SparkDataManager(additional_conf=SPARK_CONFIGS["HALF_SAFE"])
    raw_df = sdm.read_parquet(cfg.raw_pm_data_path)

    if args.command == "summary":
        make_summary_json(raw_df, cfg.output_path)
    elif args.command == "kpi-analysis":
        make_kpi_analysis(raw_df, args.kpi_list, cfg.output_path)


if __name__ == "__main__":
    main()
