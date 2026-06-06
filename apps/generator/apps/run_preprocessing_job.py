"""Spark/Airflow entry point for GenPM preprocessing."""

import argparse
import os
import sys
from pathlib import Path


def _ensure_genpm_importable() -> None:
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path("/home/sparkuser/app/apps/generator"),
        Path("/opt/genpm/generator"),
    ]
    for root in candidates:
        if (root / "genpm").is_dir():
            sys.path.insert(0, str(root))
            return
    raise RuntimeError(
        "Could not locate the genpm package. Mount apps/generator into the runtime environment.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GenPM preprocessing pipeline.")
    parser.add_argument("--pm-data-raw-path", required=True)
    parser.add_argument("--kpi-definitions-raw-path", required=True)
    parser.add_argument("--simple-reports-raw-path", required=True)
    parser.add_argument("--output-path-prefix", required=True)
    parser.add_argument("--kpi-min-global-density", type=float, default=0.5)
    parser.add_argument("--kpi-global-min-frac-cells-passing", type=float, default=0.8)
    parser.add_argument("--kpi-window-coverage-frac", type=float, default=0.917)
    parser.add_argument("--min-imputable-gap-frac", type=float, default=0.8)
    parser.add_argument("--kpi-min-std-val", type=float, default=0.01)
    parser.add_argument("--max-zero-frac", type=float, default=0.95)
    parser.add_argument("--window-width-hours", type=int, default=168)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--max-gap-hours", type=int, default=6)
    parser.add_argument("--min-joint-windows-abs", type=int, default=None)
    parser.add_argument("--impute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    _ensure_genpm_importable()

    from genpm.preprocessing.configs import PreprocessingConfig
    from genpm.preprocessing.run import run_preprocessing
    from genpm.utils.consts import SPARK_CONFIGS
    from genpm.utils.utils import SparkDataManager

    args = _build_parser().parse_args(argv)
    cfg = PreprocessingConfig(
        pm_data_raw_path=args.pm_data_raw_path,
        kpi_definitions_raw_path=args.kpi_definitions_raw_path,
        simple_reports_raw_path=args.simple_reports_raw_path,
        output_path_prefix=args.output_path_prefix,
        kpi_min_global_density=args.kpi_min_global_density,
        kpi_global_min_frac_cells_passing=args.kpi_global_min_frac_cells_passing,
        kpi_window_coverage_frac=args.kpi_window_coverage_frac,
        min_imputable_gap_frac=args.min_imputable_gap_frac,
        kpi_min_std_val=args.kpi_min_std_val,
        max_zero_frac=args.max_zero_frac,
        window_width_hours=args.window_width_hours,
        stride_hours=args.stride_hours,
        max_gap_hours=args.max_gap_hours,
        min_joint_windows_abs=args.min_joint_windows_abs,
        impute=args.impute,
    )

    spark_profile = os.getenv("GENPM_SPARK_PROFILE", "AIRFLOW")
    spark_conf = {**SPARK_CONFIGS[spark_profile], **SparkDataManager.s3_spark_conf()}
    sdm = SparkDataManager(spark_conf)
    run_preprocessing(sdm, cfg)


if __name__ == "__main__":
    main()
