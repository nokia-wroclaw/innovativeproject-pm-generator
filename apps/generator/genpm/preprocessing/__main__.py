"""genpm.preprocessing — runnable as `python -m genpm.preprocessing`."""

import argparse

from genpm.preprocessing.configs import PreprocessingConfig
from genpm.preprocessing.run import run_preprocessing
from genpm.utils.consts import SPARK_CONFIGS
from genpm.utils.utils import SparkDataManager


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocessing Pipeline for PM Data Synthetic Data Generation",
    )
    # RAW paths - input
    parser.add_argument("--pm-data-raw-path", required=True, help="Path to raw PM data")
    parser.add_argument("--kpi-definitions-raw-path", required=True, help="Path to KPI definitions")
    parser.add_argument("--simple-reports-raw-path", required=True, help="Path to simple reports")
    # Output
    parser.add_argument("--output-path-prefix", required=True, help="Prefix for output paths")
    # KPI global density thresholds
    parser.add_argument("--kpi-min-global-density", type=float, default=0.5)
    parser.add_argument("--kpi-global-min-frac-cells-passing", type=float, default=0.8)
    # KPI window coverage
    parser.add_argument("--kpi-window-coverage-frac", type=float, default=0.917)
    # Max gap filtering
    parser.add_argument("--min-imputable-gap-frac", type=float, default=0.8)
    # Stale KPI filtering
    parser.add_argument("--kpi-min-std-val", type=float, default=0.01)
    parser.add_argument("--max-zero-frac", type=float, default=0.95)
    # Training data windows
    parser.add_argument("--window-width-hours", type=int, default=168)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--max-gap-hours", type=int, default=6)
    parser.add_argument("--min-joint-windows-abs", type=int, default=None, required=False)
    parser.add_argument("--no-impute", action="store_true")

    args = parser.parse_args(argv)

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
        impute=not args.no_impute,
    )

    sdm = SparkDataManager(SPARK_CONFIGS["HALF_SAFE"])

    run_preprocessing(sdm, cfg)


if __name__ == "__main__":
    main()
