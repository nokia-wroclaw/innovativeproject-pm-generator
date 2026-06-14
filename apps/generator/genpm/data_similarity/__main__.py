"""genpm.data_similarity — runnable as `python -m genpm.data_similarity`."""

import argparse

from genpm.data_similarity.configs import DataSimilarityConfig
from genpm.data_similarity.run import run_data_similarity
from genpm.utils.consts import SPARK_CONFIGS
from genpm.utils.utils import SparkDataManager


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Data similarity validation: real vs synthetic time series.",
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
        required=True,
        help="Local directory for figures and summary JSON",
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

    cfg = DataSimilarityConfig(
        real_data_path=args.real_data_path,
        synth_data_path=args.synth_data_path,
        output_path_prefix=args.output_path_prefix,
        single_kpi_cols=args.single_kpi_cols,
        multi_kpi_cols=args.multi_kpi_cols,
        ts_col=args.ts_col,
        acf_max_lag=args.acf_max_lag,
        ls_min_period_h=args.ls_min_period_h,
        ls_max_period_h=args.ls_max_period_h,
        ls_n_freq=args.ls_n_freq,
        kde_n_grid=args.kde_n_grid,
        n_projections=args.n_projections,
    )

    sdm = SparkDataManager(SPARK_CONFIGS["HALF_SAFE"])

    run_data_similarity(sdm, cfg)


if __name__ == "__main__":
    main()
