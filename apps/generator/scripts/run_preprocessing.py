"""Quick-run preprocessing with hardcoded dev values.

Usage:  python scripts/run_preprocessing.py
        (from the repo root, no install needed)
"""

import sys
from pathlib import Path

# make `import genpm` work without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genpm.preprocessing.configs import PreprocessingConfig
from genpm.preprocessing.run import run_preprocessing
from genpm.utils.consts import SHARED_DIR_PATH, SPARK_CONFIGS
from genpm.utils.spark_session import SparkDataManager

cfg = PreprocessingConfig(
    pm_data_raw_path=str(SHARED_DIR_PATH / "raw_data" / "pm_data"),
    kpi_definitions_raw_path=str(SHARED_DIR_PATH / "raw_data" / "kpi_definitions"),
    simple_reports_raw_path=str(SHARED_DIR_PATH / "raw_data" / "simple_reports"),
    output_path_prefix=str(SHARED_DIR_PATH / "preprocessed_dataset" / "final_pmcm"),
    kpi_min_global_density=0.5,
    min_imputable_gap_frac=0.8,
    kpi_min_std_val=0.01,
    max_zero_frac=0.95,
    window_width_hours=168,
    stride_hours=24,
    max_gap_hours=24,
    min_joint_windows_abs=None,
    forced_kpis=None,
    impute=True,
)

if __name__ == "__main__":
    sdm = SparkDataManager(additional_conf=SPARK_CONFIGS["STANDARD_FIFTH"])
    run_preprocessing(sdm, cfg)
