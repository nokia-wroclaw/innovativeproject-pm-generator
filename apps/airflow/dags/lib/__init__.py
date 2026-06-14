"""GenPM Airflow DAG helpers.

Ensure the genpm package (mounted/copied at GENPM_GENERATOR_ROOT) is importable by the Airflow
scheduler / dag-processor / worker python at parse time. The Airflow image also sets PYTHONPATH for
this, but inserting here keeps local DAG validation and tests working too.
"""

from __future__ import annotations

import os
import sys

_GENPM_ROOT = os.environ.get("GENPM_GENERATOR_ROOT", "/opt/airflow/generator")
if os.path.isdir(os.path.join(_GENPM_ROOT, "genpm")) and _GENPM_ROOT not in sys.path:
    sys.path.insert(0, _GENPM_ROOT)
