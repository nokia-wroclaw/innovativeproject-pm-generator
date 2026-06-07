"""PM schema validation for RAW dataset visualizations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _schema_candidates() -> list[Path]:
    candidates: list[Path] = []
    if custom := os.getenv("GENPM_SCHEMA_PATH"):
        candidates.append(Path(custom))
    candidates.extend(
        (
            Path("/app/shared/pm_schema_columns.json"),
            Path("/opt/airflow/shared/pm_schema_columns.json"),
        )
    )
    here = Path(__file__).resolve()
    for depth in (5, 4):
        try:
            candidates.append(here.parents[depth] / "shared" / "pm_schema_columns.json")
        except IndexError:
            pass
    return candidates


def _schema_path() -> Path:
    for path in _schema_candidates():
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in _schema_candidates())
    raise FileNotFoundError(
        "PM schema file not found. Mount repo shared/ or set GENPM_SCHEMA_PATH. " f"Tried: {tried}"
    )


_SCHEMA_PATH = _schema_path()
with _SCHEMA_PATH.open(encoding="utf-8") as f:
    PM_REQUIRED_COLUMNS: tuple[str, ...] = tuple(json.load(f)["required_columns"])


def validate_pm_columns(columns: list[str]) -> tuple[bool, list[str]]:
    present = {name.strip() for name in columns if name}
    missing = [col for col in PM_REQUIRED_COLUMNS if col not in present]
    return len(missing) == 0, missing


def unsupported_schema_payload(
    missing: list[str],
    *,
    present_columns: list[str] | None = None,
) -> dict[str, Any]:
    present = present_columns or []
    return {
        "status": "unsupported_schema",
        "missing_columns": missing,
        "required_columns": list(PM_REQUIRED_COLUMNS),
        "present_columns": present,
        "message": (
            "This dataset is not compatible with PM visualizations. "
            f"Missing required columns: {', '.join(missing)}. "
            f"Found columns: {', '.join(present) if present else '(none)'}."
        ),
    }
