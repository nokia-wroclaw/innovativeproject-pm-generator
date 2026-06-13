"""PM dataset schema validation for raw visualizations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


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
    for depth in (4, 5):
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
        "PM schema file not found. Mount repo shared/ (e.g. /opt/airflow/shared) "
        f"or set GENPM_SCHEMA_PATH. Tried: {tried}"
    )


_SCHEMA_PATH = _schema_path()
with _SCHEMA_PATH.open(encoding="utf-8") as f:
    _SCHEMA = json.load(f)

PM_REQUIRED_COLUMNS: tuple[str, ...] = tuple(_SCHEMA["required_columns"])
PM_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    canonical: tuple(aliases) for canonical, aliases in _SCHEMA.get("column_aliases", {}).items()
}
PM_DERIVED_COLUMNS: dict[str, str] = dict(_SCHEMA.get("derived_columns", {}))


def _column_satisfied(canonical: str, present: set[str]) -> bool:
    if canonical in present:
        return True
    for alias in PM_COLUMN_ALIASES.get(canonical, ()):
        if alias in present:
            return True
    source = PM_DERIVED_COLUMNS.get(canonical)
    if source is not None and _column_satisfied(source, present):
        return True
    return False


def validate_pm_schema(df: DataFrame) -> tuple[bool, list[str]]:
    present = set(df.columns)
    missing = [col for col in PM_REQUIRED_COLUMNS if not _column_satisfied(col, present)]
    return len(missing) == 0, missing


def normalize_pm_dataframe(df: DataFrame) -> DataFrame:
    """Rename known aliases and derive missing canonical columns (e.g. start_date)."""
    from pyspark.sql import functions as spark_f

    result = df
    for canonical, aliases in PM_COLUMN_ALIASES.items():
        if canonical in result.columns:
            continue
        for alias in aliases:
            if alias in result.columns:
                result = result.withColumnRenamed(alias, canonical)
                break

    for canonical, source in PM_DERIVED_COLUMNS.items():
        if canonical in result.columns:
            continue
        if source in result.columns:
            result = result.withColumn(canonical, spark_f.to_date(source))

    return result


def unsupported_schema_payload(missing: list[str]) -> dict:
    return {
        "status": "unsupported_schema",
        "missing_columns": missing,
        "required_columns": list(PM_REQUIRED_COLUMNS),
        "message": (
            "Dataset does not match PM schema required for visualizations. "
            f"Missing columns: {', '.join(missing)}."
        ),
    }
