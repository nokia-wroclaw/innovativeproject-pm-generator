"""PM dataset schema validation for raw visualizations.

The schema JSON is loaded **lazily** on first use, not at import time, so importing this module never
requires the file to be present (image build, DAG parse, offline tooling). The file is only read when
a function actually needs the schema, then cached.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _schema_candidates() -> list[Path]:
    """Return candidate filesystem paths to search for pm_schema_columns.json."""
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
    """Find the first readable pm_schema_columns.json from the candidate list."""
    for path in _schema_candidates():
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in _schema_candidates())
    raise FileNotFoundError(
        "PM schema file not found. Mount repo shared/ (e.g. /opt/airflow/shared) "
        f"or set GENPM_SCHEMA_PATH. Tried: {tried}"
    )


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    """Load and cache the JSON schema file."""
    with _schema_path().open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def required_columns() -> tuple[str, ...]:
    """Return the tuple of required PM column names."""
    return tuple(_load_schema()["required_columns"])


@lru_cache(maxsize=1)
def column_aliases() -> dict[str, tuple[str, ...]]:
    """Return canonical → alias tuple mapping from the schema."""
    return {
        canonical: tuple(aliases)
        for canonical, aliases in _load_schema().get("column_aliases", {}).items()
    }


@lru_cache(maxsize=1)
def derived_columns() -> dict[str, str]:
    """Return canonical → source column mapping for derivable columns."""
    return dict(_load_schema().get("derived_columns", {}))


# PEP 562: keep PM_REQUIRED_COLUMNS / PM_COLUMN_ALIASES / PM_DERIVED_COLUMNS importable for external
# callers without reading the schema file at import time — resolved (and cached) on first access.
_LAZY_ATTRS = {
    "PM_REQUIRED_COLUMNS": required_columns,
    "PM_COLUMN_ALIASES": column_aliases,
    "PM_DERIVED_COLUMNS": derived_columns,
}


def __getattr__(name: str):
    """Resolve PM_REQUIRED_COLUMNS / PM_COLUMN_ALIASES / PM_DERIVED_COLUMNS lazily on first access."""
    loader = _LAZY_ATTRS.get(name)
    if loader is not None:
        return loader()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _column_satisfied(canonical: str, present: set[str]) -> bool:
    """True if the canonical column or any of its aliases or derivable sources is present."""
    if canonical in present:
        return True
    for alias in column_aliases().get(canonical, ()):
        if alias in present:
            return True
    source = derived_columns().get(canonical)
    if source is not None and _column_satisfied(source, present):
        return True
    return False


def validate_pm_schema(df: DataFrame) -> tuple[bool, list[str]]:
    """Check that all required PM columns (or acceptable aliases) are present in df."""
    present = set(df.columns)
    missing = [col for col in required_columns() if not _column_satisfied(col, present)]
    return len(missing) == 0, missing


def normalize_pm_dataframe(df: DataFrame) -> DataFrame:
    """Rename known aliases and derive missing canonical columns (e.g. start_date)."""
    from pyspark.sql import functions as spark_f

    result = df
    for canonical, aliases in column_aliases().items():
        if canonical in result.columns:
            continue
        for alias in aliases:
            if alias in result.columns:
                result = result.withColumnRenamed(alias, canonical)
                break

    for canonical, source in derived_columns().items():
        if canonical in result.columns:
            continue
        if source in result.columns:
            result = result.withColumn(canonical, spark_f.to_date(source))

    return result


def unsupported_schema_payload(missing: list[str]) -> dict:
    """Build the error payload returned when the input schema doesn't match PM requirements."""
    return {
        "status": "unsupported_schema",
        "missing_columns": missing,
        "required_columns": list(required_columns()),
        "message": (
            "Dataset does not match PM schema required for visualizations. "
            f"Missing columns: {', '.join(missing)}."
        ),
    }
