#!/usr/bin/env python3
"""Verify PM schema columns stay in sync across backend, generator, and frontend."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_JSON = REPO_ROOT / "shared" / "pm_schema_columns.json"
EXPECTED = tuple(
    json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))["required_columns"]
)


def _load_backend() -> tuple[str, ...]:
    sys.path.insert(0, str(REPO_ROOT / "apps" / "backend"))
    from app.services.s3.pm_schema import PM_REQUIRED_COLUMNS

    return PM_REQUIRED_COLUMNS


def _load_generator() -> tuple[str, ...]:
    sys.path.insert(0, str(REPO_ROOT / "apps" / "generator"))
    from genpm.raw_vis.pm_schema import PM_REQUIRED_COLUMNS

    return PM_REQUIRED_COLUMNS


def _load_frontend() -> tuple[str, ...]:
    text = (
        REPO_ROOT / "apps" / "frontend" / "src" / "features" / "storage" / "pmSchema.js"
    ).read_text(encoding="utf-8")
    match = re.search(r"import pmSchemaColumns from ['\"](.+?)['\"]", text)
    if not match:
        raise RuntimeError("Could not find pmSchemaColumns import in pmSchema.js")
    import_spec = match.group(1)
    if import_spec in {"@genpm/pm-schema", "../../../pm_schema_columns.json"}:
        json_path = REPO_ROOT / "apps" / "frontend" / "pm_schema_columns.json"
    else:
        rel = import_spec
        json_path = (
            REPO_ROOT / "apps" / "frontend" / "src" / "features" / "storage" / rel
        ).resolve()
    columns = json.loads(json_path.read_text(encoding="utf-8"))["required_columns"]
    return tuple(columns)


def main() -> int:
    frontend_json = tuple(
        json.loads(
            (REPO_ROOT / "apps" / "frontend" / "pm_schema_columns.json").read_text(
                encoding="utf-8"
            )
        )["required_columns"]
    )
    checks = {
        "shared/pm_schema_columns.json": EXPECTED,
        "frontend pm_schema_columns.json": frontend_json,
        "backend pm_schema.py": _load_backend(),
        "generator pm_schema.py": _load_generator(),
        "frontend pmSchema.js": _load_frontend(),
    }
    failed = False
    for name, columns in checks.items():
        if columns != EXPECTED:
            print(f"FAIL {name}: {columns!r} != {EXPECTED!r}", file=sys.stderr)
            failed = True
        else:
            print(f"OK   {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
