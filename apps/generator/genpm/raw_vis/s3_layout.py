"""S3 key layout for visualization artifacts (collocated with dataset upload prefix)."""

from __future__ import annotations

from pathlib import PurePosixPath


def dataset_visualization_prefix(dataset_s3_key: str) -> str:
    """Parent prefix of the dataset object — same tree the user chose at upload."""
    key = (dataset_s3_key or "").strip().lstrip("/")
    if not key:
        raise ValueError("dataset s3_key is empty")
    parent = PurePosixPath(key).parent.as_posix()
    if parent and parent != ".":
        return parent
    # File at bucket root (e.g. mock_pm_kpi.parquet) — namespace by stem, not ".".
    return PurePosixPath(key).stem


def visualization_artifact_key(dataset_s3_key: str, filename: str) -> str:
    base = dataset_visualization_prefix(dataset_s3_key).strip("/")
    return f"{base}/{filename.lstrip('/')}"
