from __future__ import annotations


def s3a_path(bucket: str, key_or_path: str) -> str:
    """Return an s3a:// URI; pass-through if already an s3a URI, prefix with bucket otherwise."""
    value = (key_or_path or "").strip()
    if not value:
        return ""
    if value.startswith("s3a://"):
        return value
    return f"s3a://{bucket}/{value.lstrip('/')}"
