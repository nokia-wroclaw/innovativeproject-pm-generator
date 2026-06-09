#!/usr/bin/env python3
"""
Generate a sizable PM-shaped Parquet file and upload it to S3 (MinIO).

Required columns (visualization schema):
  kpi_id, bts_id, start_time, kpi_value, start_date, distname

Usage (from repo root, with MinIO reachable):

  cd apps/generator && uv run python ../../scripts/upload_mock_pm_kpi.py

  # or with explicit endpoint (host machine → localhost:9000):
  uv run python scripts/upload_mock_pm_kpi.py \\
    --s3-endpoint http://localhost:9000 \\
    --s3-key mock_pm_kpi.parquet

Then register in the app (Storage → register) with the printed s3_key, or use the API.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.client import Config

PM_COLUMNS = (
    "kpi_id",
    "bts_id",
    "start_time",
    "kpi_value",
    "start_date",
    "distname",
)

# Defaults tuned for a "spory" dataset (heatmap/catalog stress-test scale).
DEFAULT_NUM_BTS = 120
DEFAULT_NUM_KPIS = 72
DEFAULT_DAYS = 14
DEFAULT_FREQ = "15min"
DEFAULT_COVERAGE = 0.92


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def generate_pm_dataframe(
    *,
    num_bts: int,
    num_kpis: int,
    days: int,
    freq: str,
    coverage: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    bts_ids = np.array([f"BTS_{i:05d}" for i in range(1, num_bts + 1)], dtype=object)
    kpi_ids = np.array([f"KPI_{i:05d}" for i in range(1, num_kpis + 1)], dtype=object)

    periods = int(pd.Timedelta(days=days) / pd.Timedelta(freq))
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    times = pd.date_range(start, periods=periods, freq=freq)

    n_slots = num_bts * num_kpis * len(times)
    print(
        f"Generating up to {n_slots:,} rows "
        f"({num_bts} BTS × {num_kpis} KPIs × {len(times)} intervals)…",
        flush=True,
    )

    bts_idx = np.repeat(np.arange(num_bts), num_kpis * len(times))
    kpi_idx = np.tile(np.repeat(np.arange(num_kpis), len(times)), num_bts)
    time_idx = np.tile(np.arange(len(times)), num_bts * num_kpis)

    keep_mask = rng.random(n_slots) < coverage
    bts_idx = bts_idx[keep_mask]
    kpi_idx = kpi_idx[keep_mask]
    time_idx = time_idx[keep_mask]

    n_rows = len(bts_idx)
    print(f"Kept {n_rows:,} rows ({100 * n_rows / n_slots:.1f}% coverage)", flush=True)

    bts_col = bts_ids[bts_idx]
    kpi_col = kpi_ids[kpi_idx]
    time_col = times[time_idx]

    base = rng.normal(50.0, 12.0, size=n_rows)
    hourly = 3.0 * np.sin(2 * np.pi * time_idx / max(len(times) / days, 1))
    kpi_shift = (kpi_idx % 7) * 0.8
    values = base + hourly + kpi_shift + rng.normal(0, 2.5, size=n_rows)
    null_mask = rng.random(n_rows) < 0.03
    values = values.astype(np.float64)
    values[null_mask] = np.nan

    start_date = pd.DatetimeIndex(time_col).normalize()

    distname = np.array(
        [
            f"SubNetwork=LTE,MeContext={bts},Kpi={kpi}"
            for bts, kpi in zip(bts_col, kpi_col, strict=True)
        ],
        dtype=object,
    )

    return pd.DataFrame(
        {
            "kpi_id": kpi_col,
            "bts_id": bts_col,
            "start_time": time_col,
            "kpi_value": values,
            "start_date": start_date,
            "distname": distname,
        }
    )


def write_parquet(df: pd.DataFrame, path: Path) -> int:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    return path.stat().st_size


def upload_to_s3(
    local_path: Path,
    *,
    bucket: str,
    key: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: str,
) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint.rstrip("/"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    with local_path.open("rb") as body:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/octet-stream",
        )

    head = client.head_object(Bucket=bucket, Key=key)
    print(f"Uploaded s3://{bucket}/{key} ({head['ContentLength']:,} bytes)")


def parse_args() -> argparse.Namespace:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate mock_pm_kpi.parquet (PM schema) and upload to S3."
    )
    parser.add_argument(
        "--s3-endpoint",
        default=os.getenv("S3_URL", "http://localhost:9000").rstrip("/"),
        help="S3/MinIO endpoint (default: S3_URL or http://localhost:9000)",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("S3_BUCKET", "datasets"),
    )
    parser.add_argument(
        "--s3-key",
        default="mock_pm_kpi.parquet",
        help="Object key in the bucket (default: mock_pm_kpi.parquet)",
    )
    parser.add_argument(
        "--access-key", default=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    )
    parser.add_argument(
        "--secret-key", default=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    )
    parser.add_argument(
        "--region", default=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    parser.add_argument("--num-bts", type=int, default=DEFAULT_NUM_BTS)
    parser.add_argument("--num-kpis", type=int, default=DEFAULT_NUM_KPIS)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument(
        "--freq", default=DEFAULT_FREQ, help="pandas offset alias, e.g. 15min, 1h"
    )
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-local",
        type=Path,
        default=None,
        help="Also save parquet to this path (skip temp delete)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = generate_pm_dataframe(
        num_bts=args.num_bts,
        num_kpis=args.num_kpis,
        days=args.days,
        freq=args.freq,
        coverage=args.coverage,
        seed=args.seed,
    )

    print("Column dtypes:")
    for col in PM_COLUMNS:
        print(f"  {col}: {df[col].dtype}")

    if args.keep_local:
        local_path = args.keep_local.expanduser().resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        size = write_parquet(df, local_path)
        print(f"Wrote {local_path} ({size:,} bytes)")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        local_path = Path(tmp.name)
        tmp.close()
        size = write_parquet(df, local_path)
        print(f"Wrote temp parquet ({size:,} bytes)")

    try:
        upload_to_s3(
            local_path,
            bucket=args.bucket,
            key=args.s3_key,
            endpoint=args.s3_endpoint,
            access_key=args.access_key,
            secret_key=args.secret_key,
            region=args.region,
        )
    finally:
        if not args.keep_local and local_path.exists():
            local_path.unlink()

    print()
    print("Done. Register in GenPM (RAW, COMPLETED) with:")
    print(f"  s3_key: {args.s3_key}")
    print("  file_name: mock_pm_kpi.parquet")
    print()
    print("Example API (admin token):")
    print(
        '  curl -X POST "$VITE_API_BASE_URL/api/v1/datasets/register" \\',
    )
    print('    -H "Authorization: Bearer $TOKEN" \\')
    print('    -H "Content-Type: application/json" \\')
    print(
        f'    -d \'{{"s3_key": "{args.s3_key}", "file_name": "mock_pm_kpi.parquet"}}\''
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
