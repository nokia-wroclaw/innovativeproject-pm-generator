from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import numpy as np
from botocore.client import Config

from genpm.modelling.core.artifacts import load_trained_model
from genpm.modelling.core.generation import generate_windows
from genpm.utils.logger import get_logger

logger = get_logger()


def _s3_client():
    """Build an S3/MinIO boto3 client from environment variables."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _download(client, bucket: str, key: str, dest: Path) -> None:
    """Download a single S3 object to a local path, creating parent directories as needed."""
    logger.info(f"Downloading s3://{bucket}/{key} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))


def run_generation_from_conf(conf: dict[str, Any], *, bucket: str) -> None:
    """Download model artifacts from S3, generate synthetic data, upload results to S3.

    ``conf`` is the finalized ``dag_run.conf`` dict produced by the DAG ``prepare_conf`` task.
    Top-level keys ``model_path``, ``encoder_path``, ``config_path`` carry the S3 keys of the
    trained model artifacts. ``dag_args`` carries generation parameters.
    """
    dag_args: dict[str, Any] = conf.get("dag_args") or {}

    model_key: str = conf.get("model_path", "")
    encoder_key: str = conf.get("encoder_path", "")
    config_key: str = conf.get("config_path", "")
    if not model_key or not encoder_key or not config_key:
        raise ValueError(
            "conf must contain non-empty 'model_path', 'encoder_path', 'config_path' keys"
        )

    # Derive the run-dir S3 prefix from the encoder key (sibling artifacts live there too).
    run_dir_prefix = str(PurePosixPath(encoder_key).parent)

    cell_id: str | None = dag_args.get("cell_id") or None
    anchor_date: str = str(dag_args.get("anchor_date", ""))
    n_weeks: int = int(dag_args.get("n_weeks", 4))
    holiday: int = int(dag_args.get("holiday", 0))
    batch_size: int = int(dag_args.get("batch_size", 64))
    seed: int = int(dag_args.get("seed", 42))
    kpi_list: list[str] = list(dag_args.get("kpi_list") or [])
    genpm_run_id: str = conf.get("genpm_run_id") or "run"
    output_prefix: str = (dag_args.get("output_path_prefix") or f"generated/{genpm_run_id}").rstrip(
        "/"
    )

    client = _s3_client()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        weights_local = tmp / PurePosixPath(model_key).name
        _download(client, bucket, model_key, weights_local)

        _download(client, bucket, encoder_key, tmp / "config_encoder.pkl")
        _download(client, bucket, config_key, tmp / "cell_config_map.pkl")

        # arch_params.json and kpi_columns.npy live in the same prefix as encoder/config.
        for artifact in ("arch_params.json", "kpi_columns.npy"):
            key = f"{run_dir_prefix}/{artifact}"
            try:
                _download(client, bucket, key, tmp / artifact)
            except Exception:
                logger.warning(f"Optional artifact not found: s3://{bucket}/{key}")

        # Always load the full KPI column list from kpi_columns.npy — this matches
        # what the model was trained on and is used to name the generated DataFrame columns.
        # kpi_list from dag_args is a user-requested *filter* (subset to keep in the output).
        kpi_filter: list[str] = kpi_list  # user's selection (may be empty = keep all)
        kpi_npy = tmp / "kpi_columns.npy"
        if kpi_npy.exists():
            kpi_columns_full = np.load(str(kpi_npy), allow_pickle=True).tolist()
            logger.info(f"Loaded {len(kpi_columns_full)} KPIs from kpi_columns.npy")
        elif kpi_filter:
            # No kpi_columns.npy but user provided a list — assume it IS the full list.
            kpi_columns_full = kpi_filter
            kpi_filter = []
        else:
            raise ValueError(
                "kpi_list not provided in dag_args and kpi_columns.npy not found on S3 "
                f"(looked under s3://{bucket}/{run_dir_prefix}/kpi_columns.npy)"
            )

        model, config_encoder, cell_config_map = load_trained_model(
            run_id_path=tmp,
            weights_path=weights_local,
        )

        # Determine which cells to generate for.
        # When cell_id is provided → single cell; when omitted → all cells in config map.
        # cell_config_map structure: {"config_cols": [...], "map": {cell_id: configs}}
        if cell_id:
            cell_ids_to_generate = [cell_id]
        else:
            cell_ids_to_generate = sorted(cell_config_map["map"].keys())
            logger.info(
                f"No cell_id specified — generating for all {len(cell_ids_to_generate)} cells"
            )

        generated_outputs: list[dict[str, Any]] = []

        for cid in cell_ids_to_generate:
            logger.info(
                f"Generating {n_weeks} week(s) for cell_id={cid!r} "
                f"from {anchor_date}, holiday={holiday}, seed={seed}"
            )
            windows = generate_windows(
                model=model,
                config_encoder=config_encoder,
                cell_config_map=cell_config_map,
                cell_id=cid,
                anchor_date=anchor_date,
                n_weeks=n_weeks,
                holiday=holiday,
                batch_size=batch_size,
                seed=seed,
                kpi_list=kpi_columns_full,
            )

            # Filter to user-requested KPIs (if a subset was specified)
            if kpi_filter:
                missing_kpis = [k for k in kpi_filter if k not in windows.columns]
                if missing_kpis:
                    logger.warning(f"  KPIs not found in model output, skipping: {missing_kpis}")
                keep = [k for k in kpi_filter if k in windows.columns]
                meta_cols = [c for c in windows.columns if c not in kpi_columns_full]
                windows = windows[meta_cols + keep]

            target_label = cid.replace("/", "_")
            output_key = f"{output_prefix}/{target_label}_{anchor_date}.parquet"
            buf = io.BytesIO()
            windows.to_parquet(buf, index=False, coerce_timestamps="us")
            buf.seek(0)
            logger.info(f"Uploading {len(windows)} rows → s3://{bucket}/{output_key}")
            client.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=buf.read(),
                ContentType="application/octet-stream",
            )
            generated_outputs.append(
                {"cell_id": cid, "output_key": output_key, "n_rows": len(windows)}
            )

        meta = {
            "genpm_run_id": genpm_run_id,
            "cell_id": cell_id or None,
            "cells_generated": [o["cell_id"] for o in generated_outputs],
            "anchor_date": anchor_date,
            "n_weeks": n_weeks,
            "holiday": holiday,
            "seed": seed,
            "total_cells": len(generated_outputs),
            "kpi_count": len(kpi_list),
            "kpis": kpi_list,
            "outputs": generated_outputs,
        }
        meta_key = f"{output_prefix}/generation_metadata.json"
        client.put_object(
            Bucket=bucket,
            Key=meta_key,
            Body=json.dumps(meta, default=str, indent=2).encode(),
            ContentType="application/json",
        )
        logger.info(f"Metadata written to s3://{bucket}/{meta_key}")
        logger.info(f"Generation complete — {len(generated_outputs)} cell(s) generated.")
