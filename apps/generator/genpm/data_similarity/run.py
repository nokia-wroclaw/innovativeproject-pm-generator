"""Data similarity pipeline: real vs synthetic time series validation."""

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from pyspark.sql import DataFrame

from genpm.data_similarity.configs import DataSimilarityConfig
from genpm.data_similarity.data_similarity_utils import (
    compute_multi_metrics,
    compute_single_metrics,
    create_multi_kpi_figure,
    create_single_kpi_figure,
)
from genpm.utils.logger import get_logger
from genpm.utils.utils import SparkDataManager
from scripts.fake_timeseries import (
    make_multi,
    multi_schema,
    pdf_to_spark_with_schema,
)

logger = get_logger()


def _generate_fake_real(cfg: DataSimilarityConfig) -> DataFrame:
    """
    Generates fake 'real' data via make_multi.

    Uses the fixed multi-KPI schema (ts, kpi_a, kpi_b, kpi_c). Single-KPI
    validation works on each of these columns individually.
    """
    pdf = make_multi(
        start_date=cfg.fake_real_start_date,
        n_days=cfg.fake_real_n_days,
        noise_std=cfg.fake_real_noise_std,
        seed=cfg.fake_real_seed,
    )

    return pdf_to_spark_with_schema(pdf, multi_schema)


def _generate_fake_synth(cfg: DataSimilarityConfig) -> DataFrame:
    """
    Generates fake 'synth' data via make_multi with different noise / seed.
    """
    pdf = make_multi(
        start_date=cfg.fake_synth_start_date,
        n_days=cfg.fake_synth_n_days,
        noise_std=cfg.fake_synth_noise_std,
        seed=cfg.fake_synth_seed,
    )

    return pdf_to_spark_with_schema(pdf, multi_schema)


def _load_data(
    sdm: SparkDataManager,
    cfg: DataSimilarityConfig,
) -> tuple[DataFrame, DataFrame]:
    """
    Loads real and synthetic data as Spark DataFrames.

    Each source independently:
    - cfg.real_data_path / cfg.synth_data_path = None -> generate fake via make_multi
    - otherwise -> read parquet from cfg.real_data_path / cfg.synth_data_path

    Path validation (path present when fake flag not set) is done in __main__.py
    before construction of cfg.
    """
    if cfg.real_data_path is None:
        logger.info("Real: generating fake series via make_multi")
        real_sdf = _generate_fake_real(cfg)
    else:
        logger.info(f"Real: reading parquet from {cfg.real_data_path}")
        real_sdf = sdm.read_parquet(cfg.real_data_path)

    if cfg.synth_data_path is None:
        logger.info("Synth: generating fake series via make_multi")
        synth_sdf = _generate_fake_synth(cfg)
    else:
        logger.info(f"Synth: reading parquet from {cfg.synth_data_path}")
        synth_sdf = sdm.read_parquet(cfg.synth_data_path)

    return real_sdf, synth_sdf


def _figure_to_dict(fig: go.Figure) -> dict[str, Any]:
    """
    Serializes a Plotly figure into a JSON-safe nested dict.
    """
    return json.loads(fig.to_json())


def _summary_from_single_metrics(
    metrics: dict[str, Any],
    fig: go.Figure,
) -> dict[str, Any]:
    """
    Extracts JSON-serializable summary from a full single-KPI metrics dict and the Plotly figure as a nested dict.
    """
    return {
        "n_real_rows": int(len(metrics["real_full_pdf"])),
        "n_synth_rows": int(len(metrics["synth_full_pdf"])),
        "n_real_observed": int(len(metrics["real_values_observed"])),
        "n_synth_observed": int(len(metrics["synth_values_observed"])),
        "wasserstein_1d": float(metrics["wasserstein_1d"]),
        "mmd_rbf": float(metrics["mmd_rbf"]),
        "jensen_shannon": float(metrics["jensen_shannon"]),
        "ls_spectrum_distance": float(metrics["ls_spectrum_distance"]),
        "acf_distance": float(metrics["acf_distance"]),
        "hourly_profile_rmse": float(metrics["hourly_profile_rmse"]),
        "real_missing": metrics["real_missing"],
        "synth_missing": metrics["synth_missing"],
        "figure": _figure_to_dict(fig),
    }


def _summary_from_multi_metrics(
    metrics: dict[str, Any],
    fig: go.Figure,
) -> dict[str, Any]:
    """
    Extracts JSON-serializable summary from a full multi-KPI metrics dict and the Plotly figure.
    """
    return {
        "value_cols": list(metrics["value_cols"]),
        "n_real_complete": int(metrics["real_complete_values"].shape[0]),
        "n_synth_complete": int(metrics["synth_complete_values"].shape[0]),
        "sliced_wasserstein": float(metrics["sliced_wasserstein"]),
        "mmd_multivariate": float(metrics["mmd_multivariate"]),
        "pairwise_corr_distance": float(metrics["pairwise_corr_distance"]),
        "partial_corr_distance": float(metrics["partial_corr_distance"]),
        "pairwise_corr_real": metrics["pairwise_corr_real"].tolist(),
        "pairwise_corr_synth": metrics["pairwise_corr_synth"].tolist(),
        "partial_corr_real": metrics["partial_corr_real"].tolist(),
        "partial_corr_synth": metrics["partial_corr_synth"].tolist(),
        "figure": _figure_to_dict(fig),
    }


def _validate_single_kpi(
    real_sdf: DataFrame,
    synth_sdf: DataFrame,
    kpi: str,
    cfg: DataSimilarityConfig,
) -> dict[str, Any] | None:
    logger.info(f"Single-KPI validation: {kpi!r}")
    """
    Runs the full single-KPI validation pipeline for one KPI column.
    """
    try:
        metrics = compute_single_metrics(
            real_sdf=real_sdf,
            synth_sdf=synth_sdf,
            value_col=kpi,
            ts_col=cfg.ts_col,
            acf_max_lag=cfg.acf_max_lag,
            ls_min_period_h=cfg.ls_min_period_h,
            ls_max_period_h=cfg.ls_max_period_h,
            ls_n_freq=cfg.ls_n_freq,
            kde_n_grid=cfg.kde_n_grid,
        )
    except Exception as exc:
        logger.exception(f"  metric computation failed for {kpi!r}: {exc}")
        return None

    fig = create_single_kpi_figure(metrics)

    return _summary_from_single_metrics(metrics, fig)


def _validate_multi_kpi(
    real_sdf: DataFrame,
    synth_sdf: DataFrame,
    cfg: DataSimilarityConfig,
) -> dict[str, Any] | None:
    """
    Runs the full multi-KPI validation pipeline for selected KPI columns (if less than 2 are selected then skipped).
    """
    if len(cfg.multi_kpi_cols) < 2:
        logger.info(f"Multi-KPI validation skipped: need >= 2 KPIs, got {len(cfg.multi_kpi_cols)}")
        return None

    logger.info(f"Multi-KPI validation: {cfg.multi_kpi_cols}")

    try:
        metrics = compute_multi_metrics(
            real_sdf=real_sdf,
            synth_sdf=synth_sdf,
            value_cols=cfg.multi_kpi_cols,
            ts_col=cfg.ts_col,
            n_projections=cfg.n_projections,
        )
    except Exception as exc:
        logger.exception(f"  multi-KPI metric computation failed: {exc}")
        return None

    fig = create_multi_kpi_figure(metrics)

    return _summary_from_multi_metrics(metrics, fig)


def run_data_similarity(sdm: SparkDataManager, cfg: DataSimilarityConfig) -> dict[str, Any]:
    """
    Main entry point for the data similarity pipeline.

    Returns the full summary dict (also written to summary.json if configured).
    """
    output_root = Path(cfg.output_path_prefix)
    output_root.mkdir(parents=True, exist_ok=True)

    real_sdf, synth_sdf = _load_data(sdm, cfg)
    real_sdf.cache()
    synth_sdf.cache()

    summary: dict[str, Any] = {
        "config": {
            "real_data_path": cfg.real_data_path,
            "synth_data_path": cfg.synth_data_path,
            "ts_col": cfg.ts_col,
            "single_kpi_cols": list(cfg.single_kpi_cols),
            "multi_kpi_cols": list(cfg.multi_kpi_cols),
        },
        "single_kpi": {},
        "multi_kpi": None,
    }

    for kpi in cfg.single_kpi_cols:
        result = _validate_single_kpi(
            real_sdf=real_sdf,
            synth_sdf=synth_sdf,
            kpi=kpi,
            cfg=cfg,
        )
        if result is not None:
            summary["single_kpi"][kpi] = result

    summary["multi_kpi"] = _validate_multi_kpi(
        real_sdf=real_sdf,
        synth_sdf=synth_sdf,
        cfg=cfg,
    )

    real_sdf.unpersist()
    synth_sdf.unpersist()

    if cfg.save_summary_json:
        summary_path = output_root / "summary.json"
        with summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info(f"Wrote summary to {summary_path}")

    logger.info("Data similarity pipeline finished.")
    return summary
