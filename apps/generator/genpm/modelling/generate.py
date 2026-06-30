"""Generation entrypoint: reload a trained model and write synthetic windows.

Thin orchestration over :func:`load_trained_model` and :func:`generate_windows`; the
runnable CLI wrapper lives in ``__main__.py``.
"""

from pathlib import Path

import polars as pl

from genpm.modelling.configs import GenerateConfig
from genpm.modelling.core.artifacts import load_trained_model
from genpm.modelling.core.generation import _config_label, generate_windows
from genpm.utils.logger import get_logger

logger = get_logger()


def run_generation(cfg: GenerateConfig) -> None:
    """Reload the trained model and generate synthetic KPI windows to parquet.

    Loads the model/encoder/config-map from ``cfg.run_dir_path``, generates
    ``cfg.n_weeks`` weeks conditioned on ``cfg.cell_id`` (or ``cfg.cell_configs``), and
    writes one parquet file to ``cfg.output_path``. The output is labelled by the
    cell_id when given, otherwise by a label derived from the explicit config values.

    Args:
        cfg: Populated :class:`GenerateConfig`.
    """
    logger.info(f"Loading artifacts from {cfg.run_dir_path}")
    model, config_encoder, cell_config_map = load_trained_model(
        run_id_path=Path(cfg.run_dir_path),
        weights_path=Path(cfg.weights_path),
        global_latent_dim=cfg.global_latent_dim,
        local_latent_dim=cfg.local_latent_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        use_attention=cfg.use_attention,
        n_heads=cfg.n_heads,
        free_bits_global=cfg.free_bits_global,
        free_bits_local=cfg.free_bits_local,
        output_activation=cfg.output_activation,
        seq_len=cfg.seq_len,
        feat_dim=cfg.n_dim,
    )
    target = cfg.cell_id if cfg.cell_id is not None else _config_label(cfg.cell_configs)
    logger.info(f"Generating {cfg.n_weeks} week(s) for '{target}' from {cfg.anchor_date}")
    windows = generate_windows(
        model=model,
        config_encoder=config_encoder,
        cell_config_map=cell_config_map,
        cell_id=cfg.cell_id,
        anchor_date=cfg.anchor_date,
        n_weeks=cfg.n_weeks,
        holiday=cfg.holiday,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        kpi_list=cfg.kpi_list,
        cell_configs=cfg.cell_configs,
    )

    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{target.replace('/', '_')}_{cfg.anchor_date}.parquet"
    pl.from_pandas(windows).write_parquet(out_file)
    logger.info(f"Saved {len(windows)} rows to {out_file}")
