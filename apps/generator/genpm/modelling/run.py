from pathlib import Path

from genpm.modelling.configs import GenerateConfig
from genpm.modelling.generate import generate_windows
from genpm.modelling.model_utils.cvae_utils import load_artifacts
from genpm.utils.logger import get_logger

logger = get_logger()


def run_generation(cfg: GenerateConfig) -> None:
    logger.info(f"Loading artifacts from {cfg.run_dir_path}")
    model, cell_encoder = load_artifacts(
        run_id_path=Path(cfg.run_dir_path),
        weights_path=Path(cfg.weights_path),
        global_latent_dim=cfg.global_latent_dim,
        local_latent_dim=cfg.local_latent_dim,
        cell_embed_dim=cfg.cell_embed_dim,
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
    logger.info(f"Generating {cfg.n_weeks} week(s) for cell '{cfg.cell_id}' from {cfg.anchor_date}")
    windows = generate_windows(
        model=model,
        cell_encoder=cell_encoder,
        cell_id=cfg.cell_id,
        anchor_date=cfg.anchor_date,
        n_weeks=cfg.n_weeks,
        holiday=cfg.holiday,
        seq_len=cfg.seq_len,
        n_dim=cfg.n_dim,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        kpi_list=cfg.kpi_list,
    )

    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{cfg.cell_id.replace('/', '_')}_{cfg.anchor_date}.parquet"
    windows.to_parquet(out_file, index=False)
    logger.info(f"Saved {len(windows)} rows to {out_file}")
