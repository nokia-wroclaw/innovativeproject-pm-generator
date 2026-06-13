"""genpm.modelling — runnable as `python -m genpm.modelling`."""

import argparse

from genpm.modelling.configs import GenerateConfig
from genpm.modelling.run import run_generation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generation Pipeline for PM Data Synthetic Data Generation",
    )
    # Artifact paths
    parser.add_argument("--run-dir-path", required=True, help="Path to run directory (artifacts)")
    parser.add_argument("--weights-path", required=True, help="Path to model weights file")
    parser.add_argument("--output-path", required=True, help="Directory to write output parquet")
    # Generation parameters
    parser.add_argument("--cell-id", required=True, help="Cell identifier to generate for")
    parser.add_argument(
        "--anchor-date", required=True, help="Start date for generation (YYYY-MM-DD)"
    )
    parser.add_argument("--n-weeks", type=int, required=True, help="Number of weeks to generate")
    parser.add_argument("--holiday", type=int, default=0, help="Holiday flag (0 or 1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    # Model architecture
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--n-dim", type=int, default=235)
    parser.add_argument("--global-latent-dim", type=int, default=64)
    parser.add_argument("--local-latent-dim", type=int, default=0)
    parser.add_argument("--cell-embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--use-attention", action="store_true", default=True)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--free-bits-global", type=float, default=0.002)
    parser.add_argument("--free-bits-local", type=float, default=0.0)
    parser.add_argument("--output-activation", type=str, default="sigmoid")

    args = parser.parse_args(argv)

    cfg = GenerateConfig(
        run_dir_path=args.run_dir_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        cell_id=args.cell_id,
        anchor_date=args.anchor_date,
        n_weeks=args.n_weeks,
        holiday=args.holiday,
        batch_size=args.batch_size,
        seed=args.seed,
        seq_len=args.seq_len,
        n_dim=args.n_dim,
        global_latent_dim=args.global_latent_dim,
        local_latent_dim=args.local_latent_dim,
        cell_embed_dim=args.cell_embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        use_attention=args.use_attention,
        n_heads=args.n_heads,
        free_bits_global=args.free_bits_global,
        free_bits_local=args.free_bits_local,
        output_activation=args.output_activation,
    )

    run_generation(cfg)


if __name__ == "__main__":
    main()
