import argparse
import json
import os
from pathlib import Path

from genpm.modelling.configs import GenerateConfig, TrainConfig
from genpm.modelling.generate import run_generation
from genpm.modelling.train import run_training


def _add_generate_args(p: argparse.ArgumentParser) -> None:
    """Register the ``generate`` subcommand's CLI arguments on ``p``.

    The architecture flags (``--seq-len``, ``--hidden-dim``, ...) must match the
    trained checkpoint; they exist for overrides — the loader otherwise reads the
    saved values.
    """
    p.add_argument("--run-dir-path", required=True)
    p.add_argument("--weights-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--cell-id", required=True)
    p.add_argument("--anchor-date", required=True)
    p.add_argument("--n-weeks", type=int, required=True)
    p.add_argument("--holiday", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cell-configs",
        nargs="+",
        default=None,
        help="Explicit config values for conditioning. If omitted, looked up by --cell-id.",
    )
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--n-dim", type=int, default=235)
    p.add_argument("--global-latent-dim", type=int, default=64)
    p.add_argument("--local-latent-dim", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    # NOTE: store_true + default=True makes this a no-op flag (always True). Kept as-is
    # to avoid changing behaviour; use a store_false/--no-attention flag to toggle off.
    p.add_argument("--use-attention", action="store_true", default=True)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--free-bits-global", type=float, default=0.002)
    p.add_argument("--free-bits-local", type=float, default=0.0)
    p.add_argument("--output-activation", type=str, default="sigmoid")
    # kpi_list is loaded from kpi_columns.npy unless overridden
    p.add_argument(
        "--kpi-columns-path",
        type=str,
        default=None,
        help="Path to kpi_columns.npy. Defaults to <run-dir-path>/kpi_columns.npy.",
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    """Register the ``train`` subcommand's CLI arguments on ``p`` (cVAE-LSTM)."""
    p.add_argument("--training-data-path", required=True)
    p.add_argument("--run-dir-path", required=True)
    p.add_argument("--weights-path", required=True)
    p.add_argument("--global-latent-dim", type=int, default=64)
    p.add_argument("--local-latent-dim", type=int, default=0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    # NOTE: no-op flag (always True) — see the matching note in _add_generate_args.
    p.add_argument("--use-attention", action="store_true", default=True)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--free-bits-global", type=float, default=0.002)
    p.add_argument("--free-bits-local", type=float, default=0.0)
    p.add_argument("--output-activation", type=str, default="sigmoid")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-beta", type=float, default=2e-4)
    p.add_argument("--no-cyclical-kl", action="store_true", default=False)
    p.add_argument("--cycle-epochs", type=int, default=40)
    p.add_argument("--n-cycles", type=int, default=6)
    p.add_argument("--cycle-ratio", type=float, default=0.5)
    p.add_argument("--anneal-epochs", type=int, default=150)
    p.add_argument("--lr-patience", type=int, default=20)
    p.add_argument("--early-stop-patience", type=int, default=60)
    p.add_argument("--no-collapse-monitor", action="store_true", default=False)


def main(argv=None):
    """CLI entrypoint: parse args and dispatch to ``run_generation`` or ``run_training``.

    For ``generate``, ``kpi_list`` is loaded from ``kpi_columns.npy`` in the run dir
    (or ``--kpi-columns-path``) so the output columns match the trained checkpoint.
    """
    parser = argparse.ArgumentParser(description="PM synthetic data generation/training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_generate_args(subparsers.add_parser("generate", help="Generate synthetic KPI windows"))
    _add_train_args(subparsers.add_parser("train", help="Train the CVAE-LSTM model"))

    args = parser.parse_args(argv)

    if args.command == "generate":
        if args.conf_json:
            from genpm.modelling.generate_s3 import run_generation_from_conf

            conf = json.loads(args.conf_json)
            bucket = os.getenv("S3_BUCKET", "datasets")
            run_generation_from_conf(conf, bucket=bucket)
        else:
            import numpy as np

            if not args.run_dir_path or not args.weights_path or not args.output_path:
                raise SystemExit(
                    "error: --run-dir-path, --weights-path, and --output-path are required "
                    "when not using --conf-json"
                )
            if args.anchor_date is None or args.n_weeks is None:
                raise SystemExit(
                    "error: --anchor-date and --n-weeks are required when not using --conf-json"
                )

            kpi_path = args.kpi_columns_path or str(Path(args.run_dir_path) / "kpi_columns.npy")
            kpi_list = np.load(kpi_path, allow_pickle=True).tolist()

            cfg = GenerateConfig(
                run_dir_path=args.run_dir_path,
                weights_path=args.weights_path,
                output_path=args.output_path,
                cell_id=args.cell_id,
                anchor_date=args.anchor_date,
                n_weeks=args.n_weeks,
                kpi_list=kpi_list,
                holiday=args.holiday,
                batch_size=args.batch_size,
                seed=args.seed,
                cell_configs=args.cell_configs,
                seq_len=args.seq_len,
                n_dim=args.n_dim,
                global_latent_dim=args.global_latent_dim,
                local_latent_dim=args.local_latent_dim,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                use_attention=args.use_attention,
                n_heads=args.n_heads,
                free_bits_global=args.free_bits_global,
                free_bits_local=args.free_bits_local,
                output_activation=args.output_activation,
            )
            run_generation(cfg)

    elif args.command == "train":
        cfg = TrainConfig(
            training_data_path=args.training_data_path,
            run_dir_path=args.run_dir_path,
            weights_path=args.weights_path,
            global_latent_dim=args.global_latent_dim,
            local_latent_dim=args.local_latent_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            use_attention=args.use_attention,
            n_heads=args.n_heads,
            beta=args.beta,
            learning_rate=args.learning_rate,
            free_bits_global=args.free_bits_global,
            free_bits_local=args.free_bits_local,
            output_activation=args.output_activation,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_beta=args.target_beta,
            use_cyclical_kl=not args.no_cyclical_kl,
            cycle_epochs=args.cycle_epochs,
            n_cycles=args.n_cycles,
            cycle_ratio=args.cycle_ratio,
            anneal_epochs=args.anneal_epochs,
            lr_patience=args.lr_patience,
            early_stop_patience=args.early_stop_patience,
            collapse_monitor=not args.no_collapse_monitor,
        )
        run_training(cfg)


if __name__ == "__main__":
    main()
