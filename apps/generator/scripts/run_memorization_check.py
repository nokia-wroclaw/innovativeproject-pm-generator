"""Memorization probe for a trained generator: is generated data closer to specific
real windows than real windows normally are to each other?

Usage:  python scripts/run_memorization_check.py
        (from the repo root, no install needed)

For each config present in the training data: generate N_GEN_PER_CONFIG synthetic
windows using the EXACT y (config+holiday+seasonal) of randomly chosen real windows
in that config, then compare nearest-neighbor distances (see
genpm.modelling.validation.nearest_neighbor_check for the metric definition and why
same-cell real windows are excluded from the baseline).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from genpm.modelling.core.artifacts import load_saved_windows, load_trained_model  # noqa: E402
from genpm.modelling.core.generation import _to_numpy  # noqa: E402
from genpm.modelling.validation import nearest_neighbor_check, summarize_nn_check  # noqa: E402
from genpm.utils.consts import SHARED_DIR_PATH  # noqa: E402

RUN_DIR = SHARED_DIR_PATH / "model_runs" / "diffusion_run_3"
WEIGHTS_PATH = RUN_DIR / "models_weights_debug" / "ddpm_3.weights.h5"

N_GEN_PER_CONFIG = 32
MIN_REAL_WINDOWS = 5  # configs with fewer real windows than this are skipped
TOP_K = 5
SEED = 0


def main() -> None:
    """Run the memorization probe per config and print the NN-distance summary.

    Edit ``RUN_DIR``/``WEIGHTS_PATH`` above to point at the checkpoint to probe.
    """
    print(f"Loading model from {WEIGHTS_PATH}")
    model, config_encoder, _cell_config_map = load_trained_model(RUN_DIR, WEIGHTS_PATH)

    print(f"Loading training data from {RUN_DIR}")
    data = load_saved_windows(RUN_DIR)
    X, y, cell_ids = data["X_scaled"], data["y"], data["cell_ids"]

    n_onehot = sum(len(c) for c in config_encoder.categories_)
    config_block = y[:, :n_onehot]
    uniq_configs = np.unique(config_block, axis=0)
    print(f"{len(uniq_configs)} distinct config(s), {len(X)} real windows total\n")

    rng = np.random.default_rng(SEED)

    for cfg_row in uniq_configs:
        mask = np.all(config_block == cfg_row, axis=1)
        X_c, y_c, cell_c = X[mask], y[mask], cell_ids[mask]
        label = "|".join(str(v) for v in config_encoder.inverse_transform(cfg_row[None, :])[0])

        if len(X_c) < MIN_REAL_WINDOWS:
            print(f"[{label}] only {len(X_c)} real windows — skipping\n")
            continue

        n_gen = min(N_GEN_PER_CONFIG, len(X_c))
        probe_idx = rng.choice(len(X_c), size=n_gen, replace=False)
        x_gen, _ = model.generate(y_c[probe_idx])
        x_gen = _to_numpy(x_gen)

        result = nearest_neighbor_check(X_c, cell_c, x_gen)
        summary = summarize_nn_check(result["real_nn"], result["gen_nn"])

        n_cells = len(set(cell_c))
        print(
            f"[{label}]  {len(X_c)} real windows, {n_cells} cells, "
            f"{n_gen} generated probes ({result['n_excluded_real']} real windows "
            f"had no different-cell baseline and were excluded)"
        )
        print(
            f"  real-to-real (diff cell) nearest:  median={summary['real_median']:.6f}  "
            f"p05={summary['real_p05']:.6f}"
        )
        print(
            f"  gen-to-real           nearest:  median={summary['gen_median']:.6f}  "
            f"p05={summary['gen_p05']:.6f}"
        )
        print(
            f"  ratio (gen median / real median) = {summary['ratio_median']:.3f}   "
            f"frac(gen below real p05) = {summary['frac_gen_below_real_p05']:.3f}"
        )

        # Drill into the closest individual matches: is the low ratio above driven by
        # a few near-exact copies (memorization), or is every gen sample uniformly a
        # bit closer than usual (mild under-dispersion, a different and lower-stakes
        # issue)? The percentile tells us how this match compares to NATURAL real-vs-
        # real variation for this config — if it's inside the normal range, it's not
        # evidence of copying; if it's below anything real windows ever do, it is.
        order = np.argsort(result["gen_nn"])[:TOP_K]
        print(f"  top-{TOP_K} closest generated-to-real matches:")
        for rank, gi in enumerate(order, start=1):
            dist = result["gen_nn"][gi]
            match_cell = result["gen_nn_match_cell"][gi]
            pct_within_real = float(np.mean(result["real_nn"] <= dist)) * 100
            flag = (
                "  <-- BELOW ANY real-to-real distance ever observed in this config"
                if dist < result["real_nn"].min()
                else ""
            )
            print(
                f"    #{rank}: dist={dist:.6f}  matched cell={match_cell}  "
                f"matched real idx={result['gen_nn_match_idx'][gi]}  "
                f"(real-to-real percentile: {pct_within_real:.2f}%){flag}"
            )
        print()


if __name__ == "__main__":
    main()
