"""Pure generation logic — no I/O, no hardcoded paths."""

import numpy as np
import pandas as pd

from genpm.modelling.model_utils.cvae_utils import _to_numpy, seasonal_features


def _run_batched_generation(model, y_windows: np.ndarray, batch_size: int) -> np.ndarray:
    decoded = []
    for start in range(0, len(y_windows), batch_size):
        yb = y_windows[start : start + batch_size]
        x_syn, _ = model.generate(yb)
        decoded.append(_to_numpy(x_syn))
    return np.concatenate(decoded)


def generate_windows(
    model,
    cell_encoder,
    cell_id: str,
    anchor_date: str,
    n_weeks: int,
    holiday: int,
    seq_len: int,
    n_dim: int,
    batch_size: int,
    seed: int,
    kpi_list: list,
) -> pd.DataFrame:
    cell_idx = cell_encoder.transform([cell_id])[0]

    anchors = []
    y_windows = []
    for week in range(n_weeks):
        anchor = pd.Timestamp(anchor_date) + pd.Timedelta(weeks=week)
        seasonal = seasonal_features(anchor)
        y_windows.append([cell_idx, holiday, *seasonal])
        anchors.append(anchor)

    y_windows = np.array(y_windows, dtype=np.float32)
    anchors_arr = np.array(anchors)

    kpi_array = _run_batched_generation(model, y_windows, batch_size=batch_size)
    kpi_flat = kpi_array.reshape(n_weeks * seq_len, n_dim)

    df = pd.DataFrame(kpi_flat, columns=kpi_list)
    df.insert(0, "seed", seed)
    df.insert(
        0,
        "timestamp",
        pd.to_datetime(np.repeat(anchors_arr, seq_len))
        + pd.to_timedelta(np.tile(np.arange(seq_len), n_weeks), unit="h"),
    )
    df.insert(0, "window_anchor", np.repeat(anchors_arr, seq_len))
    df.insert(0, "cell_id", cell_id)

    return df
