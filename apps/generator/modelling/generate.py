from pathlib import Path

import numpy as np
import pandas as pd
from cvae_utils import (
    _to_numpy,
    load_artifacts,
    seasonal_features,
)

USER = "user"

SHARED_DIR_PATH = Path(f"/home/{USER}/app/apps/apps/generator/data/shared_dir")
TRAINING_DATA_PATH = SHARED_DIR_PATH / "preprocessed_dataset" / "final_scaled_only_minmax"

ARTIFACTS_DIR_PATH = SHARED_DIR_PATH / "artifacts"
RUN_DIR_PATH = ARTIFACTS_DIR_PATH / "run_4"
WEIGHTS_PATH = RUN_DIR_PATH / "models_weights"

MODEL_PATH = WEIGHTS_PATH / "cvae_lstm_v5_0.weights.h5"

DISTNAME = "bts_24/cell_5"
HOLIDAY = 0
ANCHOR_DATE = "2024-01-15"
N_WEEKS = 3

HP_V5 = dict(
    seq_len=168,
    n_dim=235,  # kpi number
    seed=42,
    epochs=300,
    batch_size=64,
    global_latent_dim=64,
    local_latent_dim=0,
    cell_embed_dim=32,
    hidden_dim=256,
    n_layers=2,
    use_attention=True,
    n_heads=4,
    beta=0.0,
    learning_rate=3e-4,
    free_bits_global=0.002,  # floor ≈ 0.128 nats (was 46.4 at 0.1×128 + 0.05×4×168)
    free_bits_local=0.0,
    output_activation="sigmoid",
    target_beta=2e-4,
    anneal_epochs=150,
    cycle_epochs=40,
    n_cycles=6,
    cycle_ratio=0.5,
)

KPI_COLS = [
    "NR_1025",
    "NR_11",
    "NR_1224",
    "NR_1225",
    "NR_125",
    "NR_126",
    "NR_1260",
    "NR_1266",
    "NR_127",
    "NR_128",
    "NR_1325",
    "NR_135",
    "NR_1395",
    "NR_1397",
    "NR_1399",
    "NR_1400",
    "NR_1401",
    "NR_1402b",
    "NR_1403",
    "NR_1403b",
    "NR_1418",
    "NR_1419",
    "NR_1425",
    "NR_1428",
    "NR_1439",
    "NR_1441",
    "NR_1472",
    "NR_1473",
    "NR_1474a",
    "NR_1479",
    "NR_1480",
    "NR_1482",
    "NR_1483",
    "NR_1484",
    "NR_1485",
    "NR_151",
    "NR_1542",
    "NR_1594",
    "NR_1595",
    "NR_1596",
    "NR_167",
    "NR_168",
    "NR_182",
    "NR_194",
    "NR_216",
    "NR_218",
    "NR_2193",
    "NR_2194",
    "NR_226",
    "NR_227",
    "NR_231",
    "NR_233",
    "NR_253",
    "NR_283",
    "NR_46",
    "NR_47",
    "NR_473",
    "NR_5003",
    "NR_5009",
    "NR_5012",
    "NR_5013",
    "NR_5015",
    "NR_5031",
    "NR_5035",
    "NR_5036",
    "NR_5037",
    "NR_5039",
    "NR_5040",
    "NR_5045",
    "NR_5046",
    "NR_505",
    "NR_5054",
    "NR_5055",
    "NR_5056",
    "NR_5057",
    "NR_5058",
    "NR_5059",
    "NR_5061",
    "NR_5062",
    "NR_5064",
    "NR_5065",
    "NR_5066",
    "NR_5067",
    "NR_5068",
    "NR_5069",
    "NR_5070",
    "NR_5072",
    "NR_5076",
    "NR_5077",
    "NR_5082",
    "NR_5083",
    "NR_5088",
    "NR_5089",
    "NR_5090",
    "NR_5091",
    "NR_5091d",
    "NR_5096",
    "NR_5099",
    "NR_5100",
    "NR_5101",
    "NR_5105",
    "NR_5108",
    "NR_5109",
    "NR_5109e",
    "NR_5110",
    "NR_5111",
    "NR_5112",
    "NR_5114",
    "NR_5115",
    "NR_5116",
    "NR_5118",
    "NR_5119",
    "NR_5120",
    "NR_5121",
    "NR_5122",
    "NR_5123",
    "NR_5124",
    "NR_5125",
    "NR_5127",
    "NR_5128",
    "NR_5129",
    "NR_5131",
    "NR_5142",
    "NR_5143",
    "NR_5144",
    "NR_5148",
    "NR_5149",
    "NR_515",
    "NR_5150",
    "NR_5151",
    "NR_5152",
    "NR_5160",
    "NR_5161",
    "NR_5162",
    "NR_5163",
    "NR_5164",
    "NR_5165",
    "NR_5166",
    "NR_5167",
    "NR_5168",
    "NR_5174",
    "NR_5177",
    "NR_5182",
    "NR_5183",
    "NR_5184",
    "NR_5192",
    "NR_5193",
    "NR_5194",
    "NR_5196",
    "NR_5205",
    "NR_5230",
    "NR_5241",
    "NR_5242",
    "NR_5243",
    "NR_5244",
    "NR_5245",
    "NR_5246",
    "NR_5247",
    "NR_5251",
    "NR_5258",
    "NR_5259",
    "NR_527",
    "NR_5323",
    "NR_5324",
    "NR_5325",
    "NR_5337",
    "NR_5342",
    "NR_5344",
    "NR_5345",
    "NR_5346",
    "NR_5347",
    "NR_5349",
    "NR_535",
    "NR_5350",
    "NR_5351",
    "NR_5356",
    "NR_5358d",
    "NR_5362",
    "NR_5363",
    "NR_5364",
    "NR_5366",
    "NR_5367",
    "NR_539",
    "NR_5398",
    "NR_5400",
    "NR_5414",
    "NR_5415",
    "NR_5416",
    "NR_5421",
    "NR_5422",
    "NR_5427",
    "NR_5432",
    "NR_5433",
    "NR_5445",
    "NR_5446",
    "NR_5447",
    "NR_5452",
    "NR_548",
    "NR_5486",
    "NR_5487b",
    "NR_5488",
    "NR_5513",
    "NR_5523",
    "NR_5525",
    "NR_5526",
    "NR_5527",
    "NR_5528",
    "NR_5530",
    "NR_5584",
    "NR_5593",
    "NR_5594",
    "NR_5621",
    "NR_5636",
    "NR_568",
    "NR_5737",
    "NR_5746",
    "NR_5749",
    "NR_578",
    "NR_579",
    "NR_584",
    "NR_5873",
    "NR_6",
    "NR_630",
    "NR_636",
    "NR_638",
    "NR_651b",
    "NR_668",
    "NR_669",
    "NR_672",
    "NR_707",
    "NR_709",
    "NR_718",
    "NR_954",
    "NR_972",
    "NR_973",
]


def _run_batched_generation(model, y_windows, batch_size=64):
    """Prior sampling from N(0,I) — true synthetic generation for v5."""
    decoded = []
    for start in range(0, len(y_windows), batch_size):
        yb = y_windows[start : start + batch_size]
        x_syn, _ = model.generate(yb)
        decoded.append(_to_numpy(x_syn))
    return np.concatenate(decoded)


def generate_windows(
    model,
    cell_encoder,
    cell_id,
    anchor_date,
    n_weeks,
    holiday,
    model_config,
):
    cell_idx = cell_encoder.transform([cell_id])[0]
    seq_len = model_config["seq_len"]

    # 1. Build all conditions upfront
    anchors = []
    y_windows = []
    for week in range(n_weeks):
        anchor = pd.Timestamp(anchor_date) + pd.Timedelta(weeks=week)
        seasonal = seasonal_features(anchor)
        y_windows.append([cell_idx, holiday, *seasonal])
        anchors.append(anchor)

    y_windows = np.array(y_windows, dtype=np.float32)  # (n_weeks, 6)
    anchors_arr = np.array(anchors)

    # 2. Batched generation
    kpi_array = _run_batched_generation(
        model, y_windows, batch_size=model_config["batch_size"]
    )  # (n_weeks, 168, n_kpis)
    kpi_flat = kpi_array.reshape(n_weeks * seq_len, model_config["n_dim"])

    # 3. Format into DataFrame
    df = pd.DataFrame(kpi_flat, columns=KPI_COLS)
    df.insert(0, "seed", model_config["seed"])
    df.insert(
        0,
        "timestamp",
        pd.to_datetime(np.repeat(anchors_arr, seq_len))
        + pd.to_timedelta(np.tile(np.arange(seq_len), n_weeks), unit="h"),
    )
    df.insert(0, "window_anchor", np.repeat(anchors_arr, seq_len))
    df.insert(0, "cell_id", cell_id)

    return df


def main(run_id_path, weights_path, cell_id, anchor_date, holiday, n_weeks, model_config):
    model, cell_encoder = load_artifacts(
        run_id_path=RUN_DIR_PATH,
        weights_path=MODEL_PATH,
    )
    synth_windows = generate_windows(
        model=model,
        cell_encoder=cell_encoder,
        cell_id=DISTNAME,
        anchor_date=ANCHOR_DATE,
        holiday=HOLIDAY,
        n_weeks=N_WEEKS,
        model_config=HP_V5,
    )
    return synth_windows


windows = main(
    run_id_path=RUN_DIR_PATH,
    weights_path=MODEL_PATH,
    cell_id=DISTNAME,
    anchor_date=ANCHOR_DATE,
    holiday=HOLIDAY,
    n_weeks=N_WEEKS,
    model_config=HP_V5,
)
