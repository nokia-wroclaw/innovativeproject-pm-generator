import json
import logging
import os
from typing import Any

import httpx

from app.models.modeling import (
    ModelingDatasetOption,
    ModelingProcessType,
    ModelingTrainedModelOption,
)

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_S = 60.0

_PROCESS_ROLES: dict[ModelingProcessType, str] = {
    "preprocessing_feature_engineering": "preprocessing and feature engineering",
    "training_dataset": "training dataset creation",
    "generate": "synthetic event log generation",
}

_FIELD_SPECS: dict[ModelingProcessType, dict[str, str]] = {
    "preprocessing_feature_engineering": {
        "dataset_id": "integer; must match one of the available dataset ids",
        "dataset_type": '"working_days" or "weekends"',
        "kpi_definitions_raw_path": "string; path to KPI definitions parquet",
        "simple_reports_raw_path": "string; path to simple reports parquet",
        "output_path_prefix": "string; output directory prefix",
        "kpi_min_global_density": "float between 0 and 1",
        "kpi_global_min_frac_cells_passing": "float between 0 and 1",
        "kpi_window_coverage_frac": "float between 0 and 1",
        "min_imputable_gap_frac": "float between 0 and 1",
        "kpi_min_std_val": "float >= 0",
        "max_zero_frac": "float between 0 and 1",
        "window_width_hours": "integer >= 1",
        "stride_hours": "integer >= 1",
        "max_gap_hours": "integer >= 1",
        "min_joint_windows_abs": "integer >= 1 or null",
        "impute": "boolean",
    },
    "training_dataset": {
        "dataset_id": "integer; must match one of the available dataset ids",
        "dataset_type": '"working_days" or "weekends"',
        "target_column": "string; name of the target column in the dataset",
        "test_size": "float between 0.05 and 0.5",
        "random_seed": "positive integer",
        "split_date": 'ISO date string "YYYY-MM-DD" or null',
        "shuffle": "boolean",
        "stratify": "boolean",
    },
    "generate": {
        "model_id": "string; must match one of the available trained model ids",
        "prompt": "string; detailed generation prompt for the synthetic event log DAG",
    },
}


class LlmAutofillError(Exception):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _llm_api_key() -> str | None:
    key = os.getenv("LLM_API_KEY", "").strip()
    return key or None


def _llm_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _llm_model() -> str:
    return os.getenv("LLM_MODEL", "gpt-5-nano").strip() or "gpt-5-nano"


def _llm_temperature() -> float | None:
    raw = os.getenv("LLM_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _coerce_float(value: Any, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = default
    if lo is not None:
        num = max(lo, num)
    if hi is not None:
        num = min(hi, num)
    return num


def _coerce_int(value: Any, default: int, *, lo: int = 1) -> int:
    try:
        num = int(value)
    except (TypeError, ValueError):
        num = default
    return max(lo, num)


def _coerce_optional_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, "", "null"):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _pick_enum(value: Any, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _current_float(current: dict[str, Any], key: str, fallback: float) -> float:
    raw = current.get(key, fallback)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _current_int(current: dict[str, Any], key: str, fallback: int) -> int:
    raw = current.get(key, fallback)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback


def _resolve_dataset_id(
    value: Any,
    datasets: list[ModelingDatasetOption],
    current: Any = None,
) -> int:
    dataset_ids = {d.id for d in datasets}
    dataset_id = value
    if isinstance(dataset_id, str) and dataset_id.isdigit():
        dataset_id = int(dataset_id)
    if isinstance(dataset_id, (int, float)) and int(dataset_id) in dataset_ids:
        return int(dataset_id)

    if current is not None:
        if isinstance(current, str) and current.isdigit():
            current = int(current)
        if isinstance(current, (int, float)) and int(current) in dataset_ids:
            return int(current)

    if datasets:
        return datasets[0].id
    raise LlmAutofillError("No completed datasets available.", status_code=409)


def _normalize(
    process_type: ModelingProcessType,
    raw: dict[str, Any],
    current_values: dict[str, Any],
    datasets: list[ModelingDatasetOption],
    models: list[ModelingTrainedModelOption],
) -> dict[str, Any]:
    allowed = _FIELD_SPECS[process_type]
    patch = {key: value for key, value in raw.items() if key in allowed}
    if not patch:
        raise LlmAutofillError(
            "Instruction did not map to any form fields. Mention specific parameters to update.",
            status_code=422,
        )

    current = current_values or {}
    result: dict[str, Any] = {}

    if process_type == "generate":
        model_ids = {m.id for m in models}
        if "model_id" in patch:
            model_id = patch["model_id"]
            if not isinstance(model_id, str) or model_id not in model_ids:
                current_model = current.get("model_id")
                model_id = current_model if isinstance(current_model, str) and current_model in model_ids else None
                if not model_id and models:
                    model_id = models[0].id
            if not model_id:
                raise LlmAutofillError("No trained models available to select.", status_code=409)
            result["model_id"] = model_id
        if "prompt" in patch:
            prompt = patch["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise LlmAutofillError("LLM did not return a valid generation prompt.", status_code=422)
            result["prompt"] = prompt.strip()
        return result

    if "dataset_id" in patch:
        result["dataset_id"] = _resolve_dataset_id(patch["dataset_id"], datasets, current.get("dataset_id"))
    if "dataset_type" in patch:
        result["dataset_type"] = _pick_enum(
            patch["dataset_type"],
            {"working_days", "weekends"},
            _pick_enum(current.get("dataset_type"), {"working_days", "weekends"}, "working_days"),
        )

    if process_type == "preprocessing_feature_engineering":
        if "kpi_definitions_raw_path" in patch:
            result["kpi_definitions_raw_path"] = _coerce_str(
                patch["kpi_definitions_raw_path"], _coerce_str(current.get("kpi_definitions_raw_path"))
            )
        if "simple_reports_raw_path" in patch:
            result["simple_reports_raw_path"] = _coerce_str(
                patch["simple_reports_raw_path"], _coerce_str(current.get("simple_reports_raw_path"))
            )
        if "output_path_prefix" in patch:
            result["output_path_prefix"] = _coerce_str(
                patch["output_path_prefix"], _coerce_str(current.get("output_path_prefix"))
            )
        if "kpi_min_global_density" in patch:
            result["kpi_min_global_density"] = _coerce_float(
                patch["kpi_min_global_density"], _current_float(current, "kpi_min_global_density", 0.5), lo=0, hi=1
            )
        if "kpi_global_min_frac_cells_passing" in patch:
            result["kpi_global_min_frac_cells_passing"] = _coerce_float(
                patch["kpi_global_min_frac_cells_passing"],
                _current_float(current, "kpi_global_min_frac_cells_passing", 0.8),
                lo=0,
                hi=1,
            )
        if "kpi_window_coverage_frac" in patch:
            result["kpi_window_coverage_frac"] = _coerce_float(
                patch["kpi_window_coverage_frac"],
                _current_float(current, "kpi_window_coverage_frac", 0.7),
                lo=0,
                hi=1,
            )
        if "min_imputable_gap_frac" in patch:
            result["min_imputable_gap_frac"] = _coerce_float(
                patch["min_imputable_gap_frac"],
                _current_float(current, "min_imputable_gap_frac", 0.1),
                lo=0,
                hi=1,
            )
        if "kpi_min_std_val" in patch:
            result["kpi_min_std_val"] = _coerce_float(
                patch["kpi_min_std_val"], _current_float(current, "kpi_min_std_val", 0.01), lo=0
            )
        if "max_zero_frac" in patch:
            result["max_zero_frac"] = _coerce_float(
                patch["max_zero_frac"], _current_float(current, "max_zero_frac", 0.9), lo=0, hi=1
            )
        if "window_width_hours" in patch:
            result["window_width_hours"] = _coerce_int(
                patch["window_width_hours"], _current_int(current, "window_width_hours", 168)
            )
        if "stride_hours" in patch:
            result["stride_hours"] = _coerce_int(patch["stride_hours"], _current_int(current, "stride_hours", 24))
        if "max_gap_hours" in patch:
            result["max_gap_hours"] = _coerce_int(patch["max_gap_hours"], _current_int(current, "max_gap_hours", 6))
        if "min_joint_windows_abs" in patch:
            current_optional = current.get("min_joint_windows_abs")
            default_optional = None if current_optional in (None, "") else _coerce_optional_int(current_optional)
            result["min_joint_windows_abs"] = _coerce_optional_int(patch["min_joint_windows_abs"], default_optional)
        if "impute" in patch:
            result["impute"] = _coerce_bool(patch["impute"], _coerce_bool(current.get("impute"), True))
        return result

    if "target_column" in patch:
        target = patch["target_column"]
        if not isinstance(target, str) or not target.strip():
            raise LlmAutofillError('LLM did not return a valid "target_column".', status_code=422)
        result["target_column"] = target.strip()
    if "test_size" in patch:
        result["test_size"] = _coerce_float(
            patch["test_size"], _current_float(current, "test_size", 0.2), lo=0.05, hi=0.5
        )
    if "random_seed" in patch:
        result["random_seed"] = _coerce_int(patch["random_seed"], _current_int(current, "random_seed", 42))
    if "split_date" in patch:
        split_date = patch["split_date"]
        if split_date in (None, "", "null"):
            result["split_date"] = ""
        else:
            result["split_date"] = _coerce_str(split_date, _coerce_str(current.get("split_date")))[:10]
    if "shuffle" in patch:
        result["shuffle"] = _coerce_bool(patch["shuffle"], _coerce_bool(current.get("shuffle"), True))
    if "stratify" in patch:
        result["stratify"] = _coerce_bool(patch["stratify"], _coerce_bool(current.get("stratify"), False))
    return result


def _build_context(
    process_type: ModelingProcessType,
    datasets: list[ModelingDatasetOption],
    models: list[ModelingTrainedModelOption],
) -> str:
    if process_type == "generate":
        lines = ["Available trained models:"]
        for model in models:
            lines.append(f"- id={model.id!r}, name={model.name!r}")
        return "\n".join(lines)

    lines = ["Available datasets (COMPLETED only):"]
    for dataset in datasets:
        lines.append(f"- id={dataset.id}, file_name={dataset.file_name!r}, type={dataset.type}")
    return "\n".join(lines)


def _build_system_prompt(process_type: ModelingProcessType) -> str:
    role = _PROCESS_ROLES[process_type]
    field_lines = "\n".join(
        f'  "{name}": {desc}' for name, desc in _FIELD_SPECS[process_type].items()
    )
    return (
        f"You configure a {role} form. "
        "Respond with a JSON object containing ONLY the fields the user instruction explicitly "
        "mentions or clearly implies should change. "
        "Omit every other field — unchanged values are kept as-is. "
        "Use exact enum values where specified.\n"
        f"Available fields:\n{field_lines}"
    )


async def autofill_modeling_form(
    process_type: ModelingProcessType,
    instruction: str,
    *,
    current_values: dict[str, Any],
    datasets: list[ModelingDatasetOption],
    models: list[ModelingTrainedModelOption],
) -> dict[str, Any]:
    api_key = _llm_api_key()
    if not api_key:
        raise LlmAutofillError(
            "LLM autofill is not configured. Set LLM_API_KEY in the backend environment.",
            status_code=503,
        )

    current_json = json.dumps(current_values or {}, ensure_ascii=False)
    user_message = (
        f"Process type: {process_type}\n\n"
        f"{_build_context(process_type, datasets, models)}\n\n"
        f"Current form values:\n{current_json}\n\n"
        f"User instruction:\n{instruction.strip()}"
    )

    payload: dict[str, Any] = {
        "model": _llm_model(),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _build_system_prompt(process_type)},
            {"role": "user", "content": user_message},
        ],
    }
    temperature = _llm_temperature()
    if temperature is not None:
        payload["temperature"] = temperature

    url = f"{_llm_base_url()}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=_LLM_TIMEOUT_S) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as exc:
        raise LlmAutofillError("LLM request timed out.", status_code=504) from exc
    except httpx.HTTPError as exc:
        logger.exception("LLM HTTP error")
        raise LlmAutofillError("Failed to reach the LLM API.", status_code=502) from exc

    if response.status_code >= 400:
        logger.warning("LLM API error %s: %s", response.status_code, response.text[:500])
        raise LlmAutofillError(
            f"LLM API returned status {response.status_code}.",
            status_code=502,
        )

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        logger.exception("Invalid LLM response structure")
        raise LlmAutofillError("LLM returned an invalid JSON response.", status_code=502) from exc

    if not isinstance(parsed, dict):
        raise LlmAutofillError("LLM response must be a JSON object.", status_code=502)

    return _normalize(process_type, parsed, current_values, datasets, models)
