from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db.schemas import DagRunStatus, DatasetStatus, DatasetType

ModelingProcessType = Literal[
    "preprocessing_feature_engineering",
    "training_dataset",
    "generate",
]


class ModelingDatasetOption(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: int
    file_name: str
    s3_key: str
    status: DatasetStatus
    type: DatasetType


class ModelingFormOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | int | float | bool
    label: str


class ModelingFormField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    label: str
    type: Literal["dataset_select", "radio", "select", "integer", "float", "json", "text"]
    required: bool = False
    default: Any | None = None
    options: list[ModelingFormOption] = Field(default_factory=list)
    help: str | None = None
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None


class ModelingFormSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_type: ModelingProcessType
    title: str
    fields: list[ModelingFormField]


class ModelingTrainedModelOption(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    name: str
    path: str
    encoder_s3_key: str | None = None
    config_s3_key: str | None = None
    dataset_id: int | None = None
    created_at: str | None = None

    @field_validator("created_at", mode="before")
    @classmethod
    def serialize_created_at(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return v.isoformat() + "Z"
        return v


class ModelingRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(gt=0)
    dataset_type: Literal["working_days", "weekends"] = "working_days"
    dag_args: dict[str, Any] = Field(default_factory=dict)


class GenerateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    encoder_s3_key: str = Field(min_length=1)
    config_s3_key: str = Field(min_length=1)
    cell_id: str = ""  # Empty = generate for all cells
    anchor_date: str = Field(min_length=1)
    n_weeks: int = Field(ge=1)
    holiday: int = Field(default=0, ge=0, le=1)
    prompt: str = ""
    comparison_dataset_id: int | None = Field(default=None, gt=0)
    dag_args: dict[str, Any] = Field(default_factory=dict)
    kpis: list[str] = Field(default_factory=list)


class ModelingAutofillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(min_length=1, max_length=4000)
    current_values: dict[str, Any] = Field(default_factory=dict)


class ModelingAutofillResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_type: ModelingProcessType
    values: dict[str, Any]


class ModelingRunCreated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_type: ModelingProcessType
    dag_id: str
    run_id: str | None
    message: str
    airflow_status: int
    conf: dict[str, Any]


class ModelingArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "preprocessed_dataset",
        "featured_dataset",
        "training_dataset",
        "model_pickle",
        "generated_event_log",
        "generation_report",
    ]
    path: str
    status: Literal["pending", "saved"]


class ModelingRunStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_type: ModelingProcessType
    dag_id: str
    run_id: str
    status: DagRunStatus
    raw_state: str
    start_date: str | None = None
    end_date: str | None = None
    duration_ms: int | None = None
    logs: list[str] = Field(default_factory=list)
    metrics: dict[str, float] | None = None
    artifacts: list[ModelingArtifact] = Field(default_factory=list)


class TrainedModelCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    s3_key: str = Field(min_length=1)
    encoder_s3_key: str = Field(min_length=1)
    config_s3_key: str = Field(min_length=1)
    dataset_id: int


class ModelUploadInitiateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    file_name: str
    dataset_id: int


class ModelUploadInitiateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: int
    s3_key: str
    upload_url: str


class TrainedModelUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1)
    s3_key: str | None = Field(default=None, min_length=1)
    encoder_s3_key: str | None = Field(default=None, min_length=1)
    config_s3_key: str | None = Field(default=None, min_length=1)
    dataset_id: int | None = Field(default=None, gt=0)


class DeleteModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    model_id: int
    deleted_from_s3: bool
