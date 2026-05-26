from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db.schemas import DagRunStatus, DatasetStatus


class ModelingDatasetOption(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: int
    file_name: str
    s3_key: str
    status: DatasetStatus


ModelingProcessType = Literal["preprocessing_feature_engineering", "training_dataset"]


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


class ModelingRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(gt=0)
    dataset_type: Literal["working_days", "weekends"]
    epochs: int = Field(default=10, ge=1, le=10_000)
    batch_size: Literal[16, 32, 64, 128] = 32
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    target_s3_key: str | None = None
    dag_args: dict[str, Any] = Field(default_factory=dict)


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
