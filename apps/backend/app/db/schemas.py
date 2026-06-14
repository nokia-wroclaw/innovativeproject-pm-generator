import datetime
import enum
import uuid
from typing import Literal

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, Uuid
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.database import Base

RunType = Literal["manual", "scheduled", "backfill", "asset_triggered"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class DatasetStatus(enum.StrEnum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DatasetType(enum.StrEnum):
    RAW = "RAW"
    PREPROCESSED = "PREPROCESSED"
    GENERATED = "GENERATED"
    KPI_DEFINITIONS = "KPI_DEFINITIONS"
    SIMPLE_REPORTS = "SIMPLE_REPORTS"


class PipelineType(enum.StrEnum):
    PREPROCESSING = "PREPROCESSING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    TRAINING = "TRAINING"


class PipelineRunStatus(enum.StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskStatus(enum.StrEnum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    UP_FOR_RETRY = "up_for_retry"
    QUEUED = "queued"
    SKIPPED = "skipped"
    NONE = "none"


class DagRunStatus(enum.StrEnum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    QUEUED = "queued"



class Generation(Base):
    __tablename__ = "generations"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    number: Mapped[int] = mapped_column(Integer)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement=True)
    user_uuid: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)

    s3_key: Mapped[str] = mapped_column(String, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    pm_metadata_s3_key: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[DatasetStatus] = mapped_column(default=DatasetStatus.PENDING)
    type: Mapped[DatasetType] = mapped_column(
        SQLEnum(DatasetType, name="dataset_type"),
        default=DatasetType.RAW,
        nullable=False,
        index=True,
        primary_key=True,
    )

    __table_args__ = (
        UniqueConstraint("s3_key", "type", name="uq_dataset_s3_key_type"),
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"))
    pipeline_type: Mapped[PipelineType] = mapped_column(default=PipelineType.PREPROCESSING)
    status: Mapped[PipelineRunStatus] = mapped_column(default=PipelineRunStatus.PENDING)
    airflow_run_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )



class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, unique=True, autoincrement=True)
    user_uuid: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    s3_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    encoder_s3_key: Mapped[str] = mapped_column(String, nullable=False)
    config_s3_key: Mapped[str] = mapped_column(String, nullable=False)
    dataset_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    @property
    def path(self) -> str:
        return f"s3://{get_settings().s3_bucket}/{self.s3_key}"

