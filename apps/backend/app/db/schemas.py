import datetime
import enum
import uuid
from typing import Literal

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, Uuid
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class DatasetStatus(enum.Enum):
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


class PipelineType(enum.Enum):
    PREPROCESSING = "PREPROCESSING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    TRAINING = "TRAINING"


class PipelineRunStatus(enum.Enum):
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


RunType = Literal["manual", "scheduled", "backfill", "asset_triggered"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String)


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
