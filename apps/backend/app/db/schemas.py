import uuid
from typing import Literal

from sqlalchemy import Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

import enum


class DatasetStatus(enum.Enum):
    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


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

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_uuid: Mapped[uuid.UUID] = mapped_column(Uuid, index=True)

    s3_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)

    status: Mapped[DatasetStatus] = mapped_column(default=DatasetStatus.PENDING)


class TaskStatus(str, enum.Enum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    UP_FOR_RETRY = "up_for_retry"
    QUEUED = "queued"
    SKIPPED = "skipped"
    NONE = "none"


class DagRunStatus(str, enum.Enum):
    SUCCESS = "success"
    RUNNING = "running"
    FAILED = "failed"
    QUEUED = "queued"


RunType = Literal["manual", "scheduled", "backfill", "asset_triggered"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
