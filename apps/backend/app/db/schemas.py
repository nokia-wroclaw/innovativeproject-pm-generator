import uuid

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

    s3_key: Mapped[str] = mapped_column(String)
    s3_bucket: Mapped[str] = mapped_column(String)
    file_name: Mapped[str] = mapped_column(String)

    status: Mapped[DatasetStatus] = mapped_column(default=DatasetStatus.PENDING)
