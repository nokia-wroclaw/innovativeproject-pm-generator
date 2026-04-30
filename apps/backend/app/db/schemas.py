from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String)


class Generation(Base):
    __tablename__ = "generations"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    number: Mapped[int] = mapped_column(Integer)
