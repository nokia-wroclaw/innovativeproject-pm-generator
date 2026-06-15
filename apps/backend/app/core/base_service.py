from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import Base


class ItemNotFoundError(Exception):
    pass


class BaseService[ModelType: Base, CreateSchemaType: BaseModel, UpdateSchemaType: BaseModel]:
    def __init__(self, model: type[ModelType], db: Session):
        self.model = model
        self._db = db

    def get(self, id: Any) -> ModelType | None:
        return self._db.query(self.model).filter(self.model.id == id).first()

    def get_multi(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
        return self._db.query(self.model).offset(skip).limit(limit).all()

    def create(self, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data)
        self._db.add(db_obj)
        self._db.commit()
        self._db.refresh(db_obj)
        return db_obj

    def update(self, db_obj: ModelType, obj_in: UpdateSchemaType | dict[str, Any]) -> ModelType:
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)

        self._db.add(db_obj)
        self._db.commit()
        self._db.refresh(db_obj)
        return db_obj

    def delete(self, id: int) -> ModelType:
        obj = self._db.query(self.model).filter(self.model.id == id).first()
        if not obj:
            raise ItemNotFoundError(f"Object with id {id} not found")
        self._db.delete(obj)
        self._db.commit()
        return obj
