from sqlalchemy.orm import Session

from ..db.schemas import Generation


class GenerationService:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_generation(self, name: str, number: int) -> Generation:
        generation = Generation(name=name, number=number)
        self._db.add(generation)
        self._db.commit()
        self._db.refresh(generation)
        return generation
