from sqlalchemy.orm import Session


class AirflowService:
    def __init__(self, db: Session) -> None:
        self._db = db
