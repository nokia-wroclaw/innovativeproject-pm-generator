import uuid

from pydantic import BaseModel, ConfigDict


class TokenPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    session_id: str | None = None
    preferred_username: str | None = None
    email: str | None = None
    sub: str | None = None

    def get_uuid(self) -> uuid.UUID:
        return uuid.UUID(self.user_id)
