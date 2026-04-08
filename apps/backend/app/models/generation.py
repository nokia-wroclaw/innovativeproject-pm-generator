from pydantic import BaseModel, ConfigDict


class GenerationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    number: int


class GenerationCreate(BaseModel):
    name: str
    number: int
