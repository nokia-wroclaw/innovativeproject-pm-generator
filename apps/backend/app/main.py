from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1 import generation

from .core.logging import setup_logging
from .db import schemas
from .db.database import db_manager


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    schemas.Base.metadata.create_all(bind=db_manager.engine)
    setup_logging()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(generation.router, prefix="/api/v1")
