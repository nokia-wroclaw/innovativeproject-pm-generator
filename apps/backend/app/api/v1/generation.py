from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import require_auth
from app.db.database import db_manager
from app.models.generation import GenerationRead
from app.services.generation import GenerationService

router = APIRouter(dependencies=[Depends(require_auth)])


def get_generation_service(
    db: Session = Depends(db_manager.get_db),
) -> GenerationService:
    return GenerationService(db=db)


@router.post("/generate", response_model=GenerationRead)
def generate(
    generation: GenerationRead,
    service: GenerationService = Depends(get_generation_service),
) -> GenerationRead:
    generate_response = service.create_generation(generation.name, generation.number)
    return GenerationRead.model_validate(generate_response)


# dummy endpoint for testing
@router.get("/test", response_model=list[GenerationRead])
def test(service: GenerationService = Depends(get_generation_service)) -> list[GenerationRead]:
    return [GenerationRead(name=f"Generation {i}", number=i) for i in range(1, 68)]
