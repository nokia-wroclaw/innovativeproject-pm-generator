from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import db_manager
from app.models.generation import GenerationRead
from app.services.generation import GenerationService

router = APIRouter()


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
@router.get("/test", response_model=GenerationRead)
def test(service: GenerationService = Depends(get_generation_service)):
    return GenerationRead(name="test123", number=67)
