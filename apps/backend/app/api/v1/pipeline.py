from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.auth import require_admin, require_auth
from app.db.database import db_manager
from app.models.pipeline import PipelineRunCreate, PipelineRunRead
from app.services.pipeline import PipelineService

router = APIRouter(dependencies=[Depends(require_auth)])


def get_pipeline_service(
    db: Session = Depends(db_manager.get_db),
) -> PipelineService:
    return PipelineService(db=db)


@router.get("/pipelines", response_model=list[PipelineRunRead])
def list_pipeline_runs(
    service: PipelineService = Depends(get_pipeline_service),
) -> list[PipelineRunRead]:
    return [PipelineRunRead.model_validate(r) for r in service.get_runs()]


@router.post("/pipelines", response_model=PipelineRunRead)
def create_pipeline_run(
    body: PipelineRunCreate,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineRunRead:
    return PipelineRunRead.model_validate(service.create_run(body.dataset_id, body.pipeline_type))


@router.delete("/pipelines/{run_id}")
def delete_pipeline_run(
    run_id: int,
    service: PipelineService = Depends(get_pipeline_service),
    _: dict = Depends(require_admin),
) -> dict:
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    service.delete_run(run_id)
    return {"message": "deleted", "run_id": run_id}
