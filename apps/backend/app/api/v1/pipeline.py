from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.auth import require_admin, require_auth
from app.db.database import db_manager
from app.models.auth import TokenPayload
from app.models.pipeline import (
    PipelineRunCreate,
    PipelineRunDeleteResponse,
    PipelineRunRead,
)
from app.services.pipeline import PipelineService

router = APIRouter(dependencies=[Depends(require_auth)])


def get_pipeline_service(
    db: Session = Depends(db_manager.get_db),
) -> PipelineService:
    return PipelineService(db=db)


@router.get(
    "/pipelines",
    response_model=list[PipelineRunRead],
)
def list_pipeline_runs(
    service: PipelineService = Depends(get_pipeline_service),
) -> list[PipelineRunRead]:
    runs = service.get_runs()
    return [PipelineRunRead.model_validate(r) for r in runs]


@router.post(
    "/pipelines",
    response_model=PipelineRunRead,
    status_code=status.HTTP_201_CREATED,
)
def create_pipeline_run(
    body: PipelineRunCreate,
    service: PipelineService = Depends(get_pipeline_service),
) -> PipelineRunRead:
    new_run = service.create_run(body.dataset_id, body.pipeline_type)
    return PipelineRunRead.model_validate(new_run)


@router.delete(
    "/pipelines/{run_id}",
    response_model=PipelineRunDeleteResponse,
)
def delete_pipeline_run(
    run_id: int,
    service: PipelineService = Depends(get_pipeline_service),
    _user: TokenPayload = Depends(require_admin),
) -> PipelineRunDeleteResponse:
    run = service.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run with ID {run_id} not found."
        )
    
    service.delete_run(run_id)
    return PipelineRunDeleteResponse(message="deleted", run_id=run_id)
