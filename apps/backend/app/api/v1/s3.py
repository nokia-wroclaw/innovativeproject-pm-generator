import typing

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.auth import require_auth
from app.db.database import db_manager
from app.models.s3 import DatasetRead, DatasetCreate, UploadUrlResponse

from app.services.s3 import S3Service

router = APIRouter(dependencies=[Depends(require_auth)])


def get_s3_service(
    db: Session = Depends(db_manager.get_db),
) -> S3Service:
    return S3Service(db=db)


@router.post("/datasets", response_model=UploadUrlResponse)
async def create_s3_dataset(
    dataset: DatasetCreate,
    service: S3Service = Depends(get_s3_service),
    token_payload: dict[str, typing.Any] = Depends(require_auth),
) -> UploadUrlResponse:
    user_uuid = token_payload.get("sub")

    s3_dataset = service.create_dataset(
        user_uuid=user_uuid,
        file_name=dataset.file_name,
        s3_key=dataset.s3_key,
    )

    presigned_url = service.get_presigned_url(s3_dataset)
    return UploadUrlResponse.model_validate({"url": presigned_url, "file_name": dataset.file_name})


@router.get("/datasets", response_model=list[DatasetRead])
def get_s3_datasets(service: S3Service = Depends(get_s3_service)) -> list[DatasetRead]:
    return [DatasetRead.model_validate(dataset) for dataset in service.get_datasets()]
