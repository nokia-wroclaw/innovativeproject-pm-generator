import typing

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.auth import require_admin, require_auth
from app.db.database import db_manager
from app.services.s3.service import S3Service

from app.models.s3 import (
    DatasetRead,
    DatasetCreate,
    DatasetStatusUpdate,
    CompleteMultipartRequest,
    AbortMultipartRequest,
    MultipartInitiateResponse,
    PartUrlResponse,
    DatasetRegisterRequest,
    DatasetPreviewResponse,
)

router = APIRouter(dependencies=[Depends(require_auth)])


def get_s3_service(
    db: Session = Depends(db_manager.get_db),
) -> S3Service:
    return S3Service(db=db)


@router.post("/datasets", response_model=DatasetRead)
async def create_s3_dataset(
    dataset: DatasetCreate,
    service: S3Service = Depends(get_s3_service),
    token_payload: dict[str, typing.Any] = Depends(require_auth),
) -> DatasetRead:
    user_uuid = token_payload.get("sub")

    s3_dataset = service.create_dataset(
        user_uuid=user_uuid,
        file_name=dataset.file_name,
        s3_key=dataset.s3_key,
    )

    return DatasetRead.model_validate(s3_dataset)


@router.post("/datasets/{dataset_id}/multipart/initiate", response_model=MultipartInitiateResponse)
async def initiate_multipart(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
):
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    result = service.initiate_multipart_upload(dataset)
    return MultipartInitiateResponse.model_validate(result)


@router.get("/datasets/{dataset_id}/multipart/part-url", response_model=PartUrlResponse)
async def get_part_url(
    dataset_id: int,
    upload_id: str = Query(...),
    part_number: int = Query(...),
    service: S3Service = Depends(get_s3_service),
):
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    url = service.get_presigned_part_url(dataset.s3_key, upload_id, part_number)
    return PartUrlResponse.model_validate({"url": url})


@router.post("/datasets/{dataset_id}/multipart/complete")
async def complete_multipart(
    dataset_id: int, request: CompleteMultipartRequest, service: S3Service = Depends(get_s3_service)
):
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    parts_dicts = [{"PartNumber": p.PartNumber, "ETag": p.ETag} for p in request.parts]
    return service.complete_multipart_upload(dataset.s3_key, request.upload_id, parts_dicts)


@router.post("/datasets/register", response_model=DatasetRead)
async def register_s3_dataset(
        request: DatasetRegisterRequest,
        service: S3Service = Depends(get_s3_service),
        token_payload: dict[str, typing.Any] = Depends(require_auth),
) -> DatasetRead:
    user_uuid = token_payload["user_id"]

    s3_dataset = service.register_existing_dataset(
        user_uuid=user_uuid,
        s3_key=request.s3_key,
        file_name=request.file_name,
    )

    return DatasetRead.model_validate(s3_dataset)


@router.post("/datasets/{dataset_id}/multipart/abort")
async def abort_multipart(
    dataset_id: int, request: AbortMultipartRequest, service: S3Service = Depends(get_s3_service)
):
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    service.abort_multipart_upload(dataset.s3_key, request.upload_id)
    return {"status": "aborted"}


@router.post("/datasets/update_status", response_model=DatasetRead)
async def confirm_s3_dataset(
    dataset_status: DatasetStatusUpdate, service: S3Service = Depends(get_s3_service)
) -> DatasetRead:
    return DatasetRead.model_validate(
        service.change_dataset_status(dataset_status.dataset_id, dataset_status.status)
    )


@router.get("/datasets", response_model=list[DatasetRead])
def get_s3_datasets(service: S3Service = Depends(get_s3_service)) -> list[DatasetRead]:
    return [DatasetRead.model_validate(dataset) for dataset in service.get_datasets()]


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_s3_dataset(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
) -> DatasetPreviewResponse:
    return DatasetPreviewResponse.model_validate(service.preview_dataset(dataset_id))


@router.delete("/datasets/{dataset_id}")
def delete_s3_dataset(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
    _: dict[str, typing.Any] = Depends(require_admin),
) -> dict:
    service.delete_dataset(dataset_id)
    return {"message": "dataset deleted successfully", "dataset_id": dataset_id}
