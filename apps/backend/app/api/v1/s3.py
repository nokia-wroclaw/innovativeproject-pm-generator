from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.auth import get_user_identity, require_admin, require_auth
from app.core.storage_access import (
    assert_dataset_accessible,
    assert_raw_dataset,
    assert_storage_type_allowed,
    require_storage_admin,
)
from app.db.database import db_manager
from app.db.schemas import DatasetStatus, DatasetType
from app.models.auth import TokenPayload
from app.models.s3 import (
    AbortMultipartRequest,
    AbortMultipartResponse,
    CompleteMultipartRequest,
    CompleteMultipartResponse,
    DatasetCreate,
    DatasetPreviewResponse,
    DatasetRead,
    DatasetRegisterRequest,
    DatasetStatusUpdate,
    DatasetVisualizationResponse,
    DatasetVisualizationStatusResponse,
    DeleteDatasetResponse,
    MultipartInitiateResponse,
    PartUrlResponse,
)
from app.services.airflow.errors import AirflowIntegrationError, AirflowNotFound
from app.services.airflow.runtime import get_airflow_service
from app.services.airflow.service import AirflowService
from app.services.s3.service import S3Service
from app.services.s3.visualization import (
    DATASET_VISUALIZATION_DAG_ID,
    VisualizationSchemaError,
    get_dataset_visualization_status,
    trigger_dataset_visualization,
    trigger_dataset_visualization_on_raw_completed,
)
from app.services.s3.visualization_artifacts import VisualizationStorageError

router = APIRouter(dependencies=[Depends(require_auth)])


def get_s3_service(
    db: Session = Depends(db_manager.get_db),
) -> S3Service:
    return S3Service(db=db)


def _airflow_service() -> AirflowService:
    return get_airflow_service()


@router.post("/datasets", response_model=DatasetRead)
async def create_s3_dataset(
    dataset: DatasetCreate,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> DatasetRead:
    user_uuid = token_payload.get_uuid()

    s3_dataset = service.create_dataset(
        user_uuid=user_uuid,
        file_name=dataset.file_name,
        s3_key=dataset.s3_key,
        type=dataset.type,
        pm_metadata_s3_key=dataset.pm_metadata_s3_key,
    )

    return DatasetRead.model_validate(s3_dataset)


@router.post("/datasets/{dataset_id}/multipart/initiate", response_model=MultipartInitiateResponse)
async def initiate_multipart(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> MultipartInitiateResponse:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_accessible(token_payload, dataset)
    assert_raw_dataset(dataset, context="upload")

    result = service.initiate_multipart_upload(dataset)
    return MultipartInitiateResponse.model_validate(result)


@router.get("/datasets/{dataset_id}/multipart/part-url", response_model=PartUrlResponse)
async def get_part_url(
    dataset_id: int,
    upload_id: str = Query(...),
    part_number: int = Query(...),
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> PartUrlResponse:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_accessible(token_payload, dataset)
    assert_raw_dataset(dataset, context="upload")

    url = service.get_presigned_part_url(dataset.s3_key, upload_id, part_number)
    return PartUrlResponse.model_validate({"url": url})


@router.post("/datasets/{dataset_id}/multipart/complete", response_model=CompleteMultipartResponse)
async def complete_multipart(
    dataset_id: int,
    request: CompleteMultipartRequest,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> CompleteMultipartResponse:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_accessible(token_payload, dataset)
    assert_raw_dataset(dataset, context="upload")

    parts_dicts = [{"PartNumber": p.PartNumber, "ETag": p.ETag} for p in request.parts]
    response_dict = service.complete_multipart_upload(
        dataset.s3_key, request.upload_id, parts_dicts
    )
    return CompleteMultipartResponse.model_validate(response_dict)


@router.post("/datasets/register", response_model=DatasetRead)
async def register_s3_dataset(
    request: DatasetRegisterRequest,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> DatasetRead:
    user_uuid = token_payload.get_uuid()

    s3_dataset = service.register_existing_dataset(
        user_uuid=user_uuid,
        s3_key=request.s3_key,
        file_name=request.file_name,
        type=request.type,
        pm_metadata_s3_key=request.pm_metadata_s3_key,
    )

    await trigger_dataset_visualization_on_raw_completed(
        s3_dataset,
        s3_service=service,
        triggered_by=get_user_identity(token_payload),
    )

    return DatasetRead.model_validate(s3_dataset)


@router.post("/datasets/{dataset_id}/multipart/abort", response_model=AbortMultipartResponse)
async def abort_multipart(
    dataset_id: int,
    request: AbortMultipartRequest,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> AbortMultipartResponse:
    dataset = service.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    assert_dataset_accessible(token_payload, dataset)
    assert_raw_dataset(dataset, context="upload")

    service.abort_multipart_upload(dataset.s3_key, request.upload_id)
    return AbortMultipartResponse(status="aborted")


@router.post("/datasets/update_status", response_model=DatasetRead)
async def confirm_s3_dataset(
    dataset_status: DatasetStatusUpdate,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_storage_admin),
) -> DatasetRead:
    dataset = service.get_dataset(dataset_status.dataset_id)
    assert_dataset_accessible(token_payload, dataset)
    assert_raw_dataset(dataset, context="status")
    was_completed = dataset.status == DatasetStatus.COMPLETED
    updated = service.change_dataset_status(dataset_status.dataset_id, dataset_status.status)
    if dataset_status.status == DatasetStatus.COMPLETED and not was_completed:
        await trigger_dataset_visualization_on_raw_completed(
            updated,
            s3_service=service,
            triggered_by=get_user_identity(token_payload),
        )
    return DatasetRead.model_validate(updated)


@router.get("/datasets", response_model=list[DatasetRead])
def get_s3_datasets(
    type: DatasetType = Query(..., description="Dataset category: RAW, PREPROCESSED, or GENERATED"),
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_auth),
) -> list[DatasetRead]:
    assert_storage_type_allowed(token_payload, type)
    return [DatasetRead.model_validate(dataset) for dataset in service.get_datasets(type)]


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def preview_s3_dataset(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_auth),
) -> DatasetPreviewResponse:
    dataset = service.get_dataset(dataset_id)
    assert_dataset_accessible(token_payload, dataset)
    return DatasetPreviewResponse.model_validate(service.preview_dataset(dataset_id))


@router.post(
    "/datasets/{dataset_id}/visualization",
    response_model=DatasetVisualizationResponse,
)
async def request_dataset_visualization(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
    airflow: AirflowService = Depends(_airflow_service),
    token_payload: TokenPayload = Depends(require_auth),
) -> DatasetVisualizationResponse:
    dataset = service.get_dataset(dataset_id)
    assert_dataset_accessible(token_payload, dataset)

    try:
        return await trigger_dataset_visualization(
            dataset,
            airflow=airflow,
            s3_service=service,
            triggered_by=get_user_identity(token_payload),
        )
    except VisualizationSchemaError as exc:
        raise HTTPException(status_code=422, detail=exc.payload) from exc
    except VisualizationStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AirflowNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"DAG '{DATASET_VISUALIZATION_DAG_ID}' is not registered in Airflow. "
                "Ensure apps/airflow/dags is mounted and the scheduler has parsed the DAG."
            ),
        ) from exc
    except AirflowIntegrationError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc


@router.get(
    "/datasets/{dataset_id}/visualization/status",
    response_model=DatasetVisualizationStatusResponse,
)
async def get_dataset_visualization_status_endpoint(
    dataset_id: int,
    service: S3Service = Depends(get_s3_service),
    airflow: AirflowService = Depends(_airflow_service),
    token_payload: TokenPayload = Depends(require_auth),
) -> DatasetVisualizationStatusResponse:
    dataset = service.get_dataset(dataset_id)
    assert_dataset_accessible(token_payload, dataset)
    try:
        return await get_dataset_visualization_status(
            dataset_id,
            dataset,
            airflow=airflow,
            s3_service=service,
        )
    except AirflowNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"DAG '{DATASET_VISUALIZATION_DAG_ID}' is not registered in Airflow. "
                "Ensure apps/airflow/dags is mounted and the scheduler has parsed the DAG."
            ),
        ) from exc
    except AirflowIntegrationError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.message) from exc


@router.delete("/datasets/{dataset_id}", response_model=DeleteDatasetResponse)
def delete_s3_dataset(
    dataset_id: int,
    delete_from_s3: bool = Query(
        False,
        description="When true, also remove the object from S3/MinIO.",
    ),
    service: S3Service = Depends(get_s3_service),
    token_payload: TokenPayload = Depends(require_admin),
) -> DeleteDatasetResponse:
    dataset = service.get_dataset(dataset_id)
    assert_dataset_accessible(token_payload, dataset)
    service.delete_dataset(dataset_id, delete_from_s3=delete_from_s3)
    scope = "database and S3" if delete_from_s3 else "database only"
    return DeleteDatasetResponse(
        message=f"Dataset deleted successfully ({scope})",
        dataset_id=dataset_id,
        deleted_from_s3=delete_from_s3,
    )
