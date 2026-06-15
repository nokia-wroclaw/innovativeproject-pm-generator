from typing import Literal

from fastapi import Depends, HTTPException

from app.core.auth import _extract_roles, _forbidden, get_keycloak_settings, require_auth
from app.db.schemas import Dataset, DatasetType
from app.models.auth import TokenPayload

_ERR_RAW_UPLOAD = "Multipart upload allowed only for RAW datasets"
_ERR_RAW_STATUS = "Status updates allowed only for RAW datasets"


def is_storage_admin(payload: TokenPayload) -> bool:
    settings = get_keycloak_settings()
    roles = _extract_roles(payload, settings.client_id)
    return settings.admin_role in roles


def visible_storage_types(payload: TokenPayload) -> set[DatasetType]:
    if is_storage_admin(payload):
        return {
            DatasetType.RAW,
            DatasetType.KPI_DEFINITIONS,
            DatasetType.SIMPLE_REPORTS,
            DatasetType.PREPROCESSED,
            DatasetType.GENERATED,
        }
    return {DatasetType.GENERATED}


def assert_storage_type_allowed(payload: TokenPayload, dataset_type: DatasetType) -> None:
    if dataset_type not in visible_storage_types(payload):
        raise _forbidden("You do not have permission to access this dataset category")


def assert_raw_dataset(dataset: Dataset, *, context: Literal["upload", "status"]) -> None:
    if dataset.type not in {
        DatasetType.RAW,
        DatasetType.KPI_DEFINITIONS,
        DatasetType.SIMPLE_REPORTS,
        DatasetType.PREPROCESSED,
    }:
        detail = _ERR_RAW_STATUS if context == "status" else _ERR_RAW_UPLOAD
        raise HTTPException(status_code=403, detail=detail)


def assert_dataset_accessible(payload: TokenPayload, dataset: Dataset) -> None:
    dataset_type = (
        dataset.type if isinstance(dataset.type, DatasetType) else DatasetType(str(dataset.type))
    )
    assert_storage_type_allowed(payload, dataset_type)


def require_storage_admin(
    payload: TokenPayload = Depends(require_auth),
) -> TokenPayload:
    if not is_storage_admin(payload):
        raise _forbidden("Admin role required to upload or manage datasets")
    return payload
