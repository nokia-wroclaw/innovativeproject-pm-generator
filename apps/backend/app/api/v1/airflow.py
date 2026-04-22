import requests
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import db_manager
from app.services.airflow import AirflowService
router = APIRouter()


def get_airflow_service(
    db: Session = Depends(db_manager.get_db),
) -> AirflowService:
    return AirflowService(db=db)


@router.get("/airflow_test")
def airflow_test(service: Depends = Depends(get_airflow_service)):
    response = requests.get('http://administrator-genpm-airflow-apiserver-1:9005/api/v2/monitor/health')
    print(response.json())
    return response.json()