from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, get_current_context
from datetime import datetime, timedelta
from pprint import pformat


def model_validate(process_type: str) -> dict:
    common_fields = [
        {
            "name": "dataset_id",
            "label": "Dataset input",
            "type": "dataset_select",
            "required": True,
            "help": "Dataset ze statusem COMPLETED trackowany przez backend.",
        },
        {
            "name": "dataset_type",
            "label": "Typ zestawu danych",
            "type": "radio",
            "required": True,
            "default": "working_days",
            "options": [
                {"value": "working_days", "label": "Dni robocze"},
                {"value": "weekends", "label": "Weekendy"},
            ],
        },
        {
            "name": "target_s3_key",
            "label": "S3 key",
            "type": "text",
            "required": False,
            "default": "",
            "help": "Key w S3 przekazywany do DAG-a. DAG tylko go wypisuje.",
        },
        {
            "name": "dag_args",
            "label": "Dodatkowe argumenty DAG-a",
            "type": "json",
            "required": False,
            "default": {},
            "help": "Obiekt JSON przekazywany do conf.dag_args.",
        },
    ]
    training_fields = [
        {
            "name": "epochs",
            "label": "Epoki",
            "type": "integer",
            "required": True,
            "default": 10,
            "min": 1,
        },
        {
            "name": "batch_size",
            "label": "Batch Size",
            "type": "select",
            "required": True,
            "default": 32,
            "options": [
                {"value": 16, "label": "16"},
                {"value": 32, "label": "32"},
                {"value": 64, "label": "64"},
                {"value": 128, "label": "128"},
            ],
        },
        {
            "name": "learning_rate",
            "label": "Learning Rate",
            "type": "float",
            "required": True,
            "default": 0.001,
            "min": 0.000001,
            "max": 1,
            "step": 0.0001,
        },
    ]

    if process_type == "training_dataset":
        return {
            "process_type": process_type,
            "title": "Tworzenie datasetu treningowego",
            "fields": [*common_fields, *training_fields],
        }
    return {
        "process_type": "preprocessing_feature_engineering",
        "title": "Preprocessing + Feature Engineering",
        "fields": common_fields,
    }


def wypisz_powitanie():
    context = get_current_context()
    conf = context["dag_run"].conf or {}

    print("Dag dziala tu")
    print("Parametry przekazane do DAG-a:")
    print(pformat(conf))
    print(f"process_type={conf.get('process_type')}")
    print(f"dataset_id={conf.get('dataset_id')}")
    print(f"dataset_name={conf.get('dataset_name')}")
    print(f"source_s3_key={conf.get('s3_key')}")
    print(f"target_s3_key={conf.get('target_s3_key')}")
    print(f"dataset_type={conf.get('dataset_type')}")
    print(f"training={pformat(conf.get('training'))}")
    print(f"dag_args={pformat(conf.get('dag_args'))}")
    return "Finito praca zakonczona."


default_args = {
    "owner": "macsko6154",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="moj_pierwszy_dag",
    default_args=default_args,
    description="Prosty testowy DAG",
    schedule=timedelta(days=1),
    start_date=datetime(2023, 10, 1),
    catchup=False,
    tags=["test", "nauka"],
) as dag:
    zadanie_bash = BashOperator(
        task_id="wypisz_date_w_bashu",
        bash_command="date",
    )

    zadanie_python = PythonOperator(
        task_id="uruchom_funkcje_python",
        python_callable=wypisz_powitanie,
    )

    zadanie_czekaj = BashOperator(
        task_id="odczekaj_5_sekund",
        bash_command="sleep 5",
    )

    zadanie_bash >> zadanie_czekaj >> zadanie_python
