from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, get_current_context
from datetime import datetime, timedelta
from pprint import pformat


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
    "owner": "macsko6154s",
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
