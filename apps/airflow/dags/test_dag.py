from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def wypisz_powitanie():
    print("Dag dziala tu")
    return "Finito praca zakonczona."

default_args = {
    'owner': 'macsko6154',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='moj_pierwszy_dag',
    default_args=default_args,
    description='Prosty testowy DAG',
    schedule=timedelta(days=1),
    start_date=datetime(2023, 10, 1),
    catchup=False,
    tags=['test', 'nauka'],
) as dag:



    zadanie_bash = BashOperator(
        task_id='wypisz_date_w_bashu',
        bash_command='date',
    )


    zadanie_python = PythonOperator(
        task_id='uruchom_funkcje_python',
        python_callable=wypisz_powitanie,
    )


    zadanie_czekaj = BashOperator(
        task_id='odczekaj_5_sekund',
        bash_command='sleep 5',
    )



    zadanie_bash >> zadanie_czekaj >> zadanie_python