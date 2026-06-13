# GenPM Spark DAGs

This folder holds the Airflow DAGs that run genpm Spark jobs. The backend (FastAPI) triggers them
over the Airflow REST API with a `dag_run.conf` payload; the DAG forwards that conf to the Spark job
as a single `--conf-json` argument.

## Architecture

```
backend ──REST trigger(conf)──▶ Airflow DAG
                                  prepare_conf (PythonOperator)
                                    • rebuild genpm.zip from the live mount (py_files)
                                    • validate + apply defaults to conf
                                    • push finalized conf to XCom as JSON
                                          │
                                          ▼
                                  run_<dag> (SparkSubmitOperator)
                                    • submits to spark://<user>-genpm-spark:7077 (conn_id)
                                    • application_args: [<subcmd>...] --conf-json <json>
                                          │
                                          ▼
                                  genpm Spark job (reads/writes MinIO s3a)
```

Key design points:

- **One execution model.** Resource sizing (master, memory, cores) comes from `spark-submit` via
  the `SparkSubmitOperator` `conf=` built in `lib/spark_config.py:spark_submit_conf()`. The genpm
  `SparkDataManager` respects that master and does **not** force `local[N]` (it only falls back to
  local for bare-CLI/notebook runs). Resource presets live once in
  `genpm.utils.consts.SPARK_CONFIGS`.
- **One parameter schema.** Defaults + validation for preprocessing live in
  `genpm.preprocessing.defaults`. The backend keeps a parallel Pydantic model kept honest by
  `apps/backend/tests/test_spark_jobs_contract.py`.
- **Fresh code on executors.** `lib/spark_submit.py:rebuild_py_files()` rebuilds `genpm.zip` from the
  live `GENPM_GENERATOR_ROOT` mount at submit time so the driver and cluster executors run identical
  code (the build-time zip in the image is only a fallback).
- **Fail loud on misconfig.** `lib/spark_config.py` reads infra from the environment and raises a
  clear error when a required var (`GENPM_SPARK_EXECUTOR_PYTHON`, `GENPM_SCHEMA_PATH`, AWS creds) is
  missing — so a wrong-user / misconfigured stack surfaces immediately.

## Adding a new DAG

1. Add the job entrypoint in genpm (e.g. `genpm/<area>/__main__.py`) that accepts `--conf-json` and
   builds its config via a `from_conf(conf, bucket=...)` classmethod (see
   `genpm.preprocessing.configs.PreprocessingConfig`).
2. Create `apps/airflow/dags/<name>_spark_dag.py`:

   ```python
   import os, sys
   from pathlib import Path

   _DAGS_ROOT = Path(__file__).resolve().parent
   for _p in (str(_DAGS_ROOT), os.environ.get("GENPM_GENERATOR_ROOT", "/opt/airflow/generator")):
       if _p and _p not in sys.path:
           sys.path.insert(0, _p)

   from lib.spark_config import GENPM_GENERATOR_ROOT  # noqa: E402
   from lib.spark_dag import build_spark_job_dag  # noqa: E402

   dag = build_spark_job_dag(
       dag_id="my_new_job",
       application=f"{GENPM_GENERATOR_ROOT}/genpm/<area>/__main__.py",
       app_name="MyNewSparkApp",
       command=["<subcommand>"],           # optional argv prefix before --conf-json
       conf_finalizer=my_finalizer,        # optional: validate/apply defaults
       spark_preset="HALF_SAFE",           # any key in SPARK_CONFIGS
       tags=["spark", "my-area"],
   )
   ```

3. Map it in the backend (`DAG_ID_MAP` / service DAG-id constants) and trigger it.

## Required environment (per-user Airflow stack)

Set in `infra/airflow-docker-compose.yml` (via repo-root `.env` + `USER`):

| Var | Purpose |
| --- | --- |
| `AIRFLOW_CONN_SPARK_DEFAULT` | `spark://<user>-genpm-spark:7077` — the Spark master |
| `GENPM_GENERATOR_ROOT` | genpm source mount (default `/opt/airflow/generator`) |
| `GENPM_PYSPARK_PYTHON` | driver python (Airflow genpm-venv) |
| `GENPM_SPARK_EXECUTOR_PYTHON` | executor python in the `<user>-genpm-spark` container venv |
| `GENPM_SCHEMA_PATH` | shared PM schema JSON path |
| `S3_URL` / `S3_BUCKET` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | MinIO access |
