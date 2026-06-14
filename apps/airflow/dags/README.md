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
- **genpm distribution = pinned wheel (hybrid).** Both images build and install the genpm wheel
  (`uv build --wheel` → `uv pip install --no-deps`), so the Spark driver and executors run the exact
  same versioned `genpm` — no editable/snapshot skew. The `application` for `SparkSubmitOperator` is a
  stable launcher under `<generator>/apps/run_*.py` that just calls the package `__main__.main()`.
  - **Dev override (fast iteration):** the dev compose sets `GENPM_PY_FILES` and mounts the source, so
    `lib/spark_submit.py:rebuild_py_files()` ships the live code to executors via `--py-files` and the
    driver picks it up via `GENPM_GENERATOR_ROOT` on `PYTHONPATH` — no image rebuild per change.
  - **Prod:** `GENPM_PY_FILES` unset → no `--py-files`; the installed wheel is used everywhere.
- **Fail loud on misconfig.** `lib/spark_config.py` reads infra from the environment and raises a
  clear error when a required var (`GENPM_SPARK_EXECUTOR_PYTHON`, `GENPM_SCHEMA_PATH`, AWS creds) is
  missing — so a wrong-user / misconfigured stack surfaces immediately.

## Adding a new DAG

1. Add the job entrypoint in genpm (e.g. `genpm/<area>/__main__.py`) that accepts `--conf-json` and
   builds its config via a `from_conf(conf, bucket=...)` classmethod (see
   `genpm.preprocessing.configs.PreprocessingConfig`).
2. Add a launcher `apps/generator/apps/run_<area>.py` (two lines: import the package `main` and call
   it). This is the `application` file the operator submits; it works with the wheel or a dev mount.
3. Create `apps/airflow/dags/<name>_spark_dag.py`:

   ```python
   import os, sys
   from pathlib import Path

   _DAGS_ROOT = Path(__file__).resolve().parent
   for _p in (str(_DAGS_ROOT), os.environ.get("GENPM_GENERATOR_ROOT", "/opt/airflow/generator")):
       if _p and _p not in sys.path:
           sys.path.insert(0, _p)

   from lib.spark_config import GENPM_SPARK_APPS_DIR  # noqa: E402
   from lib.spark_dag import build_spark_job_dag  # noqa: E402

   dag = build_spark_job_dag(
       dag_id="my_new_job",
       application=f"{GENPM_SPARK_APPS_DIR}/run_<area>.py",
       app_name="MyNewSparkApp",
       command=["<subcommand>"],           # optional argv prefix before --conf-json
       conf_finalizer=my_finalizer,        # optional: validate/apply defaults
       spark_preset="HALF_SAFE",           # any key in SPARK_CONFIGS
       tags=["spark", "my-area"],
   )
   ```

4. Map it in the backend (`DAG_ID_MAP` / service DAG-id constants) and trigger it.

## Building / releasing the genpm wheel

`cd apps/generator && uv build` produces `dist/genpm_generator-<version>-py3-none-any.whl`. The
Airflow and Spark images build and install this wheel at image-build time (pinned by the `version`
in `apps/generator/pyproject.toml` — bump it per release). To distribute a new genpm version to a
real cluster, bump the version and rebuild both images.

## Required environment (per-user Airflow stack)

Set in `infra/airflow-docker-compose.yml` (via repo-root `.env` + `USER`):

| Var | Purpose |
| --- | --- |
| `AIRFLOW_CONN_SPARK_DEFAULT` | `spark://<user>-genpm-spark:7077` — the Spark master |
| `GENPM_GENERATOR_ROOT` | genpm source mount (default `/opt/airflow/generator`) |
| `GENPM_PYSPARK_PYTHON` | driver python (Airflow genpm-venv) |
| `GENPM_SPARK_EXECUTOR_PYTHON` | executor python in the `<user>-genpm-spark` container venv |
| `GENPM_SCHEMA_PATH` | shared PM schema JSON path |
| `GENPM_SPARK_APPS_DIR` | launcher dir (default `<generator>/apps`) |
| `GENPM_PY_FILES` | **dev only**: set → ship live genpm to executors via `--py-files`; unset in prod (wheel) |
| `S3_URL` / `S3_BUCKET` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | MinIO access |
