# PM Generator
Performance Management Generator project

## DAG management (Airflow proxy)

The web UI provides a modern, opinionated overlay on top of Apache Airflow 3.2.
Contract & data model: [`docs/architecture/dag-management.md`](docs/architecture/dag-management.md).

* Backend lives under `apps/backend/app/integrations/airflow/` (httpx async
  client, HS512 service-account JWT, retries with tenacity) and exposes
  REST + SSE under `/api/v1/dags/*`.
* Frontend lives under `apps/frontend/src/features/dags/` (Vue 3, Vue Flow
  + dagre auto-layout, Tailwind v4 + Reka UI, TanStack Vue Query for server
  state with adaptive polling).

After updating the repo for the first time (the backend has new dependencies),
re-build the FastAPI container so `httpx`, `tenacity` and `sse-starlette` are
pulled into the image:

```bash
docker compose -f infra/docker-compose.yml build fastapi
```

The frontend will install the new packages (`@tanstack/vue-query`, Tailwind v4,
Vue Flow, Reka UI, etc.) automatically on next `docker compose up frontend`.

## How to set up profile on VM:
sudo useradd -m -s /bin/bash \<login from PWR AD> <br>
sudo usermod -aG docker \<profile name> <br>
sudo chown -R \<profile name>:<profile name> /home/\<profile name>/.docker <br>

## How to route a docker prots through OUR port on vm
run `./tools/bin/start_env.sh` and paste the command filled with your ports

# How to login
username: admin
passwd: admin

## Devcontainer support (VS code users)
1. run `./tools/bin/start_env.sh` (run ONLY ONCE)
2. restart VS code instance and install Devcontainer vs-code extension
3. run in vs-code: `>Dev Containers: Rebuild and Reopen in Container`

### Dependencies
Backend FastAPI uses uv.

### How to add new dependencies to the project
# Runtime dependency:
docker compose exec fastapi sh -c "cd /app && uv add <package>"

# Dev dependency:
docker compose exec fastapi sh -c "cd /app && uv add --dev <package>"


### Ruff and MyPy scripts
Use simple scripts from bin:
- ./tools/bin/be_ruff.sh            # lint
- ./tools/bin/be_ruff.sh --fix      # lint + autofix
- ./tools/bin/be_mypy.sh
- ./tools/bin/be_verify.sh          # ruff + mypy
