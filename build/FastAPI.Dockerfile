FROM python:3.13-alpine AS builder

WORKDIR /app

RUN apk add --no-cache gcc musl-dev postgresql-dev libffi-dev

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

COPY apps/backend/pyproject.toml apps/backend/uv.lock ./

RUN uv sync --frozen --dev

FROM python:3.13-alpine

WORKDIR /app

RUN apk add --no-cache libpq

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/
COPY --from=builder /app/.venv /app/.venv
COPY apps/backend/pyproject.toml apps/backend/uv.lock ./
COPY apps/backend/app ./app

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
