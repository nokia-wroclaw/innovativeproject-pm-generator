"""Async HTTP client for the Airflow REST API v2.

This module owns the raw HTTP layer: connection pooling, timeouts, retries,
and authentication header injection. It returns *raw* JSON payloads — domain
mapping into our DTOs lives in :mod:`app.integrations.airflow.mapper`.

The client is created once at app startup and disposed in lifespan teardown
(see ``app/main.py``).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .auth import AirflowAuth
from .config import AirflowSettings
from .errors import (
    AirflowAuthFailed,
    AirflowConflict,
    AirflowIntegrationError,
    AirflowNotFound,
    AirflowUnavailable,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class AirflowClient:
    """Thin wrapper around ``httpx.AsyncClient`` for Airflow API v2."""

    def __init__(self, settings: AirflowSettings, auth: AirflowAuth) -> None:
        self._settings = settings
        self._auth = auth
        self._client: httpx.AsyncClient | None = None

    # ─── Lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._client is not None:
            return
        timeout = httpx.Timeout(self._settings.http_timeout_seconds)
        self._client = httpx.AsyncClient(
            base_url=self._settings.api_base,
            timeout=timeout,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ─── Public methods (one per contract endpoint) ───────────────────────
    async def list_dags(
        self, *, limit: int = 200, offset: int = 0, only_active: bool = True
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if only_active:
            params["only_active"] = "true"
        return await self._request("GET", "/dags", params=params)

    async def get_dag(self, dag_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/dags/{dag_id}")

    async def get_dag_details(self, dag_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/dags/{dag_id}/details")

    async def list_dag_tasks(self, dag_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/dags/{dag_id}/tasks")

    async def list_dag_runs(
        self,
        dag_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "-start_date",
        start_date_gte: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset, "order_by": order_by}
        if start_date_gte is not None:
            params["start_date_gte"] = start_date_gte
        return await self._request("GET", f"/dags/{dag_id}/dagRuns", params=params)

    async def get_dag_run(self, dag_id: str, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/dags/{dag_id}/dagRuns/{run_id}")

    async def list_task_instances(self, dag_id: str, run_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
        )

    async def get_task_instance(
        self, dag_id: str, run_id: str, task_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "GET", f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}"
        )

    async def list_task_tries(
        self, dag_id: str, run_id: str, task_id: str
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/tries",
        )

    async def get_task_logs(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        try_number: int,
        *,
        token: str | None = None,
        full_content: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if token is not None:
            params["token"] = token
        if full_content:
            params["full_content"] = "true"
        return await self._request(
            "GET",
            f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}",
            params=params,
            accept="application/json",
        )

    async def trigger_dag(
        self,
        dag_id: str,
        *,
        conf: dict[str, Any] | None = None,
        logical_date: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if conf is not None:
            body["conf"] = conf
        if logical_date is not None:
            body["logical_date"] = logical_date
        if note is not None:
            body["note"] = note
        return await self._request(
            "POST", f"/dags/{dag_id}/dagRuns", json=body or None
        )

    async def patch_dag_run_state(
        self, dag_id: str, run_id: str, *, state: str, note: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state}
        if note is not None:
            body["note"] = note
        return await self._request(
            "PATCH",
            f"/dags/{dag_id}/dagRuns/{run_id}",
            json=body,
            params={"update_mask": "state"},
        )

    async def clear_task_instances(
        self,
        dag_id: str,
        run_id: str,
        *,
        task_ids: list[str] | None = None,
        include_downstream: bool = False,
        include_upstream: bool = False,
        only_failed: bool = False,
        reset_dag_runs: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "dry_run": False,
            "reset_dag_runs": reset_dag_runs,
            "only_failed": only_failed,
            "include_downstream": include_downstream,
            "include_upstream": include_upstream,
            "dag_run_id": run_id,
        }
        if task_ids:
            body["task_ids"] = task_ids
        return await self._request(
            "POST", f"/dags/{dag_id}/clearTaskInstances", json=body
        )

    async def healthcheck(self) -> dict[str, Any]:
        """Hits Airflow's monitor health endpoint (no auth required)."""
        if self._client is None:
            raise AirflowUnavailable("Airflow client not started")
        try:
            response = await self._client.get(
                f"{self._settings.base_url}/api/v2/monitor/health"
            )
        except httpx.HTTPError as exc:
            raise AirflowUnavailable(f"Cannot reach Airflow: {exc}") from exc
        if response.status_code >= 500:
            raise AirflowUnavailable(
                f"Airflow health returned {response.status_code}"
            )
        return response.json()

    # ─── Internals ────────────────────────────────────────────────────────
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        accept: str = "application/json",
    ) -> dict[str, Any]:
        if self._client is None:
            raise AirflowUnavailable("Airflow client not started")

        retrying = AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=4.0),
            retry=retry_if_exception_type((AirflowUnavailable, httpx.HTTPError)),
        )

        attempt = 0
        async for attempt_ctx in retrying:
            with attempt_ctx:
                attempt += 1
                response = await self._send(
                    method, path, params=params, json=json, accept=accept
                )
                if response.status_code in _RETRYABLE_STATUSES:
                    raise AirflowUnavailable(
                        f"Airflow returned {response.status_code} on {method} {path}"
                    )
                return self._handle_response(method, path, response)

        raise AirflowUnavailable(  # pragma: no cover - tenacity reraises before this
            f"Exhausted retries for {method} {path}"
        )

    async def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None,
        json: Any | None,
        accept: str,
    ) -> httpx.Response:
        assert self._client is not None
        headers = {
            "Authorization": f"Bearer {await self._auth.get_token()}",
            "Accept": accept,
        }
        try:
            response = await self._client.request(
                method, path, params=params, json=json, headers=headers
            )
        except httpx.HTTPError as exc:
            raise AirflowUnavailable(
                f"Network error talking to Airflow ({method} {path}): {exc}"
            ) from exc

        if response.status_code == 401:
            # Token may be stale (rotated secret, clock skew). Invalidate and
            # let tenacity retry — second attempt mints a fresh JWT.
            await self._auth.invalidate()
            raise AirflowAuthFailed(
                "Airflow rejected our service-account JWT (will retry once)"
            )

        return response

    def _handle_response(
        self, method: str, path: str, response: httpx.Response
    ) -> dict[str, Any]:
        status_code = response.status_code

        if 200 <= status_code < 300:
            if status_code == 204 or not response.content:
                return {}
            try:
                return response.json()
            except ValueError as exc:
                raise AirflowIntegrationError(
                    f"Airflow returned non-JSON 2xx body on {method} {path}"
                ) from exc

        if status_code == 404:
            raise AirflowNotFound(
                f"Airflow returned 404 on {method} {path}",
                details={"path": path},
            )
        if status_code == 409:
            raise AirflowConflict(
                f"Airflow returned 409 on {method} {path}",
                details=_safe_json(response),
            )
        if status_code == 401 or status_code == 403:
            raise AirflowAuthFailed(
                f"Airflow returned {status_code} on {method} {path}",
                details=_safe_json(response),
            )

        if status_code >= 500:
            raise AirflowUnavailable(
                f"Airflow 5xx ({status_code}) on {method} {path}",
                details=_safe_json(response),
            )

        raise AirflowIntegrationError(
            f"Unexpected Airflow status {status_code} on {method} {path}",
            details=_safe_json(response),
        )


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"raw": response.text[:500]}
    if isinstance(body, dict):
        return body
    return {"body": body}
