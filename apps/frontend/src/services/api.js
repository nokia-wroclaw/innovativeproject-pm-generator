import { getAccessToken } from '../auth/keycloak';

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Error thrown for any non-2xx response from our backend. Carries the stable
 * machine code from the contract (see docs/architecture §6), the request id,
 * and the raw HTTP status — letting UI code branch sensibly without parsing
 * strings.
 */
export class ApiError extends Error {
  constructor({ code, message, status, requestId, details }) {
    super(message || `API request failed with status ${status}`);
    this.name = 'ApiError';
    this.code = code || 'INTERNAL_ERROR';
    this.status = status;
    this.requestId = requestId || null;
    this.details = details || null;
  }
}

function createTimeoutError(timeoutMs) {
  return new ApiError({
    code: 'REQUEST_TIMEOUT',
    status: 0,
    message: `Request timed out after ${Math.round(timeoutMs / 1000)}s`,
  });
}

async function parseErrorBody(response) {
  const status = response.status;
  const requestId = response.headers.get('X-Request-ID');
  const rawBody = await response.text();
  if (!rawBody) {
    return new ApiError({ status, requestId, message: `API ${status}` });
  }

  let body;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return new ApiError({ status, requestId, message: rawBody });
  }

  // New envelope shape from backend (ApiError).
  if (body && typeof body === 'object' && typeof body.error === 'string') {
    return new ApiError({
      code: body.error,
      message: body.message ?? rawBody,
      status,
      requestId: body.request_id ?? requestId,
      details: body.details ?? null,
    });
  }

  // Legacy FastAPI HTTPException shape: { detail: "..." } or { detail: [...] }.
  let message = `API ${status}`;
  if (typeof body?.detail === 'string') {
    message = body.detail;
  } else if (Array.isArray(body?.detail)) {
    message = body.detail
      .map((item) => (typeof item === 'string' ? item : item?.msg ?? JSON.stringify(item)))
      .join('; ');
  } else if (typeof body?.message === 'string') {
    message = body.message;
  }
  return new ApiError({ status, requestId, message });
}

/**
 * @param {string} path  path under apiBaseUrl, e.g. "/api/v1/dags"
 * @param {RequestInit} [init]
 * @returns {Promise<any>} parsed JSON body
 * @throws {ApiError}
 */
export const authorizedRequest = async (path, init = {}) => {
  const {
    timeoutMs = DEFAULT_TIMEOUT_MS,
    signal: externalSignal,
    ...fetchInit
  } = init;
  const controller = new AbortController();
  let timeoutId = null;

  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener('abort', () => controller.abort(), { once: true });
    }
  }

  const request = async () => {
    const token = await getAccessToken();

    const headers = {
      Accept: 'application/json',
      ...(fetchInit.headers ?? {}),
      Authorization: `Bearer ${token}`,
    };
    if (fetchInit.body != null && !(fetchInit.body instanceof FormData) && !headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${apiBaseUrl}${path}`, {
      ...fetchInit,
      headers,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw await parseErrorBody(response);
    }

    if (response.status === 204) return null;
    const contentType = response.headers.get('Content-Type') ?? '';
    if (!contentType.includes('application/json')) {
      return response.text();
    }
    return response.json();
  };

  const timeout =
    timeoutMs > 0
      ? new Promise((_, reject) => {
        timeoutId = window.setTimeout(() => {
          controller.abort();
          reject(createTimeoutError(timeoutMs));
        }, timeoutMs);
      })
      : null;

  try {
    return await (timeout ? Promise.race([request(), timeout]) : request());
  } finally {
    if (timeoutId !== null) {
      window.clearTimeout(timeoutId);
    }
  }
};

/** Builds the absolute URL for an SSE EventSource (token must be in query). */
export const buildSseUrl = (path, params = {}) => {
  const url = new URL(`${apiBaseUrl}${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v != null) url.searchParams.set(k, String(v));
  }
  return url.toString();
};

/** Vue Query helper — unwrap ApiError message for toasts/badges. */
export const formatApiError = (error) => {
  if (error instanceof ApiError) {
    return `${error.message} (code=${error.code}${error.requestId ? `, req=${error.requestId}` : ''})`;
  }
  return error?.message ?? 'Unknown error';
};

// ─────────────────────────────────────────────────────────────────────────────
// Legacy helper kept for existing screens. Will be removed once the consumer
// (Dashboard.vue) is migrated.
// ─────────────────────────────────────────────────────────────────────────────
export const fetchGenerationPage = async ({ page, limit }) => {
  const generations = await authorizedRequest('/api/v1/test');
  const safeGenerations = Array.isArray(generations) ? generations : [];

  const perPage = Number(limit) > 0 ? Number(limit) : 10;
  const currentPage = Number(page) > 0 ? Number(page) : 1;
  const start = (currentPage - 1) * perPage;
  const end = start + perPage;

  return {
    data: safeGenerations.slice(start, end).map((generation) => ({
      id: generation.id,
      name: generation.name,
      number: generation.number,
    })),
    total: safeGenerations.length,
  };
};
