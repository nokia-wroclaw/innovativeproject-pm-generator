import { getAccessToken } from '../auth/keycloak';

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

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
  const token = await getAccessToken();

  const headers = {
    Accept: 'application/json',
    ...(init.headers ?? {}),
    Authorization: `Bearer ${token}`,
  };
  if (init.body != null && !(init.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers,
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
