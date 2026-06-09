import { getAccessToken } from '../auth/keycloak';

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const DEFAULT_TIMEOUT_MS = 30_000;

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

  if (body && typeof body === 'object' && typeof body.error === 'string') {
    return new ApiError({
      code: body.error,
      message: body.message ?? rawBody,
      status,
      requestId: body.request_id ?? requestId,
      details: body.details ?? null,
    });
  }

  let message = `API ${status}`;
  if (
    body?.detail &&
    typeof body.detail === 'object' &&
    !Array.isArray(body.detail)
  ) {
    const detail = body.detail;
    return new ApiError({
      code:
        detail.status === 'unsupported_schema'
          ? 'UNSUPPORTED_SCHEMA'
          : `HTTP_${status}`,
      message: detail.message ?? message,
      status,
      requestId,
      details: detail,
    });
  }
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
    let token;
    try {
      token = await getAccessToken();
    } catch (error) {
      throw new ApiError({
        code: 'UNAUTHENTICATED',
        status: 401,
        message: error?.message || 'Session expired. Please sign in again.',
      });
    }

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

export const buildSseUrl = (path, params = {}) => {
  const url = new URL(`${apiBaseUrl}${path}`);
  for (const [k, v] of Object.entries(params)) {
    if (v != null) url.searchParams.set(k, String(v));
  }
  return url.toString();
};

export const formatApiError = (error) => {
  if (error instanceof ApiError) {
    return `${error.message} (code=${error.code}${error.requestId ? `, req=${error.requestId}` : ''})`;
  }
  return error?.message ?? 'Unknown error';
};

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
