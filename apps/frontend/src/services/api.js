import { getAccessToken } from '../auth/keycloak';

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

async function extractApiErrorDetail(response) {
  const fallback = `API request failed with status ${response.status}`;
  const rawBody = await response.text();
  if (!rawBody) {
    return fallback;
  }

  try {
    const errorBody = JSON.parse(rawBody);
    const { detail } = errorBody ?? {};

    if (typeof detail === 'string') {
      return detail;
    }
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'msg' in item) return item.msg;
        return JSON.stringify(item);
      });
      if (messages.length > 0) {
        return messages.join('; ');
      }
    }
    if (detail != null) {
      return String(detail);
    }
    if (typeof errorBody?.message === 'string') {
      return errorBody.message;
    }
  } catch {
    return rawBody;
  }

  return fallback;
}

export const authorizedRequest = async (path, init = {}) => {
  const token = await getAccessToken();

  const headers = {
    Accept: 'application/json',
    ...(init.headers ?? {}),
    Authorization: `Bearer ${token}`,
    'Content-Type': 'application/json',
  };

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    const detail = await extractApiErrorDetail(response);
    throw new Error(detail);
  }

  return response.json();
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
