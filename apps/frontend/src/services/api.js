import { getAccessToken } from '../auth/keycloak';

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');

const authorizedRequest = async (path, init = {}) => {
  const token = await getAccessToken();

  const headers = {
    ...(init.headers ?? {}),
    Authorization: `Bearer ${token}`,
  };

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    throw new Error(`API request failed with status ${response.status}`);
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
