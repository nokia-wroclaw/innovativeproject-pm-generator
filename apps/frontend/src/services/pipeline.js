import { authorizedRequest } from './api.js';

export const PipelineType = {
  PREPROCESSING: 'PREPROCESSING',
  FEATURE_ENGINEERING: 'FEATURE_ENGINEERING',
  TRAINING: 'TRAINING',
};

export const PipelineRunStatus = {
  PENDING: 'PENDING',
  RUNNING: 'RUNNING',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
};

export const PIPELINE_TYPE_LABELS = {
  [PipelineType.PREPROCESSING]: 'Preprocessing',
  [PipelineType.FEATURE_ENGINEERING]: 'Feature Engineering',
  [PipelineType.TRAINING]: 'Training',
};

export const fetchPipelineRunsPage = async ({ page, limit }, typeFilter = null) => {
  const runs = await authorizedRequest('/api/v1/pipelines');
  const safeRuns = Array.isArray(runs) ? runs : [];
  const filtered = typeFilter ? safeRuns.filter((r) => r.pipeline_type === typeFilter) : safeRuns;
  const perPage = Number(limit) > 0 ? Number(limit) : 10;
  const currentPage = Number(page) > 0 ? Number(page) : 1;
  const start = (currentPage - 1) * perPage;
  const end = start + perPage;

  return {
    data: filtered.slice(start, end),
    total: filtered.length,
  };
};

export const createPipelineRun = async ({ dataset_id, pipeline_type }) => {
  return authorizedRequest('/api/v1/pipelines', {
    method: 'POST',
    body: JSON.stringify({ dataset_id, pipeline_type }),
  });
};

export const deletePipelineRun = async (runId) => {
  return authorizedRequest(`/api/v1/pipelines/${runId}`, {
    method: 'DELETE',
  });
};

export const fetchCompletedDatasets = async () => {
  const datasets = await authorizedRequest('/api/v1/datasets?type=RAW');
  return Array.isArray(datasets)
    ? datasets.filter((d) => d.status === 'COMPLETED')
    : [];
};
