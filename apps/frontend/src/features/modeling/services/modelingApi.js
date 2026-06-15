import { authorizedRequest } from '@/services/api';

const BASE = '/api/v1/modeling';
const enc = encodeURIComponent;
const FORM_SCHEMA_TIMEOUT_MS = 10_000;

export const listModelingDatasets = () =>
  authorizedRequest(`${BASE}/datasets`);

export const listModelingModels = () =>
  authorizedRequest(`${BASE}/models`);

export const createTrainedModel = (body) =>
  authorizedRequest(`${BASE}/models`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const getModelingFormSchema = (processType, options = {}) =>
  authorizedRequest(`${BASE}/processes/${enc(processType)}/form-schema`, {
    timeoutMs: FORM_SCHEMA_TIMEOUT_MS,
    ...options,
  });

export const triggerModelingRun = (processType, body) =>
  authorizedRequest(`${BASE}/processes/${enc(processType)}/runs`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const getModelingRunStatus = (processType, runId) =>
  authorizedRequest(`${BASE}/processes/${enc(processType)}/runs/${enc(runId)}`);

export const updateTrainedModel = (modelId, body) =>
  authorizedRequest(`${BASE}/models/${enc(modelId)}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  });

export const deleteTrainedModel = (modelId, deleteFromS3 = false) =>
  authorizedRequest(`${BASE}/models/${enc(modelId)}?delete_from_s3=${Boolean(deleteFromS3)}`, {
    method: 'DELETE',
  });

export const fetchModelKpis = (modelId) =>
  authorizedRequest(`${BASE}/models/${enc(modelId)}/kpis`);

export const fetchModelCells = (modelId) =>
  authorizedRequest(`${BASE}/models/${enc(modelId)}/cells`);
