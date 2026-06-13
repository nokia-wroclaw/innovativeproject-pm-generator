import { authorizedRequest } from './api.js';

export const fetchS3DatasetsPage = async ({ page, limit, type }) => {
  if (!type) {
    throw new Error('Dataset type is required');
  }
  const datasets = await authorizedRequest(
    `/api/v1/datasets?type=${encodeURIComponent(type)}`,
  );
  const safeDatasets = Array.isArray(datasets) ? datasets : [];
  const perPage = Number(limit) > 0 ? Number(limit) : 10;
  const currentPage = Number(page) > 0 ? Number(page) : 1;
  const start = (currentPage - 1) * perPage;
  const end = start + perPage;

  return {
    data: safeDatasets.slice(start, end).map((dataset) => ({
      id: dataset.id,
      file_name: dataset.file_name,
      s3_key: dataset.s3_key,
      status: dataset.status,
      type: dataset.type,
    })),
    total: safeDatasets.length,
  };
};

export const createS3Dataset = async (datasetData) => {
  return await authorizedRequest('/api/v1/datasets', {
    method: 'POST',
    body: JSON.stringify(datasetData),
  });
};

export const DatasetStatus = {
    PENDING: "PENDING",
    UPLOADING: "UPLOADING",
    COMPLETED: "COMPLETED",
    FAILED: "FAILED",
}

export const DatasetType = {
    RAW: "RAW",
    PREPROCESSED: "PREPROCESSED",
    GENERATED: "GENERATED",
    KPI_DEFINITIONS: "KPI_DEFINITIONS",
    SIMPLE_REPORTS: "SIMPLE_REPORTS",
}

export const updateS3Status = async (datasetId, status) => {
  return await authorizedRequest('/api/v1/datasets/update_status', {
    method: 'POST',
    body: JSON.stringify({
      dataset_id: datasetId,
      status: status
    }),
  });
};
export const initiateMultipartUpload = async (datasetId) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/multipart/initiate`, {
    method: 'POST'
  });
};

export const getPresignedPartUrl = async (datasetId, uploadId, partNumber) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/multipart/part-url?upload_id=${uploadId}&part_number=${partNumber}`, {
    method: 'GET'
  });
};

export const completeMultipartUpload = async (datasetId, uploadId, parts) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/multipart/complete`, {
    method: 'POST',
    body: JSON.stringify({ upload_id: uploadId, parts: parts })
  });
};

export const abortMultipartUpload = async (datasetId, uploadId) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/multipart/abort`, {
    method: 'POST',
    body: JSON.stringify({ upload_id: uploadId })
  });
};
export const registerExistingS3Dataset = async (datasetData) => {
  return await authorizedRequest('/api/v1/datasets/register', {
    method: 'POST',
    body: JSON.stringify(datasetData),
  });
};


export const deleteS3Dataset = async (datasetId, { deleteFromS3 = false } = {}) => {
  const params = new URLSearchParams();
  if (deleteFromS3) {
    params.set('delete_from_s3', 'true');
  }
  const query = params.toString();
  const path = `/api/v1/datasets/${datasetId}${query ? `?${query}` : ''}`;
  return await authorizedRequest(path, {
    method: 'DELETE',
  });
};

export const fetchDatasetPreview = async (datasetId) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/preview`);
};

export const fetchDatasetVisualizationStatus = async (datasetId) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/visualization/status`);
};

export const requestDatasetVisualization = async (datasetId) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/visualization`, {
    method: 'POST',
  });
};

export const fetchS3ModelsPage = async ({ page, limit }) => {
  const models = await authorizedRequest('/api/v1/modeling/models');
  const safeModels = Array.isArray(models) ? models : [];
  const perPage = Number(limit) > 0 ? Number(limit) : 10;
  const currentPage = Number(page) > 0 ? Number(page) : 1;
  const start = (currentPage - 1) * perPage;
  const end = start + perPage;

  return {
    data: safeModels.slice(start, end).map((model) => ({
      id: model.id,
      name: model.name,
      s3_key: model.path ? model.path.replace(/^s3:\/\/[^\/]+\//, '') : '',
      path: model.path,
      encoder_s3_key: model.encoder_s3_key,
      config_s3_key: model.config_s3_key,
      dataset_id: model.dataset_id,
    })),
    total: safeModels.length,
  };
};

export const deleteS3Model = async (modelId, { deleteFromS3 = false } = {}) => {
  return await authorizedRequest(`/api/v1/modeling/models/${modelId}?delete_from_s3=${Boolean(deleteFromS3)}`, {
    method: 'DELETE',
  });
};

export const updateS3Model = async (modelId, modelData) => {
  return await authorizedRequest(`/api/v1/modeling/models/${modelId}`, {
    method: 'PATCH',
    body: JSON.stringify(modelData),
  });
};


