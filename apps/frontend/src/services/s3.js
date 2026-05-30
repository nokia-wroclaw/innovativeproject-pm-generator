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


export const deleteS3Dataset = async (datasetID) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetID}`, {
    method: 'DELETE',
  });
};

export const fetchDatasetPreview = async (datasetId) => {
  return await authorizedRequest(`/api/v1/datasets/${datasetId}/preview`);
};
