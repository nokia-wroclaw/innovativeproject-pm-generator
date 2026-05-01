import { authorizedRequest } from './api.js';

export const fetchS3DatasetsPage = async ({ page, limit }) => {
  const datasets = await authorizedRequest('/api/v1/datasets');
  const safeDatasets = Array.isArray(datasets) ? datasets : [];
  const perPage = Number(limit) > 0 ? Number(limit) : 10;
  const currentPage = Number(page) > 0 ? Number(page) : 1;
  const start = (currentPage - 1) * perPage;
  const end = start + perPage;

  return {
    data: safeDatasets.slice(start, end).map((dataset) => ({
      id: dataset.id,
      file_name: dataset.file_name,
      s3_bucket: dataset.s3_bucket,
      s3_key: dataset.s3_key,
      status: dataset.status,
    })),
    total: safeDatasets.length,
  };
};

export const createS3Dataset = async (datasetData) => {
  console.log(datasetData)
  return await authorizedRequest('/api/v1/datasets', {
    method: 'POST',
    body: JSON.stringify(datasetData),
  });

};