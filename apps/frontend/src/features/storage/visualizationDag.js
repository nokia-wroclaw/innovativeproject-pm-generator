/** Airflow DAG used for RAW dataset visualizations. */
export const DATASET_VISUALIZATION_DAG_ID = 'dataset_visualization_spark';

export const DATASET_VISUALIZATION_DAG_FALLBACK = {
  displayName: 'Dataset visualization (Spark)',
  description:
    'Reads RAW PM parquet from S3, writes summary.json and kpi_analysis.json.',
  tags: ['spark', 'visualization'],
};

/** @param {string} [dagId] @param {string | null} [runId] */
export function datasetVisualizationDagPath(dagId = DATASET_VISUALIZATION_DAG_ID, runId = null) {
  const base = `/dags/${encodeURIComponent(dagId)}`;
  if (!runId) return base;
  return `${base}?run=${encodeURIComponent(runId)}`;
}
