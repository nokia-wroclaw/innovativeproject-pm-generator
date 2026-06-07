<script setup>
import { computed } from 'vue';
import { RefreshCw } from 'lucide-vue-next';
import vizLoadingGif from '@/assets/images/viz-loading-nokia.gif';
import { useDatasetVisualizationPoll } from '../composables/useDatasetVisualizationPoll';
import { validatePmColumns } from '../pmSchema';
import SummaryMetricsWidget from './SummaryMetricsWidget.vue';
import CoverageHeatmapWidget from './CoverageHeatmapWidget.vue';
import KpiCatalogTable from './KpiCatalogTable.vue';
import SchemaTableWidget from './SchemaTableWidget.vue';
import KpiTimelineWidget from './KpiTimelineWidget.vue';
import VisualizationDagLink from './VisualizationDagLink.vue';
import { DATASET_VISUALIZATION_DAG_ID } from '../visualizationDag.js';

const props = defineProps({
  datasetId: {
    type: Number,
    required: true,
  },
  previewColumns: {
    type: Array,
    default: () => [],
  },
});

const datasetIdRef = computed(() => props.datasetId);

const previewSchema = computed(() => validatePmColumns(props.previewColumns));

const pollingEnabled = computed(() => previewSchema.value.ok);

const {
  status,
  isPolling,
  pollError,
  pollTimedOut,
  pipelineTimedOut,
  isRetrying,
  retryError,
  retryVisualization,
} = useDatasetVisualizationPoll(datasetIdRef, { enabled: pollingEnabled });

const previewUnsupported = computed(
  () => !previewSchema.value.ok && props.previewColumns.length > 0,
);

const apiUnsupported = computed(
  () => status.value?.status === 'unsupported_schema',
);

const showUnsupported = computed(
  () => previewUnsupported.value || apiUnsupported.value,
);

const unsupportedSummary = computed(() => {
  if (previewUnsupported.value) {
    return previewSchema.value.summary;
  }
  return status.value?.summary ?? previewSchema.value.summary;
});

const unsupportedMessage = computed(
  () =>
    unsupportedSummary.value?.message ||
    status.value?.message ||
    previewSchema.value.message,
);

const summary = computed(() => status.value?.summary ?? null);

const showResults = computed(
  () => status.value?.status === 'success' && summary.value,
);

const coverageData = computed(
  () =>
    summary.value?.kpi_bts_coverage ??
    summary.value?.kpi_bts_coverage_heatmap ??
    null,
);

const schemaRows = computed(() => summary.value?.schema ?? []);

const kpiCatalogRows = computed(() => summary.value?.kpi_catalog ?? []);

const kpiAnalysis = computed(() => status.value?.kpi_analysis ?? null);

const visualizationDagId = computed(
  () => status.value?.dag_id ?? DATASET_VISUALIZATION_DAG_ID,
);

const visualizationRunId = computed(() => status.value?.run_id ?? null);

const hasCoverageHeatmap = computed(() => {
  const c = coverageData.value;
  if (!c) return false;
  if (typeof c === 'string') return true;
  if (c.z?.length) return true;
  if (c.data?.length) return true;
  return false;
});

const showLoading = computed(
  () =>
    pollingEnabled.value &&
    isPolling.value &&
    !pollTimedOut.value &&
    !pipelineTimedOut.value &&
    !showResults.value &&
    !apiUnsupported.value &&
    status.value?.status !== 'failed' &&
    status.value?.status !== 'unavailable',
);

const loadingMessage = computed(() => {
  const current = status.value?.status;
  const apiMessage = status.value?.message;

  if (current === 'queued') {
    return apiMessage || 'Visualization queued in Airflow…';
  }
  if (current === 'running') {
    return apiMessage || 'Running Spark visualization pipeline…';
  }
  if (current === 'not_found') {
    return (
      apiMessage ||
      'Waiting for visualization run to start (auto-trigger after RAW upload)…'
    );
  }
  return 'Checking visualization status…';
});

const showRetry = computed(() => {
  if (showResults.value || showUnsupported.value) {
    return false;
  }
  return (
    pollTimedOut.value ||
    pipelineTimedOut.value ||
    status.value?.status === 'failed' ||
    status.value?.status === 'unavailable' ||
    status.value?.status === 'not_found' ||
    Boolean(pollError.value)
  );
});

const problemTitle = computed(() => {
  if (pollError.value) {
    return 'Could not load visualization status';
  }
  if (status.value?.status === 'unavailable') {
    return 'Visualization service unavailable';
  }
  if (status.value?.status === 'failed') {
    return 'Visualization failed';
  }
  if (pipelineTimedOut.value) {
    return 'Visualization is taking too long';
  }
  if (pollTimedOut.value) {
    return 'Visualization did not start';
  }
  return 'Visualization not available';
});

const problemMessage = computed(() => {
  if (pollError.value) {
    return pollError.value;
  }
  if (pipelineTimedOut.value) {
    return (
      status.value?.message ||
      'The pipeline has been running longer than expected. The DAG may be stuck, or Spark may still be processing a large dataset.'
    );
  }
  if (pollTimedOut.value) {
    return (
      status.value?.message ||
      'No Airflow run was found for this dataset. Auto-trigger may have failed, or the DAG was never scheduled.'
    );
  }
  return (
    status.value?.message ||
    'No visualization run found for this dataset.'
  );
});
</script>

<template>
  <section class="s3-dataset-viz">
    <h2 class="s3-dataset-viz-title">Visualizations</h2>

    <div v-if="showUnsupported" class="s3-dataset-viz-unsupported">
      <p class="s3-dataset-viz-unsupported-title">Data not compatible with PM visualizations</p>
      <p class="s3-dataset-viz-unsupported-lead">
        This file does not match the Performance Management (PM) schema. Visualizations
        require KPI/BTS time-series columns, not generic tabular data like
        <code>dummy.parquet</code>.
      </p>
      <p>{{ unsupportedMessage }}</p>
      <p
        v-if="unsupportedSummary?.present_columns?.length"
        class="s3-dataset-viz-unsupported-meta"
      >
        Found columns:
        <code>{{ unsupportedSummary.present_columns.join(', ') }}</code>
      </p>
      <p v-if="unsupportedSummary?.missing_columns?.length" class="s3-dataset-viz-unsupported-meta">
        Missing columns:
        <code>{{ unsupportedSummary.missing_columns.join(', ') }}</code>
      </p>
      <p v-if="unsupportedSummary?.required_columns?.length" class="s3-dataset-viz-unsupported-meta">
        Required PM columns:
        <code>{{ unsupportedSummary.required_columns.join(', ') }}</code>
      </p>
      <p v-if="status?.spark_version || unsupportedSummary?.spark_version" class="s3-dataset-viz-unsupported-meta">
        Spark: {{ status.spark_version || unsupportedSummary.spark_version }}
      </p>
      <p v-if="status?.run_id" class="s3-dataset-viz-run-meta">
        Previous Airflow run: <code>{{ status.run_id }}</code>
      </p>
    </div>

    <div v-else-if="showLoading" class="s3-dataset-viz-loading">
      <VisualizationDagLink
        :dag-id="visualizationDagId"
        :run-id="visualizationRunId"
      />
      <img
        :src="vizLoadingGif"
        alt=""
        class="s3-dataset-viz-loading-gif"
        width="240"
        height="240"
      />
      <p class="s3-dataset-viz-loading-message">{{ loadingMessage }}</p>
    </div>

    <div
      v-else-if="showRetry || status?.status === 'failed' || status?.status === 'unavailable'"
      class="s3-dataset-viz-problem"
    >
      <VisualizationDagLink
        :dag-id="visualizationDagId"
        :run-id="visualizationRunId"
        compact
      />
      <p class="s3-dataset-viz-problem-title">{{ problemTitle }}</p>
      <p class="s3-dataset-viz-problem-message">{{ problemMessage }}</p>
      <p v-if="retryError" class="s3-dataset-viz-retry-error">{{ retryError }}</p>
      <div v-if="showRetry" class="s3-dataset-viz-actions">
        <button
          type="button"
          class="s3-dataset-viz-retry-btn"
          :disabled="isRetrying"
          @click="retryVisualization"
        >
          <RefreshCw :size="16" :class="{ 's3-dataset-viz-retry-spin': isRetrying }" />
          {{ isRetrying ? 'Starting…' : 'Retry visualization' }}
        </button>
      </div>
    </div>

    <template v-else-if="showResults">
      <VisualizationDagLink
        :dag-id="visualizationDagId"
        :run-id="visualizationRunId"
        compact
      />

      <SummaryMetricsWidget
        :basic-info="summary.basic_info"
        :spark-version="status.spark_version || summary.spark_version"
      />

      <div v-if="schemaRows.length" class="s3-dataset-viz-block">
        <h3 class="s3-dataset-viz-subtitle">Dataset schema</h3>
        <SchemaTableWidget :rows="schemaRows" />
      </div>

      <p v-if="summary.coverage_warning" class="s3-dataset-viz-warning">
        {{ summary.coverage_warning }}
      </p>
      <p v-if="summary.catalog_warning" class="s3-dataset-viz-warning">
        {{ summary.catalog_warning }}
      </p>

      <div v-if="hasCoverageHeatmap" class="s3-dataset-viz-block">
        <h3 class="s3-dataset-viz-subtitle">KPI coverage per BTS</h3>
        <CoverageHeatmapWidget :coverage="coverageData" :height="800" />
      </div>

      <div class="s3-dataset-viz-block">
        <h3 class="s3-dataset-viz-subtitle">KPI timeline &amp; distribution</h3>
        <KpiTimelineWidget :analysis="kpiAnalysis" />
      </div>

      <div class="s3-dataset-viz-block">
        <h3 class="s3-dataset-viz-subtitle">KPI catalog</h3>
        <KpiCatalogTable :rows="kpiCatalogRows" />
      </div>

      <div class="s3-dataset-viz-actions">
        <button
          type="button"
          class="s3-dataset-viz-retry-btn s3-dataset-viz-retry-btn--secondary"
          :disabled="isRetrying"
          @click="retryVisualization"
        >
          <RefreshCw :size="16" :class="{ 's3-dataset-viz-retry-spin': isRetrying }" />
          {{ isRetrying ? 'Starting…' : 'Re-run visualization' }}
        </button>
      </div>
    </template>

    <p v-else class="s3-dataset-viz-placeholder">
      Visualization results will appear here.
    </p>
  </section>
</template>
