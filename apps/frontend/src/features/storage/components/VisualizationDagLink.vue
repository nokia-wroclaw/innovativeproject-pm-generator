<script setup>
import { computed } from 'vue';
import { RouterLink } from 'vue-router';
import { useDagDetails } from '@/features/dags/composables/queries.js';
import {
  DATASET_VISUALIZATION_DAG_FALLBACK,
  datasetVisualizationDagPath,
} from '../visualizationDag.js';

const props = defineProps({
  dagId: {
    type: String,
    required: true,
  },
  runId: {
    type: String,
    default: null,
  },
  compact: {
    type: Boolean,
    default: false,
  },
});

const dagIdRef = computed(() => props.dagId);
const detailsQuery = useDagDetails(dagIdRef);

const displayName = computed(
  () =>
    detailsQuery.data.value?.summary?.display_name ??
    DATASET_VISUALIZATION_DAG_FALLBACK.displayName,
);

const description = computed(
  () =>
    detailsQuery.data.value?.summary?.description ??
    DATASET_VISUALIZATION_DAG_FALLBACK.description,
);

const tags = computed(() => {
  const fromApi = detailsQuery.data.value?.summary?.tags;
  if (Array.isArray(fromApi) && fromApi.length) return fromApi;
  return DATASET_VISUALIZATION_DAG_FALLBACK.tags;
});

const href = computed(() => datasetVisualizationDagPath(props.dagId, props.runId));
</script>

<template>
  <div class="s3-viz-dag-link" :class="{ 's3-viz-dag-link--compact': compact }">
    <p v-if="!compact" class="s3-viz-dag-link-label">Airflow pipeline</p>
    <RouterLink :to="href" class="s3-viz-dag-link-anchor">
      {{ displayName }}
      <span class="s3-viz-dag-link-anchor-hint">Open in DAGs →</span>
    </RouterLink>
    <p v-if="!compact && description" class="s3-viz-dag-link-desc">
      {{ description }}
    </p>
    <p class="s3-viz-dag-link-meta">
      <code>{{ dagId }}</code>
      <span v-if="tags.length" class="s3-viz-dag-link-tags">
        <span v-for="tag in tags" :key="tag" class="s3-viz-dag-link-tag">{{ tag }}</span>
      </span>
    </p>
    <p v-if="runId" class="s3-viz-dag-link-run">
      Run: <code>{{ runId }}</code>
    </p>
  </div>
</template>
