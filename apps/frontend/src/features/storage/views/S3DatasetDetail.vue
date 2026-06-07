<script setup>
import { computed, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { ArrowLeft } from 'lucide-vue-next';
import '../../../assets/S3.css';
import '../assets/S3DatasetDetail.css';
import DatasetPreviewContent from '../components/DatasetPreviewContent.vue';
import DatasetTypeBadge from '../components/DatasetTypeBadge.vue';
import DatasetVisualizationWidget from '../components/DatasetVisualizationWidget.vue';
import { fetchDatasetPreview } from '../../../services/s3';

const route = useRoute();
const router = useRouter();

const datasetId = computed(() => Number(route.params.datasetId));
const storageTab = computed(() =>
  typeof route.query.tab === 'string' ? route.query.tab : 'generated',
);

const isLoading = ref(true);
const error = ref('');
const previewData = ref(null);

const backHref = computed(() => ({
  path: '/s3',
  query: { tab: storageTab.value },
}));

const pageTitle = computed(
  () => previewData.value?.file_name || `Dataset #${datasetId.value}`,
);

const previewColumns = computed(
  () => previewData.value?.tables?.[0]?.columns ?? [],
);

async function loadPreview() {
  if (!Number.isFinite(datasetId.value) || datasetId.value <= 0) {
    error.value = 'Invalid dataset id';
    previewData.value = null;
    isLoading.value = false;
    return;
  }

  isLoading.value = true;
  error.value = '';
  previewData.value = null;

  try {
    previewData.value = await fetchDatasetPreview(datasetId.value);
  } catch (err) {
    error.value = err.message || 'Failed to load dataset preview';
  } finally {
    isLoading.value = false;
  }
}

watch(datasetId, loadPreview, { immediate: true });
</script>

<template>
  <div class="s3-dataset-detail">
    <header class="s3-dataset-detail-header">
      <button
        type="button"
        class="s3-dataset-detail-back"
        @click="router.push(backHref)"
      >
        <ArrowLeft :size="16" />
        Back to Storage
      </button>

      <div class="s3-dataset-detail-heading">
        <div class="s3-dataset-detail-title-row">
          <h2 class="s3-dataset-detail-title">{{ pageTitle }}</h2>
          <DatasetTypeBadge v-if="previewData?.type" :type="previewData.type" />
        </div>
        <p v-if="previewData?.s3_key" class="s3-dataset-detail-meta">
          <span>ID {{ previewData.dataset_id ?? datasetId }}</span>
          <span class="s3-dataset-detail-sep">·</span>
          <code>{{ previewData.s3_key }}</code>
        </p>
      </div>
    </header>

    <section class="s3-dataset-detail-panel">
      <h3 class="s3-dataset-detail-section-title">Data preview</h3>
      <DatasetPreviewContent
        :loading="isLoading"
        :error="error"
        :preview-data="previewData"
        :scrollable="false"
      />
    </section>

    <section v-if="!isLoading && !error && previewData" class="s3-dataset-detail-panel">
      <DatasetVisualizationWidget
        :dataset-id="datasetId"
        :preview-columns="previewColumns"
      />
    </section>
  </div>
</template>
