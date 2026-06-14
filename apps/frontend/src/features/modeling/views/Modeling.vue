<template>
  <div class="space-y-6">
    <section class="rounded-xl border border-border-default bg-surface p-5 shadow-sm">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 class="text-lg font-semibold text-fg">Modeling processes</h2>
          <p class="mt-1 text-sm text-fg-muted">
            Each process opens its own dedicated DAG configuration form.
            Statuses are tracked globally and remain available after switching sections.
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          :disabled="isDatasetsFetching"
          @click="datasetsQuery.refetch()"
        >
          <RefreshCw :size="14" :class="isDatasetsFetching && 'animate-spin'" />
          Refresh datasets
        </Button>
      </div>

      <div
        v-if="datasetsError"
        class="mt-4 flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
      >
        <AlertCircle :size="16" class="mt-0.5 shrink-0" />
        <div>
          <p class="font-semibold">Failed to fetch datasets.</p>
          <p class="text-xs">{{ datasetsError.message }}</p>
        </div>
      </div>
    </section>

    <section class="grid gap-6 xl:grid-cols-2">
      <ModelingProcessCard
        v-for="process in processCards"
        :key="process.processType"
        :process="process"
        :artifact-labels="artifactLabels"
        @configure="openProcessModal"
        @refresh="refreshProcess"
      />
    </section>

    <PreprocessingProcessModal
      v-if="activeProcess && activeProcessType === PREPROCESSING_PROCESS"
      :show="Boolean(activeProcess)"
      :process="activeProcess"
      :datasets="datasets"
      @close="activeProcessType = null"
      @started="onProcessStarted"
    />

    <TrainingDatasetProcessModal
      v-if="activeProcess && activeProcessType === TRAINING_DATASET_PROCESS"
      :show="Boolean(activeProcess)"
      :process="activeProcess"
      :datasets="datasets"
      @close="activeProcessType = null"
      @started="onProcessStarted"
    />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import { AlertCircle, RefreshCw } from 'lucide-vue-next';

import { Button } from '@/components/ui';
import { useModelingDatasets } from '../composables/queries.js';
import ModelingProcessCard from '../components/ModelingProcessCard.vue';
import PreprocessingProcessModal from '../components/PreprocessingProcessModal.vue';
import TrainingDatasetProcessModal from '../components/TrainingDatasetProcessModal.vue';
import {
  getLatestRunForProcess,
  refreshTrackedModelingRun,
  trackModelingRun,
} from '../services/modelingRunMonitor.js';

const PREPROCESSING_PROCESS = 'preprocessing_feature_engineering';
const TRAINING_DATASET_PROCESS = 'training_dataset';
const TERMINAL_STATUSES = new Set(['success', 'failed']);

const artifactLabels = {
  preprocessed_dataset: 'Preprocessed dataset',
  featured_dataset: 'Featured dataset',
  training_dataset: 'Training dataset',
  model_pickle: 'Model',
};

const processDefinitions = {
  [PREPROCESSING_PROCESS]: {
    processType: PREPROCESSING_PROCESS,
    title: 'Preprocessing + Feature Engineering',
    dagId: 'preprocessing_pipeline',
    description: 'Runs Spark preprocessing on RAW PM data and writes parquet artifacts to S3.',
    emptyText: 'Outputs include wide/long indexed windows and scaling params under the dataset prefix.',
  },
  [TRAINING_DATASET_PROCESS]: {
    processType: TRAINING_DATASET_PROCESS,
    title: 'Training dataset creation',
    dagId: 'moj_pierwszy_dag',
    description: 'Builds the final training dataset from the DAG form inputs.',
    emptyText: 'This process saves training_dataset.parquet.',
  },
};

const activeProcessType = ref(null);
const datasetsQuery = useModelingDatasets();

const datasets = computed(() => datasetsQuery.data.value ?? []);
const datasetsError = computed(() => datasetsQuery.error.value);
const isDatasetsFetching = computed(() => datasetsQuery.isFetching.value);
const activeProcess = computed(() =>
  activeProcessType.value ? processDefinitions[activeProcessType.value] : null,
);

const processCards = computed(() =>
  [PREPROCESSING_PROCESS, TRAINING_DATASET_PROCESS].map((processType) => {
    const run = getLatestRunForProcess(processType);
    const status = run?.statusData?.status ?? run?.status ?? null;
    return {
      ...processDefinitions[processType],
      runId: run?.runId ?? null,
      statusData: run?.statusData ?? null,
      error: run?.error ?? null,
      isPolling: Boolean(run && !TERMINAL_STATUSES.has(status)),
    };
  }),
);

function openProcessModal(processType) {
  activeProcessType.value = processType;
}

function onProcessStarted(response) {
  trackModelingRun({
    processType: response.process_type,
    runId: response.run_id,
    title: processDefinitions[response.process_type]?.title ?? response.process_type,
  });
}

function refreshProcess(process) {
  refreshTrackedModelingRun(process.processType, process.runId);
}
</script>
