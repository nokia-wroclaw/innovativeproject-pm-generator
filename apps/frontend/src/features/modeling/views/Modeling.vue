<template>
  <div class="space-y-6">
    <section class="rounded-xl border border-border-default bg-surface p-5 shadow-sm">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 class="text-lg font-semibold text-fg">Modeling processes</h2>
          <p class="mt-1 text-sm text-fg-muted">
            Each process opens its own dedicated DAG configuration form.
            Statuses are tracked globally and remain available after switching tabs.
          </p>
        </div>
        <Button variant="secondary" size="sm" :disabled="isDatasetsFetching" @click="datasetsQuery.refetch()">
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
      <article
        v-for="process in processCards"
        :key="process.processType"
        class="rounded-xl border border-border-default bg-surface p-5 shadow-sm"
      >
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 class="text-lg font-semibold text-fg">{{ process.title }}</h2>
            <p class="mt-1 text-sm text-fg-muted">{{ process.description }}</p>
            <p class="mt-2 text-xs text-fg-subtle">
              DAG: <span class="font-mono text-fg">{{ process.dagId }}</span>
            </p>
          </div>
          <div class="flex flex-wrap items-center justify-end gap-2">
            <RouterLink :to="`/dags/${process.dagId}`" class="inline-flex">
              <Button variant="secondary">
                DAG details
              </Button>
            </RouterLink>
            <Button @click="openProcessModal(process.processType)">
              <Play :size="14" />
              Configure and run
            </Button>
          </div>
        </div>

        <div class="mt-5 rounded-lg border border-border-default bg-surface-muted p-4">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p class="text-xs uppercase tracking-wide text-fg-subtle">Run status</p>
              <p v-if="process.runId" class="mt-2 break-all font-mono text-xs text-fg">
                {{ process.runId }}
              </p>
              <p v-else class="mt-2 text-sm text-fg-muted">{{ process.emptyText }}</p>
            </div>
            <div class="flex items-center gap-2">
              <DagStatusBadge
                v-if="process.statusData"
                :status="process.statusData.status"
                density="compact"
              />
              <span v-else class="text-xs text-fg-muted">
                {{ process.runId ? 'Queued' : 'Not started' }}
              </span>
              <Loader2
                v-if="process.isPolling"
                :size="14"
                class="animate-spin text-fg-subtle"
              />
            </div>
          </div>
          <Button
            variant="secondary"
            size="sm"
            class="mt-4"
            :disabled="!process.runId"
            @click="refreshProcess(process)"
          >
            <RefreshCw :size="14" />
            Refresh
          </Button>
        </div>

        <div
          v-if="process.error"
          class="mt-4 flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
        >
          <AlertCircle :size="16" class="mt-0.5 shrink-0" />
          <div>
            <p class="font-semibold">Failed to read run status.</p>
            <p class="text-xs">{{ process.error }}</p>
          </div>
        </div>

        <div class="mt-5 space-y-4">
          <div class="rounded-lg border border-border-default">
            <div class="border-b border-border-default px-4 py-3">
              <h3 class="text-sm font-semibold text-fg">Artifacts</h3>
            </div>
            <div v-if="artifactsFor(process).length" class="divide-y divide-border-default">
              <div
                v-for="artifact in artifactsFor(process)"
                :key="artifact.kind"
                class="flex items-start gap-3 px-4 py-3"
              >
                <FileCheck2
                  :size="16"
                  :class="artifact.status === 'saved' ? 'text-emerald-500' : 'text-fg-subtle'"
                  class="mt-0.5 shrink-0"
                />
                <div class="min-w-0">
                  <p class="text-sm font-medium text-fg">{{ formatArtifactKind(artifact.kind) }}</p>
                  <p class="break-all font-mono text-[11px] text-fg-muted">{{ artifact.path }}</p>
                </div>
              </div>
            </div>
            <div v-else class="flex items-center gap-2 p-4 text-sm text-fg-muted">
              <XCircle :size="16" class="text-fg-subtle" />
              Artifacts will appear after the run starts.
            </div>
          </div>

          <div class="rounded-lg border border-border-default">
            <div class="border-b border-border-default px-4 py-3">
              <h3 class="text-sm font-semibold text-fg">Summary</h3>
            </div>
            <div v-if="process.statusData?.metrics" class="grid grid-cols-2 gap-3 p-4">
              <div
                v-for="(value, key) in process.statusData.metrics"
                :key="key"
                class="rounded-md bg-surface-muted p-3"
              >
                <p class="text-xs uppercase tracking-wide text-fg-subtle">{{ key }}</p>
                <p class="mt-1 text-lg font-semibold text-fg">{{ value }}</p>
              </div>
            </div>
            <div v-else class="flex items-center gap-2 p-4 text-sm text-fg-muted">
              <XCircle :size="16" class="text-fg-subtle" />
              Summary will appear after the status is Success.
            </div>
          </div>
        </div>
      </article>
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
import {
  AlertCircle,
  FileCheck2,
  Loader2,
  Play,
  RefreshCw,
  XCircle,
} from 'lucide-vue-next';

import { Button } from '@/components/ui';
import DagStatusBadge from '@/features/dags/components/DagStatusBadge.vue';
import { useModelingDatasets } from '../composables/queries.js';
import PreprocessingProcessModal from '../components/PreprocessingProcessModal.vue';
import TrainingDatasetProcessModal from '../components/TrainingDatasetProcessModal.vue';
import {
  modelingRunMonitorState,
  refreshTrackedModelingRun,
  trackModelingRun,
} from '../services/modelingRunMonitor.js';

const PREPROCESSING_PROCESS = 'preprocessing_feature_engineering';
const TRAINING_DATASET_PROCESS = 'training_dataset';
const TERMINAL_STATUSES = new Set(['success', 'failed']);

const processDefinitions = {
  [PREPROCESSING_PROCESS]: {
    processType: PREPROCESSING_PROCESS,
    title: 'Preprocessing + Feature Engineering',
    dagId: 'moj_pierwszy_dag',
    description: 'Runs preprocessing and feature engineering in one asynchronous DAG.',
    emptyText: 'This process saves a preprocessed dataset and a featured dataset.',
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
    const run = latestRunFor(processType);
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

function latestRunFor(processType) {
  return [...modelingRunMonitorState.runs]
    .filter((run) => run.processType === processType)
    .sort((a, b) => String(b.key).localeCompare(String(a.key)))[0] ?? null;
}

function artifactsFor(process) {
  return process.statusData?.artifacts ?? [];
}

function formatArtifactKind(kind) {
  const labels = {
    preprocessed_dataset: 'Preprocessed dataset',
    featured_dataset: 'Featured dataset',
    training_dataset: 'Training dataset',
    model_pickle: 'Model pickle',
  };
  return labels[kind] ?? kind;
}
</script>
