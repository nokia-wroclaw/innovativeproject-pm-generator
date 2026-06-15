<template>
  <div class="space-y-6">
    <section class="grid gap-6">
      <ModelingProcessCard
        :process="generateProcessCard"
        :artifact-labels="artifactLabels"
        @configure="openProcessModal"
        @refresh="refreshProcess"
      />
    </section>

    <GenerateProcessModal
      v-if="activeProcess"
      :show="isModalOpen"
      :process="activeProcess"
      @close="isModalOpen = false"
      @started="onProcessStarted"
    />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

import GenerateProcessModal from '../components/GenerateProcessModal.vue';
import ModelingProcessCard from '../components/ModelingProcessCard.vue';
import {
  getLatestRunForProcess,
  refreshTrackedModelingRun,
  trackModelingRun,
} from '../services/modelingRunMonitor.js';

const GENERATE_PROCESS = 'generate';
const TERMINAL_STATUSES = new Set(['success', 'failed']);

const artifactLabels = {
  generated_event_log: 'Generated event log',
  generation_report: 'Generation report',
};

const processDefinition = {
  processType: GENERATE_PROCESS,
  title: 'Synthetic data generation',
  dagId: 'generate_pipeline',
  description: 'Generates synthetic data from a trained model and prompt.',
  emptyText: 'Generated event log will appear after a successful run.',
};

const isModalOpen = ref(false);

const generateProcessCard = computed(() => {
  const run = getLatestRunForProcess(GENERATE_PROCESS);
  const status = run?.statusData?.status ?? run?.status ?? null;
  return {
    ...processDefinition,
    runId: run?.runId ?? null,
    statusData: run?.statusData ?? null,
    error: run?.error ?? null,
    isPolling: Boolean(run && !TERMINAL_STATUSES.has(status)),
  };
});

const activeProcess = computed(() => processDefinition);

function openProcessModal() {
  isModalOpen.value = true;
}

function onProcessStarted(response) {
  trackModelingRun({
    processType: response.process_type,
    runId: response.run_id,
    title: processDefinition.title,
  });
}

function refreshProcess(process) {
  refreshTrackedModelingRun(process.processType, process.runId);
}
</script>
