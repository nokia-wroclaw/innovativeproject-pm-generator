<template>
  <div class="space-y-6">
    <section class="rounded-xl border border-border-default bg-surface p-5 shadow-sm">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 class="text-lg font-semibold text-fg">Synthetic data generation</h2>
          <p class="mt-1 text-sm text-fg-muted">
            Select a trained model and provide a prompt to generate synthetic data.
            Run status remains available when you navigate to other sections.
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          @click="isRegisterModalOpen = true"
        >
          Register S3 Model
        </Button>
      </div>
    </section>

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

    <RegisterS3ModelModal
      v-if="isRegisterModalOpen"
      :show="isRegisterModalOpen"
      @close="isRegisterModalOpen = false"
    />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

import { Button } from '@/components/ui';
import GenerateProcessModal from '../components/GenerateProcessModal.vue';
import ModelingProcessCard from '../components/ModelingProcessCard.vue';
import RegisterS3ModelModal from '../components/RegisterS3ModelModal.vue';
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
  dagId: 'moj_pierwszy_dag',
  description: 'Generates synthetic data from a trained model and prompt.',
  emptyText: 'Generated event log will appear after a successful run.',
};

const isModalOpen = ref(false);
const isRegisterModalOpen = ref(false);

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
