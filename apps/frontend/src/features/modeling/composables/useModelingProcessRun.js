import { computed, ref } from 'vue';

import { trackModelingRun } from '../services/modelingRunMonitor.js';
import { useModelingRunStatus, useTriggerModelingRun } from './queries.js';

const TERMINAL_STATUSES = new Set(['success', 'failed']);

/**
 * Submit + live status polling for modeling process modals.
 * @param {string} processType
 * @param {string} title
 */
export function useModelingProcessRun(processType, title) {
  const phase = ref('form');
  const formError = ref('');
  const startedRun = ref(null);
  const triggerMutation = useTriggerModelingRun();

  const runId = computed(() => startedRun.value?.run_id ?? null);
  const processTypeRef = computed(() => processType);
  const pollRunId = computed(() => (phase.value === 'running' ? runId.value : null));

  const statusQuery = useModelingRunStatus(processTypeRef, pollRunId);

  const statusData = computed(() => statusQuery.data.value ?? null);
  const isSubmitting = computed(
    () => phase.value === 'submitting' || triggerMutation.isPending.value,
  );
  const isPolling = computed(
    () =>
      phase.value === 'running' &&
      Boolean(runId.value) &&
      (!statusData.value || !TERMINAL_STATUSES.has(statusData.value.status)),
  );

  async function triggerRun(body, emit) {
    if (isSubmitting.value) return;
    phase.value = 'submitting';
    formError.value = '';
    try {
      const response = await triggerMutation.mutateAsync({ processType, body });
      startedRun.value = response;
      phase.value = 'running';
      trackModelingRun({
        processType: response.process_type,
        runId: response.run_id,
        title,
      });
      emit?.('started', response);
    } catch (error) {
      phase.value = 'error';
      formError.value = error?.message ?? 'Failed to trigger DAG.';
    }
  }

  function reset() {
    phase.value = 'form';
    formError.value = '';
    startedRun.value = null;
  }

  return {
    phase,
    formError,
    startedRun,
    statusData,
    statusQuery,
    isSubmitting,
    isPolling,
    triggerRun,
    reset,
  };
}
