import { onBeforeUnmount, ref, unref, watch, isRef } from 'vue';

import { ApiError } from '../../../services/api';
import {
  fetchDatasetVisualizationStatus,
  requestDatasetVisualization,
} from '../../../services/s3';
import { DATASET_VISUALIZATION_DAG_ID } from '../visualizationDag.js';

const TERMINAL_STATUSES = new Set([
  'success',
  'failed',
  'unsupported_schema',
  'unavailable',
]);
const POLL_INTERVAL_MS = 3_000;
const MAX_NOT_FOUND_POLLS = 40;
const MAX_PIPELINE_POLLS = 120;

export function useDatasetVisualizationPoll(datasetIdRef, { enabled = true } = {}) {
  const status = ref(null);
  const isPolling = ref(false);
  const pollError = ref('');
  const pollTimedOut = ref(false);
  const pipelineTimedOut = ref(false);
  const isRetrying = ref(false);
  const retryError = ref('');

  let timerId = null;
  let notFoundPolls = 0;
  let pipelinePolls = 0;

  function clearTimer() {
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
  }

  function resetPollCounters() {
    notFoundPolls = 0;
    pipelinePolls = 0;
    pollTimedOut.value = false;
    pipelineTimedOut.value = false;
  }

  function shouldContinuePolling(nextStatus) {
    if (TERMINAL_STATUSES.has(nextStatus)) {
      return false;
    }

    if (nextStatus === 'not_found') {
      notFoundPolls += 1;
      if (notFoundPolls >= MAX_NOT_FOUND_POLLS) {
        pollTimedOut.value = true;
        return false;
      }
      return true;
    }

    notFoundPolls = 0;

    if (nextStatus === 'queued' || nextStatus === 'running') {
      pipelinePolls += 1;
      if (pipelinePolls >= MAX_PIPELINE_POLLS) {
        pipelineTimedOut.value = true;
        return false;
      }
      return true;
    }

    pipelinePolls = 0;
    return true;
  }

  async function pollOnce() {
    const datasetId = Number(unref(datasetIdRef));
    if (!Number.isFinite(datasetId) || datasetId <= 0) {
      return;
    }

    try {
      const data = await fetchDatasetVisualizationStatus(datasetId);
      status.value = data;
      pollError.value = '';

      if (!shouldContinuePolling(data.status)) {
        isPolling.value = false;
        clearTimer();
      }
    } catch (error) {
      pollError.value = error.message || 'Failed to fetch visualization status';
      isPolling.value = false;
      clearTimer();
    }
  }

  function startPolling() {
    clearTimer();
    resetPollCounters();
    isPolling.value = true;
    pollOnce();
    timerId = window.setInterval(pollOnce, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    clearTimer();
    isPolling.value = false;
  }

  async function retryVisualization() {
    const datasetId = Number(unref(datasetIdRef));
    if (!Number.isFinite(datasetId) || datasetId <= 0) {
      return;
    }

    isRetrying.value = true;
    retryError.value = '';

    try {
      await requestDatasetVisualization(datasetId);
      status.value = null;
      startPolling();
    } catch (error) {
      if (
        error instanceof ApiError &&
        error.details?.status === 'unsupported_schema'
      ) {
        status.value = {
          dataset_id: datasetId,
          dag_id: DATASET_VISUALIZATION_DAG_ID,
          status: 'unsupported_schema',
          message: error.details.message,
          summary: error.details,
        };
        isPolling.value = false;
        clearTimer();
        return;
      }
      retryError.value =
        error.message || 'Failed to start visualization pipeline';
    } finally {
      isRetrying.value = false;
    }
  }

  watch(
    [datasetIdRef, () => (isRef(enabled) ? enabled.value : enabled)],
    ([id, isEnabled]) => {
      stopPolling();
      status.value = null;
      pollError.value = '';
      retryError.value = '';
      resetPollCounters();
      if (
        isEnabled &&
        Number.isFinite(Number(id)) &&
        Number(id) > 0
      ) {
        startPolling();
      }
    },
    { immediate: true },
  );

  onBeforeUnmount(stopPolling);

  return {
    status,
    isPolling,
    pollError,
    pollTimedOut,
    pipelineTimedOut,
    isRetrying,
    retryError,
    refresh: pollOnce,
    retryVisualization,
  };
}
