import { reactive } from 'vue';

import { getModelingRunStatus } from './modelingApi.js';

const TERMINAL_STATUSES = new Set(['success', 'failed']);
const POLL_INTERVAL_MS = 3_000;
const TOAST_TTL_MS = 6_000;
const STORAGE_KEY = 'genpm_modeling_runs';

const timers = new Map();

export const modelingRunMonitorState = reactive({
  runs: loadStoredRuns(),
  toasts: [],
});

resumeStoredRuns();

export function trackModelingRun({ processType, runId, title }) {
  if (!processType || !runId) return;

  const key = runKey(processType, runId);
  const existing = modelingRunMonitorState.runs.find((run) => run.key === key);
  if (existing) {
    existing.title = title;
  } else {
    modelingRunMonitorState.runs.push({
      key,
      processType,
      runId,
      title,
      status: 'queued',
      notified: false,
    });
  }
  persistRuns();

  if (!timers.has(key)) {
    pollRun(key);
    timers.set(key, window.setInterval(() => pollRun(key), POLL_INTERVAL_MS));
  }
}

export function refreshTrackedModelingRun(processType, runId) {
  if (!processType || !runId) return;
  pollRun(runKey(processType, runId));
}

async function pollRun(key) {
  const run = modelingRunMonitorState.runs.find((item) => item.key === key);
  if (!run) {
    clearRunTimer(key);
    return;
  }

  try {
    const statusData = await getModelingRunStatus(run.processType, run.runId);
    run.status = statusData.status;
    run.statusData = statusData;
    run.error = null;
    persistRuns();

    if (TERMINAL_STATUSES.has(statusData.status)) {
      clearRunTimer(key);
      if (!run.notified) {
        run.notified = true;
        pushToast({
          variant: statusData.status === 'success' ? 'success' : 'failed',
          title: statusData.status === 'success'
            ? 'Proces zakończony'
            : 'Proces zakończony błędem',
          message: `${run.title}: Airflow zwrócił ${statusData.raw_state}.`,
        });
        persistRuns();
      }
    }
  } catch (error) {
    run.error = error?.message ?? 'Nie udało się odczytać statusu procesu.';
    persistRuns();
  }
}

function pushToast({ variant, title, message }) {
  const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  modelingRunMonitorState.toasts.push({ id, variant, title, message });
  window.setTimeout(() => {
    const index = modelingRunMonitorState.toasts.findIndex((toast) => toast.id === id);
    if (index !== -1) {
      modelingRunMonitorState.toasts.splice(index, 1);
    }
  }, TOAST_TTL_MS);
}

function clearRunTimer(key) {
  const timer = timers.get(key);
  if (timer) {
    window.clearInterval(timer);
    timers.delete(key);
  }
}

function runKey(processType, runId) {
  return `${processType}:${runId}`;
}

function persistRuns() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(modelingRunMonitorState.runs.map((run) => ({
      key: run.key,
      processType: run.processType,
      runId: run.runId,
      title: run.title,
      status: run.status,
      statusData: run.statusData ?? null,
      notified: run.notified,
      error: run.error ?? null,
    }))),
  );
}

function loadStoredRuns() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function resumeStoredRuns() {
  window.setTimeout(() => {
    for (const run of modelingRunMonitorState.runs) {
      if (!TERMINAL_STATUSES.has(run.status) && !timers.has(run.key)) {
        pollRun(run.key);
        timers.set(run.key, window.setInterval(() => pollRun(run.key), POLL_INTERVAL_MS));
      }
    }
  }, 0);
}
