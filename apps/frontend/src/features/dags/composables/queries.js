/**
 * @file Vue Query composables for the DAG management feature.
 *
 * All server state lives here — components stay thin. Refetch intervals
 * are aligned with the polling contract in
 * `docs/architecture/dag-management.md` §8.
 */

import { computed, unref } from 'vue';
import {
  useQuery,
  useMutation,
  useQueryClient,
} from '@tanstack/vue-query';

import * as Api from '../services/dagsApi.js';

const FIVE_SECONDS = 5_000;
const TWO_SECONDS = 2_000;
const THREE_SECONDS = 3_000;
const FIFTEEN_SECONDS = 15_000;

// ─── Query keys ─────────────────────────────────────────────────────────────
export const queryKeys = {
  list: () => ['dags'],
  detail: (dagId) => ['dags', unref(dagId)],
  runs: (dagId) => ['dags', unref(dagId), 'runs'],
  taskInstances: (dagId, runId) => [
    'dags', unref(dagId), 'runs', unref(runId), 'tasks',
  ],
  taskInstance: (dagId, runId, taskId) => [
    'dags', unref(dagId), 'runs', unref(runId), 'tasks', unref(taskId),
  ],
  taskTries: (dagId, runId, taskId) => [
    'dags', unref(dagId), 'runs', unref(runId), 'tasks', unref(taskId), 'tries',
  ],
};

// ─── Reads ──────────────────────────────────────────────────────────────────

export function useDagList() {
  return useQuery({
    queryKey: queryKeys.list(),
    queryFn: Api.listDags,
    refetchInterval: FIVE_SECONDS,
    refetchIntervalInBackground: false,
  });
}

/**
 * @param {import('vue').Ref<string | null>} dagIdRef
 */
export function useDagDetails(dagIdRef) {
  return useQuery({
    queryKey: computed(() => queryKeys.detail(dagIdRef)),
    queryFn: () => Api.getDagDetails(unref(dagIdRef)),
    enabled: computed(() => Boolean(unref(dagIdRef))),
    refetchInterval: FIFTEEN_SECONDS,
  });
}

/**
 * Task instances overlayed on the graph. Polls fast when there's an active
 * run, slow otherwise. Pass an `isRunning` ref so the parent can drive this.
 *
 * @param {import('vue').Ref<string | null>} dagIdRef
 * @param {import('vue').Ref<string | null>} runIdRef
 * @param {import('vue').Ref<boolean>} isRunningRef
 */
export function useTaskInstances(dagIdRef, runIdRef, isRunningRef) {
  return useQuery({
    queryKey: computed(() => queryKeys.taskInstances(dagIdRef, runIdRef)),
    queryFn: () => Api.listTaskInstances(unref(dagIdRef), unref(runIdRef)),
    enabled: computed(() => Boolean(unref(dagIdRef) && unref(runIdRef))),
    refetchInterval: computed(() =>
      unref(isRunningRef) ? THREE_SECONDS : FIFTEEN_SECONDS,
    ),
  });
}

/**
 * @param {import('vue').Ref<string | null>} dagIdRef
 * @param {import('vue').Ref<string | null>} runIdRef
 * @param {import('vue').Ref<string | null>} taskIdRef
 */
export function useTaskInstance(dagIdRef, runIdRef, taskIdRef) {
  return useQuery({
    queryKey: computed(() => queryKeys.taskInstance(dagIdRef, runIdRef, taskIdRef)),
    queryFn: () =>
      Api.getTaskInstance(unref(dagIdRef), unref(runIdRef), unref(taskIdRef)),
    enabled: computed(() =>
      Boolean(unref(dagIdRef) && unref(runIdRef) && unref(taskIdRef)),
    ),
    refetchInterval: TWO_SECONDS,
  });
}

/**
 * @param {import('vue').Ref<string | null>} dagIdRef
 * @param {import('vue').Ref<string | null>} runIdRef
 * @param {import('vue').Ref<string | null>} taskIdRef
 */
export function useTaskTries(dagIdRef, runIdRef, taskIdRef) {
  return useQuery({
    queryKey: computed(() => queryKeys.taskTries(dagIdRef, runIdRef, taskIdRef)),
    queryFn: () =>
      Api.listTaskTries(unref(dagIdRef), unref(runIdRef), unref(taskIdRef)),
    enabled: computed(() =>
      Boolean(unref(dagIdRef) && unref(runIdRef) && unref(taskIdRef)),
    ),
    refetchInterval: FIFTEEN_SECONDS,
    // Airflow's /tries is a *history* endpoint; right after a clear/retry it
    // may return two rows with the same try_number — one for the finished
    // attempt (state=success/failed) and one freshly created for the new
    // attempt (state=null). We pick the newest row (= the one without an
    // end_date, if any) so the dropdown shows a single, accurate entry.
    select: (data) => _dedupeTriesByNumber(data ?? []),
  });
}

function _dedupeTriesByNumber(tries) {
  const byNumber = new Map();
  for (const t of tries) {
    const existing = byNumber.get(t.try_number);
    if (!existing) {
      byNumber.set(t.try_number, t);
      continue;
    }
    // Prefer the row that's still open (no end_date) — that's the live one.
    // If both are closed, prefer the one that ended later.
    const existingClosed = Boolean(existing.end_date);
    const incomingClosed = Boolean(t.end_date);
    if (existingClosed && !incomingClosed) {
      byNumber.set(t.try_number, t);
    } else if (existingClosed && incomingClosed) {
      if (new Date(t.end_date) > new Date(existing.end_date)) {
        byNumber.set(t.try_number, t);
      }
    }
  }
  return [...byNumber.values()].sort((a, b) => a.try_number - b.try_number);
}

// ─── Mutations ──────────────────────────────────────────────────────────────

function _invalidate(queryClient, { dagId, runId }) {
  queryClient.invalidateQueries({ queryKey: queryKeys.list() });
  if (dagId) {
    queryClient.invalidateQueries({ queryKey: queryKeys.detail(dagId) });
    queryClient.invalidateQueries({ queryKey: queryKeys.runs(dagId) });
    if (runId) {
      queryClient.invalidateQueries({
        queryKey: queryKeys.taskInstances(dagId, runId),
      });
      // Nested keys: any task-level query under this run (single instance,
      // tries history, etc.) — these must be invalidated too, otherwise the
      // dropdown shows stale rows ("Try N · success") alongside the freshly
      // created in-progress try ("Try N · null") after a retry/clear.
      queryClient.invalidateQueries({
        queryKey: ['dags', dagId, 'runs', runId],
      });
    }
  }
}

export function useTriggerDag() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ dagId, body }) => Api.triggerDag(dagId, body),
    onSuccess: (_data, { dagId }) => _invalidate(queryClient, { dagId }),
  });
}

export function useStopDagRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ dagId, runId }) => Api.stopDagRun(dagId, runId),
    onSuccess: (_data, { dagId, runId }) =>
      _invalidate(queryClient, { dagId, runId }),
  });
}

export function useClearDagRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ dagId, runId }) => Api.clearDagRun(dagId, runId),
    onSuccess: (_data, { dagId, runId }) =>
      _invalidate(queryClient, { dagId, runId }),
  });
}

export function useClearTaskInstance() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ dagId, runId, taskId, downstream }) =>
      Api.clearTaskInstance(dagId, runId, taskId, { downstream }),
    onSuccess: (_data, { dagId, runId }) =>
      _invalidate(queryClient, { dagId, runId }),
  });
}
