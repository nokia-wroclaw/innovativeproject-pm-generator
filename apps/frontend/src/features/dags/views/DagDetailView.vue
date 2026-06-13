<!--
  DagDetailView — interactive graph view of a single DAG.

  Layout (top → bottom):
    1. Header bar       — display name, status of selected run, actions
    2. Run selector     — dropdown with last 10 runs, current run highlighted
    3. Graph canvas     — Vue Flow with auto-layout (dagre TB)
    4. TaskDetailsSheet — slide-in, controlled by the parent
-->
<template>
  <div class="flex h-[calc(100vh-180px)] flex-col">
    <!-- ── Header / actions ─────────────────────────────────────────── -->
    <header class="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div class="space-y-1">
        <div class="flex items-center gap-3">
          <button
            type="button"
            class="rounded-md p-1 text-fg-muted hover:bg-surface-muted hover:text-fg"
            @click="$router.push('/dags')"
            aria-label="Back to dashboard"
          >
            <ArrowLeft :size="16" />
          </button>
          <h2 class="text-lg font-semibold text-fg">
            {{ dagSummary?.display_name ?? dagId }}
          </h2>
          <DagStatusBadge
            v-if="selectedRun"
            :status="selectedRun.status"
            density="compact"
          />
          <span v-else class="text-xs text-fg-subtle">No runs yet</span>
        </div>
        <p class="ml-7 font-mono text-[11px] text-fg-subtle">{{ dagId }}</p>
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <select
          v-if="recentRuns.length"
          v-model="selectedRunId"
          class="rounded-md border border-border-default bg-surface px-3 py-1.5 text-xs text-fg
                 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
        >
          <option
            v-for="run in recentRuns"
            :key="run.run_id"
            :value="run.run_id"
          >
            {{ formatRunLabel(run) }}
          </option>
        </select>

        <Button
          variant="secondary"
          size="sm"
          :disabled="!selectedRun || isStopping || !canStopRun"
          @click="onStopRun"
        >
          <Square :size="14" />
          Stop
        </Button>
        <Button
          variant="secondary"
          size="sm"
          :disabled="!selectedRun || isClearingRun"
          @click="onClearRun"
        >
          <RotateCw :size="14" />
          Clear run
        </Button>
        <Button
          variant="primary"
          size="sm"
          @click="triggerOpen = true"
        >
          <Play :size="14" />
          Trigger DAG
        </Button>
      </div>
    </header>

    <!-- ── Error banner ─────────────────────────────────────────────── -->
    <div
      v-if="detailError"
      class="mb-4 flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
    >
      <AlertCircle :size="16" class="mt-0.5 shrink-0" />
      <div class="space-y-1">
        <p class="font-semibold">Failed to load DAG.</p>
        <p class="text-xs">{{ detailError.message }}</p>
      </div>
    </div>

    <!-- ── Graph canvas ─────────────────────────────────────────────── -->
    <div class="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-border-default bg-surface-muted">
      <div
        v-if="isLoadingDetails"
        class="absolute inset-0 z-10 flex items-center justify-center text-sm text-fg-muted"
      >
        <Loader2 :size="20" class="mr-2 animate-spin" />
        Loading DAG…
      </div>

      <div
        v-else-if="!layoutNodes.length && !detailError"
        class="flex h-full items-center justify-center p-6 text-center text-sm text-fg-muted"
      >
        No tasks in this DAG.
      </div>

      <VueFlow
        v-else
        :nodes="layoutNodes"
        :edges="layoutEdges"
        :node-types="{ task: TaskNode }"
        :nodes-draggable="false"
        :nodes-connectable="false"
        :elements-selectable="true"
        :default-edge-options="defaultEdgeOptions"
        :fit-view-on-init="true"
        :max-zoom="1.5"
        :min-zoom="0.25"
        @node-click="onNodeClick"
      >
        <Background :gap="20" :size="1" />
        <MiniMap
          pannable zoomable
          class="!bg-surface !border !border-border-default !rounded-lg"
        />
        <Controls
          position="bottom-right"
          class="!bg-surface !border !border-border-default !rounded-lg !shadow-sm"
        />
      </VueFlow>
    </div>

    <!-- ── Task details (slide-in) ──────────────────────────────────── -->
    <TaskDetailsSheet
      v-model:open="sheetOpen"
      :task-instance="selectedTaskInstance"
      :available-tries="tries"
      :dag-id="dagId"
      :run-id="selectedRunId"
      @retry-task="onRetryTask"
      @stop-run="onStopRun"
    />

    <!-- ── Trigger modal ────────────────────────────────────────────── -->
    <TriggerDagDialog
      v-model:open="triggerOpen"
      :dag-id="dagId"
      @triggered="onTriggered"
    />
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useRoute } from 'vue-router';
import { VueFlow, MarkerType } from '@vue-flow/core';
import { Background } from '@vue-flow/background';
import { Controls } from '@vue-flow/controls';
import { MiniMap } from '@vue-flow/minimap';
import {
  ArrowLeft, AlertCircle, Loader2, Play, RotateCw, Square,
} from 'lucide-vue-next';

import '@vue-flow/core/dist/style.css';
import '@vue-flow/core/dist/theme-default.css';
import '@vue-flow/controls/dist/style.css';
import '@vue-flow/minimap/dist/style.css';

import { Button } from '@/components/ui';
import DagStatusBadge from '../components/DagStatusBadge.vue';
import TaskNode from '../components/TaskNode.vue';
import TaskDetailsSheet from '../components/TaskDetailsSheet.vue';
import TriggerDagDialog from '../components/TriggerDagDialog.vue';
import { useDagLayout } from '../composables/useDagLayout.js';
import {
  useDagDetails,
  useTaskInstances,
  useTaskInstance,
  useTaskTries,
  useClearTaskInstance,
  useClearDagRun,
  useStopDagRun,
} from '../composables/queries.js';

const route = useRoute();
const dagId = computed(() => {
  const raw = route.params.dagId;
  const id = Array.isArray(raw) ? raw[0] : raw;
  try {
    return decodeURIComponent(String(id ?? ''));
  } catch {
    return String(id ?? '');
  }
});

const defaultEdgeOptions = {
  type: 'smoothstep',
  style: { stroke: '#94a3b8', strokeWidth: 1.5 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8', width: 16, height: 16 },
};

// ─── Details (graph + recent runs) ──────────────────────────────────────────
const dagIdRef = computed(() => dagId.value);
const detailsQuery = useDagDetails(dagIdRef);

const isLoadingDetails = computed(
  () => detailsQuery.isLoading.value && !detailsQuery.data.value,
);
const detailError = computed(() => detailsQuery.error.value);

const dagSummary = computed(() => detailsQuery.data.value?.summary ?? null);
const graph = computed(() => detailsQuery.data.value?.graph ?? null);
const recentRuns = computed(() => detailsQuery.data.value?.recent_runs ?? []);

// ─── Run selection ──────────────────────────────────────────────────────────
const selectedRunId = ref(null);

function runIdFromRouteQuery() {
  const raw = route.query.run;
  const id = Array.isArray(raw) ? raw[0] : raw;
  return id ? String(id) : null;
}

watch(
  [recentRuns, () => route.query.run],
  ([runs]) => {
    const fromQuery = runIdFromRouteQuery();

    if (fromQuery && runs.find((r) => r.run_id === fromQuery)) {
      selectedRunId.value = fromQuery;
      return;
    }

    if (!selectedRunId.value && runs.length) {
      selectedRunId.value = runs[0].run_id;
      return;
    }

    if (selectedRunId.value && !runs.find((r) => r.run_id === selectedRunId.value)) {
      selectedRunId.value = runs[0]?.run_id ?? null;
    }
  },
  { immediate: true },
);

const selectedRun = computed(
  () => recentRuns.value.find((r) => r.run_id === selectedRunId.value) ?? null,
);

const isRunning = computed(() => selectedRun.value?.status === 'running');
const canStopRun = computed(
  () => selectedRun.value && ['running', 'queued'].includes(selectedRun.value.status),
);

// ─── Task instances (overlay) ───────────────────────────────────────────────
const taskInstancesQuery = useTaskInstances(dagIdRef, selectedRunId, isRunning);
const taskInstances = computed(() => taskInstancesQuery.data.value ?? []);

// ─── Sheet / selection ──────────────────────────────────────────────────────
// Declared early so `statusByTask` below can overlay the singular query.
const sheetOpen = ref(false);
const selectedTaskId = ref(null);

// Fresh task instance for the currently selected task (polls every 2s).
const selectedTaskQuery = useTaskInstance(dagIdRef, selectedRunId, selectedTaskId);

const statusByTask = computed(() => {
  const map = {};
  for (const ti of taskInstances.value) {
    map[ti.task_id] = ti;
  }
  // Overlay the single-task query so the graph re-renders the cell for the
  // currently-selected node even when the bulk poll is between refreshes.
  const single = selectedTaskQuery.data.value;
  if (single?.task_id) map[single.task_id] = single;
  return map;
});

// ─── Layout (dagre) ─────────────────────────────────────────────────────────
const { nodes: layoutNodes, edges: layoutEdges } = useDagLayout(
  () => graph.value,
  () => statusByTask.value,
  { direction: 'TB' },
);

/**
 * Build a *synthetic* placeholder TaskInstance from the graph node, so that
 * users still see something the moment they click a node — even before
 * Airflow has materialised a task_instance row (e.g. just after triggering).
 */
function _placeholderTI(taskId) {
  const node = graph.value?.nodes?.find((n) => n.task_id === taskId);
  if (!node) return null;
  return {
    task_id: taskId,
    run_id: selectedRunId.value ?? '',
    status: 'none',
    raw_state: 'none',
    try_number: 0,
    max_tries: node.retries_max ?? 0,
    start_date: null,
    end_date: null,
    duration_ms: null,
    operator: node.operator ?? 'Operator',
    pool: 'default_pool',
    queue: 'default',
    executor_config: {},
    note: null,
  };
}

const selectedTaskInstance = computed(() => {
  if (!selectedTaskId.value) return null;
  // 1. Freshest: dedicated /tasks/{task_id} query
  if (selectedTaskQuery.data.value) return selectedTaskQuery.data.value;
  // 2. From the bulk list polled for the whole run
  const fromList = statusByTask.value[selectedTaskId.value];
  if (fromList) return fromList;
  // 3. Synthetic from the graph, so we always have *something* to render
  return _placeholderTI(selectedTaskId.value);
});

const triesQuery = useTaskTries(dagIdRef, selectedRunId, selectedTaskId);
const tries = computed(() => triesQuery.data.value ?? []);

function onNodeClick({ node }) {
  selectedTaskId.value = node.id;
  sheetOpen.value = true;
}

// ─── Mutations ──────────────────────────────────────────────────────────────
const clearTaskMutation = useClearTaskInstance();
const clearRunMutation = useClearDagRun();
const stopRunMutation = useStopDagRun();

const isClearingRun = computed(() => clearRunMutation.isPending.value);
const isStopping = computed(() => stopRunMutation.isPending.value);

async function onRetryTask() {
  if (!selectedTaskId.value || !selectedRunId.value) return;
  await clearTaskMutation.mutateAsync({
    dagId: dagId.value,
    runId: selectedRunId.value,
    taskId: selectedTaskId.value,
    downstream: false,
  });
}

async function onStopRun() {
  if (!selectedRunId.value) return;
  await stopRunMutation.mutateAsync({
    dagId: dagId.value,
    runId: selectedRunId.value,
  });
}

async function onClearRun() {
  if (!selectedRunId.value) return;
  if (
    !window.confirm(
      `Clear DAG run ${selectedRunId.value}? All task instances will be re-run.`,
    )
  ) return;
  await clearRunMutation.mutateAsync({
    dagId: dagId.value,
    runId: selectedRunId.value,
  });
}

const triggerOpen = ref(false);

function onTriggered(result) {
  if (result?.run_id) selectedRunId.value = result.run_id;
}

// ─── Helpers ────────────────────────────────────────────────────────────────
function formatRunLabel(run) {
  const date = run.start_date || run.logical_date;
  const formatted = date
    ? new Date(date).toLocaleString(undefined, {
        month: 'short', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
      })
    : run.run_id;
  return `${formatted} · ${run.status}`;
}
</script>
