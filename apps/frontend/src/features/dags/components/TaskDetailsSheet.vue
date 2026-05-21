<!--
  TaskDetailsSheet — slide-in panel triggered by clicking a node on the graph.

  Three tabs (contract §4):
    1. Overview — task instance metadata + actions (Retry task, Stop run).
    2. Logs     — placeholder; live LogViewer will land in Faza 3.

  This component is *presentation only* in Faza 2b: it accepts a
  `taskInstance` prop (may be null) and emits events for action buttons.
  Data fetching and SSE hookup live one level above (Faza 3).
-->
<template>
  <Sheet :open="open" side="right" @update:open="$emit('update:open', $event)">
    <template #title>
      <span class="font-mono text-sm">{{ taskInstance?.task_id ?? '—' }}</span>
    </template>
    <template #description>
      <span v-if="taskInstance">
        Run <span class="font-mono">{{ taskInstance.run_id }}</span>
        · Try {{ taskInstance.try_number }}/{{ taskInstance.max_tries }}
      </span>
      <span v-else>No task selected</span>
    </template>

    <Tabs
      v-model="activeTab"
      :items="[
        { value: 'overview', label: 'Overview' },
        { value: 'logs', label: 'Logs' },
      ]"
    >
      <!-- ── Overview ─────────────────────────────────────────────── -->
      <template #overview>
        <div v-if="taskInstance" class="space-y-5">
          <section class="space-y-2">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-fg-subtle">
              Status
            </h3>
            <TaskStatusBadge :status="taskInstance.status" />
            <p class="text-[11px] text-fg-subtle">
              Airflow raw state: <span class="font-mono">{{ taskInstance.raw_state }}</span>
            </p>
          </section>

          <section class="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            <DefRow label="Operator">{{ taskInstance.operator }}</DefRow>
            <DefRow label="Pool">{{ taskInstance.pool }}</DefRow>
            <DefRow label="Queue">{{ taskInstance.queue }}</DefRow>
            <DefRow label="Duration">
              {{ formatDuration(taskInstance.duration_ms) }}
            </DefRow>
            <DefRow label="Started">
              {{ formatTimestamp(taskInstance.start_date) }}
            </DefRow>
            <DefRow label="Ended">
              {{ formatTimestamp(taskInstance.end_date) }}
            </DefRow>
          </section>

          <section v-if="taskInstance.note" class="space-y-1.5">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-fg-subtle">
              Note
            </h3>
            <p class="rounded-md bg-surface-muted px-3 py-2 text-sm text-fg">
              {{ taskInstance.note }}
            </p>
          </section>
        </div>
        <EmptyState
          v-else
          message="Select a task on the graph to view details."
        />
      </template>

      <!-- ── Logs ─────────────────────────────────────────────────── -->
      <template #logs>
        <LogViewer
          v-if="taskInstance && dagId && runId"
          :dag-id="dagId"
          :run-id="runId"
          :task-id="taskInstance.task_id"
          :current-try-number="taskInstance.try_number || 1"
          :initial-try-number="taskInstance.try_number || 1"
          :available-tries="availableTries"
          :task-status="taskInstance.status"
        />
        <EmptyState
          v-else
          icon-name="terminal"
          message="Select a task to view its logs."
        />
      </template>

    </Tabs>

    <template #footer>
      <div class="flex items-center justify-between gap-3">
        <p class="text-[11px] text-fg-subtle">
          Actions affect the live Airflow scheduler.
        </p>
        <div class="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            :disabled="!canStopRun"
            @click="$emit('stop-run')"
          >
            <Square :size="14" />
            Stop run
          </Button>
          <Button
            variant="primary"
            size="sm"
            :disabled="!canRetry"
            @click="$emit('retry-task')"
          >
            <RotateCw :size="14" />
            Retry task
          </Button>
        </div>
      </div>
    </template>
  </Sheet>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { RotateCw, Square } from 'lucide-vue-next';
import { Sheet, Tabs, Button } from '@/components/ui';
import TaskStatusBadge from './TaskStatusBadge.vue';
import LogViewer from './LogViewer.vue';
import DefRow from './_internal/DefRow.vue';
import EmptyState from './_internal/EmptyState.vue';

const props = defineProps({
  open: { type: Boolean, required: true },
  /** @type {import('@/features/dags/types.js').TaskInstance | null} */
  taskInstance: { type: Object, default: null },
  /** Used by the Logs tab. */
  dagId: { type: String, default: null },
  runId: { type: String, default: null },
  /** Optional list of tries for the dropdown inside the Logs tab. */
  availableTries: { type: Array, default: () => [] },
});
defineEmits(['update:open', 'retry-task', 'stop-run']);

const activeTab = ref('overview');

// Reset to Overview each time a fresh task is opened.
watch(
  () => props.taskInstance?.task_id,
  () => { activeTab.value = 'overview'; },
);

const canRetry = computed(() => {
  if (!props.taskInstance) return false;
  return ['failed', 'success', 'up_for_retry', 'skipped'].includes(props.taskInstance.status);
});

const canStopRun = computed(() => {
  if (!props.taskInstance) return false;
  return ['running', 'queued', 'up_for_retry'].includes(props.taskInstance.status);
});

/** @param {number | null | undefined} ms */
function formatDuration(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  const rest = Math.floor(s % 60);
  return `${m}m ${rest}s`;
}

/** @param {string | null | undefined} iso */
function formatTimestamp(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}
</script>
