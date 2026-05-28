<!--
  TaskNode — custom Vue Flow node for a single Airflow task.

  Registered against Vue Flow with:
    <VueFlow :node-types="{ task: TaskNode }" />

  Expected `data` shape (passed by Vue Flow on each node):
    {
      label:       string,                    // human-readable task label
      task_id:     string,                    // raw task_id (for tooltips)
      operator:    string,                    // e.g. "PythonOperator"
      status:      TaskStatus,                // project-level enum
      try_number?: number,                    // current try
      max_tries?:  number,
      duration_ms?: number | null,
      is_group?:   boolean,                   // TaskGroup
    }

  Design rules (contract §2.1 + Faza 1 §1.4):
    - The whole card is a single click target.
    - Selected state uses the brand ring (focus-visible-style).
    - Status is communicated by a colored left bar + the status badge; never
      by changing the entire background — keeps the graph scannable when
      zoomed out.
    - Min size 220×60 so labels fit at typical zoom levels.
-->
<template>
  <div
    :class="cn(
      'group relative min-w-[220px] max-w-[260px] rounded-lg border bg-surface',
      'shadow-sm transition-shadow duration-150',
      'hover:shadow-md',
      selected
        ? 'border-brand ring-2 ring-brand/30'
        : 'border-border-default',
    )"
  >
    <Handle
      type="target"
      :position="Position.Top"
      :class="handleClass"
    />

    <!-- Colored status bar on the left edge. -->
    <span
      :class="cn('absolute inset-y-0 left-0 w-1 rounded-l-lg', statusBarClass)"
      aria-hidden="true"
    />

    <div class="flex items-center justify-between gap-3 px-3 py-2.5 pl-4">
      <div class="min-w-0 flex-1">
        <div class="flex items-center gap-1.5">
          <Layers v-if="data.is_group" :size="12" class="shrink-0 text-fg-subtle" />
          <p class="truncate text-sm font-semibold text-fg" :title="data.task_id ?? data.label">
            {{ data.label }}
          </p>
        </div>
        <p class="mt-0.5 truncate text-[11px] text-fg-muted">
          {{ data.operator }}<template v-if="metaSuffix"> · {{ metaSuffix }}</template>
        </p>
      </div>

      <TaskStatusBadge :status="data.status" density="icon-only" />
    </div>

    <Handle
      type="source"
      :position="Position.Bottom"
      :class="handleClass"
    />
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { Handle, Position } from '@vue-flow/core';
import { Layers } from 'lucide-vue-next';
import { cn } from '@/lib/cn';
import TaskStatusBadge from './TaskStatusBadge.vue';

const props = defineProps({
  /** Vue Flow injects these; we only declare what we use. */
  data: { type: Object, required: true },
  selected: { type: Boolean, default: false },
});

const statusBarClass = computed(() => {
  const map = {
    success: 'bg-emerald-500',
    running: 'bg-sky-500',
    failed: 'bg-rose-500',
    up_for_retry: 'bg-amber-500',
    queued: 'bg-violet-400',
    skipped: 'bg-slate-400',
    none: 'bg-slate-300',
  };
  return map[props.data.status] ?? map.none;
});

const metaSuffix = computed(() => {
  const parts = [];
  if (props.data.duration_ms != null) {
    parts.push(formatDuration(props.data.duration_ms));
  }
  if (props.data.try_number != null && props.data.max_tries != null && props.data.max_tries > 1) {
    parts.push(`Try ${props.data.try_number}/${props.data.max_tries}`);
  }
  return parts.join(' · ');
});

/** @param {number} ms */
function formatDuration(ms) {
  if (ms < 1000) return `${ms} ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}m ${rest}s`;
}

const handleClass = '!h-2 !w-2 !border !border-slate-300 !bg-white';
</script>
