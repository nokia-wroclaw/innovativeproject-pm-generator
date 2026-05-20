<!--
  LogViewer — historical + live task logs with level/regex filtering.

  Behaviour (contract §4 + §5):
    * Mounted with `tryNumber = latest` (or a chosen historical try) →
      opens an SSE stream and renders chunks as they arrive.
    * Mounted with a non-latest try → fetches paginated history once,
      no SSE.
    * On try change or component unmount, the previous stream is aborted.
    * Filters (level checkboxes, regex pattern) are client-side only —
      we never re-fetch on filter change.

  We deliberately do *not* virtualise the list yet. Real Airflow task logs
  are usually < 50k lines; if we hit perf issues we'll plug in
  `@tanstack/vue-virtual`.
-->
<template>
  <div class="flex h-full min-h-0 flex-col">
    <!-- ── Toolbar ──────────────────────────────────────────────────── -->
    <div class="flex flex-wrap items-center gap-3 border-b border-border-default px-4 py-3">
      <label class="flex items-center gap-2 text-xs font-medium text-fg-muted">
        Try
        <select
          :value="selectedTry"
          @change="onSelectTry"
          class="rounded-md border border-border-default bg-surface px-2 py-1 text-xs text-fg
                 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
        >
          <option v-for="t in availableTries" :key="t.try_number" :value="t.try_number">
            Try {{ t.try_number }} · {{ t.status }}
          </option>
          <option v-if="!availableTries.length" :value="initialTryNumber">
            Try {{ initialTryNumber }}
          </option>
        </select>
      </label>

      <div class="flex items-center gap-1">
        <button
          v-for="level in LEVELS"
          :key="level"
          type="button"
          @click="toggleLevel(level)"
          :class="cn(
            'rounded-md border px-2 py-1 text-[11px] font-semibold uppercase tracking-wide transition-colors',
            activeLevels.has(level)
              ? levelBadgeClasses[level]
              : 'border-border-default text-fg-subtle hover:bg-surface-muted',
          )"
        >
          {{ level }}
        </button>
      </div>

      <div class="relative flex flex-1 items-center">
        <Search :size="14" class="absolute left-2.5 text-fg-subtle" />
        <input
          v-model="regexPattern"
          type="search"
          placeholder="Filter (regex, case-insensitive)…"
          class="w-full rounded-md border border-border-default bg-surface py-1.5 pl-8 pr-2 text-xs text-fg
                 placeholder:text-fg-subtle
                 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
        />
      </div>

      <div class="flex items-center gap-2 text-[11px] text-fg-muted">
        <span
          :class="cn(
            'inline-flex h-1.5 w-1.5 rounded-full',
            streamState === 'live' && 'bg-sky-500 animate-pulse',
            streamState === 'done' && 'bg-emerald-500',
            streamState === 'error' && 'bg-rose-500',
            streamState === 'idle' && 'bg-slate-300',
          )"
        />
        <span>{{ statusLabel }}</span>
        <button
          type="button"
          @click="onClear"
          class="ml-1 rounded p-1 text-fg-muted hover:bg-surface-muted hover:text-fg"
          title="Clear buffer"
        >
          <Trash2 :size="14" />
        </button>
      </div>
    </div>

    <!-- ── Stream body ──────────────────────────────────────────────── -->
    <div
      ref="scrollRef"
      class="min-h-0 flex-1 overflow-auto bg-slate-950 font-mono text-[12px] leading-relaxed text-slate-100"
      @scroll="onScroll"
    >
      <div v-if="!filteredLines.length" class="p-6 text-center text-slate-500">
        <span v-if="streamState === 'error'">Stream error — see badge for details.</span>
        <span v-else-if="streamState === 'live' && !lines.length">Waiting for first log lines…</span>
        <span v-else-if="!lines.length">No logs yet.</span>
        <span v-else>No lines match the current filter.</span>
      </div>

      <div v-else class="px-4 py-3">
        <div
          v-for="(line, idx) in filteredLines"
          :key="idx"
          class="grid grid-cols-[120px_60px_1fr] gap-3 py-0.5"
        >
          <span class="text-slate-500">{{ formatTime(line.timestamp) }}</span>
          <span :class="cn('text-[11px] font-bold uppercase', levelTextClasses[line.level] ?? 'text-slate-500')">
            {{ line.level ?? '' }}
          </span>
          <span class="whitespace-pre-wrap break-words">{{ line.message }}</span>
        </div>
        <div class="py-2 text-center text-slate-600">
          <span v-if="streamState === 'live'">— streaming —</span>
          <span v-else-if="streamState === 'done'">— end of log —</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {
  computed, nextTick, onBeforeUnmount, reactive, ref, useTemplateRef, watch,
} from 'vue';
import { Search, Trash2 } from 'lucide-vue-next';
import { cn } from '@/lib/cn';
import { consumeLogStream, getTaskLogs } from '../services/dagsApi.js';

const props = defineProps({
  dagId: { type: String, required: true },
  runId: { type: String, required: true },
  taskId: { type: String, required: true },
  /** Task's current try (highest try_number from TaskInstance). */
  currentTryNumber: { type: Number, required: true },
  /** Try the user actively wants to view (defaults to currentTryNumber). */
  initialTryNumber: { type: Number, default: null },
  /** Array of TaskTry for the dropdown; may be empty until loaded. */
  availableTries: { type: Array, default: () => [] },
});

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const selectedTry = ref(props.initialTryNumber ?? props.currentTryNumber);
const activeLevels = reactive(new Set(LEVELS)); // all on by default
const regexPattern = ref('');
/** @type {import('vue').Ref<import('../types.js').LogLine[]>} */
const lines = ref([]);
/** @type {import('vue').Ref<'idle' | 'live' | 'done' | 'error'>} */
const streamState = ref('idle');
const errorMessage = ref('');

const scrollRef = useTemplateRef('scrollRef');
const stickyBottom = ref(true);

let abortController = null;

// ─── Toolbar handlers ───────────────────────────────────────────────────────
function toggleLevel(level) {
  if (activeLevels.has(level)) activeLevels.delete(level);
  else activeLevels.add(level);
}

function onSelectTry(event) {
  selectedTry.value = Number(event.target.value);
}

function onClear() {
  lines.value = [];
}

// ─── Scroll auto-stick to bottom ────────────────────────────────────────────
function onScroll() {
  if (!scrollRef.value) return;
  const el = scrollRef.value;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
  stickyBottom.value = atBottom;
}

async function scrollToBottomIfSticky() {
  if (!stickyBottom.value) return;
  await nextTick();
  if (scrollRef.value) {
    scrollRef.value.scrollTop = scrollRef.value.scrollHeight;
  }
}

// ─── Filtering ──────────────────────────────────────────────────────────────
const compiledRegex = computed(() => {
  const pattern = regexPattern.value.trim();
  if (!pattern) return null;
  try {
    return new RegExp(pattern, 'i');
  } catch {
    return null;
  }
});

const filteredLines = computed(() => {
  const rx = compiledRegex.value;
  return lines.value.filter((line) => {
    if (line.level && !activeLevels.has(line.level)) return false;
    if (rx && !rx.test(line.message)) return false;
    return true;
  });
});

// ─── Status label ───────────────────────────────────────────────────────────
const statusLabel = computed(() => {
  if (streamState.value === 'error') return errorMessage.value || 'Error';
  if (streamState.value === 'live') return 'Live';
  if (streamState.value === 'done') return 'Done';
  return 'Idle';
});

// ─── Stream lifecycle ───────────────────────────────────────────────────────
async function startStream() {
  closeStream();
  lines.value = [];
  errorMessage.value = '';

  const isLatest = selectedTry.value === props.currentTryNumber;

  if (!isLatest) {
    // Historical — fetch (paginated) once.
    streamState.value = 'idle';
    try {
      await fetchHistorical(selectedTry.value);
      streamState.value = 'done';
    } catch (err) {
      streamState.value = 'error';
      errorMessage.value = err?.message ?? 'Fetch failed';
    }
    return;
  }

  // Live SSE for the current attempt.
  streamState.value = 'live';
  abortController = new AbortController();
  consumeLogStream(props.dagId, props.runId, props.taskId, {
    tryNumber: selectedTry.value,
    signal: abortController.signal,
    onEvent: onSseEvent,
  }).catch((err) => {
    if (err?.name === 'AbortError') return;
    streamState.value = 'error';
    errorMessage.value = err?.message ?? 'Stream error';
  });
}

function onSseEvent(event) {
  if (event.type === 'chunk') {
    appendLines(event.data?.lines ?? []);
  } else if (event.type === 'end') {
    if (event.data?.reason === 'max_duration') {
      // Restart transparently for >2h streams.
      streamState.value = 'live';
      startStream();
    } else {
      streamState.value = 'done';
    }
  } else if (event.type === 'error') {
    streamState.value = 'error';
    errorMessage.value = event.data?.message ?? 'Stream error';
  }
}

function appendLines(newLines) {
  if (!newLines.length) return;
  lines.value = lines.value.concat(newLines);
  scrollToBottomIfSticky();
}

async function fetchHistorical(tryNumber) {
  let token = null;
  /* Guard against pathological responses. */
  for (let i = 0; i < 50; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    const chunk = await getTaskLogs(props.dagId, props.runId, props.taskId, {
      tryNumber,
      token,
    });
    appendLines(chunk.lines ?? []);
    if (!chunk.has_more || !chunk.continuation) return;
    token = chunk.continuation;
  }
}

function closeStream() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}

// ─── Watchers ───────────────────────────────────────────────────────────────
watch(
  () => [props.dagId, props.runId, props.taskId, selectedTry.value],
  () => {
    void startStream();
  },
  { immediate: true },
);

watch(
  () => props.currentTryNumber,
  (newTry) => {
    if (!selectedTry.value) selectedTry.value = newTry;
  },
);

onBeforeUnmount(closeStream);

// ─── Formatting / classes ───────────────────────────────────────────────────
function formatTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(11, 19);
  return d.toLocaleTimeString(undefined, { hour12: false });
}

const levelBadgeClasses = {
  DEBUG: 'border-slate-300 bg-slate-50 text-slate-600',
  INFO: 'border-sky-300 bg-sky-50 text-sky-700',
  WARNING: 'border-amber-300 bg-amber-50 text-amber-700',
  ERROR: 'border-rose-300 bg-rose-50 text-rose-700',
  CRITICAL: 'border-rose-400 bg-rose-100 text-rose-800',
};

const levelTextClasses = {
  DEBUG: 'text-slate-400',
  INFO: 'text-sky-400',
  WARNING: 'text-amber-400',
  ERROR: 'text-rose-400',
  CRITICAL: 'text-rose-400',
};
</script>
