<!--
  DagListView - DAG dashboard.

  Table with clear columns and color status badges.
  Polling every 5s (Vue Query). Row click -> navigate to detail view.
-->
<template>
  <div class="space-y-6">
    <section class="rounded-xl border border-border-default bg-surface p-5 shadow-sm">
      <h2 class="text-lg font-semibold text-fg">DAG list</h2>
      <p class="mt-1 text-sm text-fg-muted">
        All DAGs from Airflow. Click a row to view the graph, trigger runs, and inspect logs.
      </p>
    </section>

    <!-- ── Toolbar / filter row ─────────────────────────────────────── -->
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div class="relative flex flex-1 items-center">
        <Search :size="14" class="absolute left-2.5 text-fg-subtle" />
        <input
          v-model="search"
          type="search"
          placeholder="Filter by name, tag, or owner..."
          class="w-full max-w-md rounded-md border border-border-default bg-surface py-2 pl-8 pr-3 text-sm text-fg
                 placeholder:text-fg-subtle
                 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand/40"
        />
      </div>
      <div class="flex items-center gap-2">
        <span class="text-xs text-fg-subtle">
          {{ visibleDags.length }} / {{ dags.length }} DAGs
        </span>
        <Button variant="secondary" size="sm" :disabled="isFetching" @click="onRefresh">
          <RefreshCw :size="14" :class="isFetching && 'animate-spin'" />
          Refresh
        </Button>
      </div>
    </div>

    <!-- ── Error banner ─────────────────────────────────────────────── -->
    <div
      v-if="isError"
      class="flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
    >
      <AlertCircle :size="16" class="mt-0.5 shrink-0" />
      <div class="space-y-1">
        <p class="font-semibold">Failed to fetch DAG list.</p>
        <p class="text-xs text-rose-600">{{ error?.message }}</p>
      </div>
    </div>

    <!-- ── Table ────────────────────────────────────────────────────── -->
    <div class="overflow-hidden rounded-xl border border-border-default bg-surface">
      <table class="w-full table-fixed text-sm">
        <thead class="border-b border-border-default bg-surface-muted text-left text-xs uppercase tracking-wide text-fg-subtle">
          <tr>
            <th class="w-[28%] px-4 py-3">DAG</th>
            <th class="w-[16%] px-4 py-3">Status</th>
            <th class="w-[18%] px-4 py-3">Last run</th>
            <th class="w-[14%] px-4 py-3">Duration</th>
            <th class="w-[12%] px-4 py-3">24h success</th>
            <th class="w-[10%] px-4 py-3 text-right">Schedule</th>
            <th class="w-[10%] px-4 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-if="isLoading"
            v-for="i in 5"
            :key="`s${i}`"
            class="border-b border-border-default last:border-0 animate-pulse"
          >
            <td class="px-4 py-4"><div class="h-3 w-3/4 rounded bg-slate-100" /></td>
            <td class="px-4 py-4"><div class="h-3 w-1/2 rounded bg-slate-100" /></td>
            <td class="px-4 py-4"><div class="h-3 w-2/3 rounded bg-slate-100" /></td>
            <td class="px-4 py-4"><div class="h-3 w-1/2 rounded bg-slate-100" /></td>
            <td class="px-4 py-4"><div class="h-3 w-1/3 rounded bg-slate-100" /></td>
            <td class="px-4 py-4 text-right"><div class="ml-auto h-3 w-1/2 rounded bg-slate-100" /></td>
            <td class="px-4 py-4 text-right"><div class="ml-auto h-3 w-1/2 rounded bg-slate-100" /></td>
          </tr>

          <tr
            v-else-if="visibleDags.length === 0"
            class="border-b border-border-default last:border-0"
          >
            <td colspan="7" class="px-4 py-10 text-center text-sm text-fg-muted">
              No DAGs to display.
            </td>
          </tr>

          <tr
            v-else
            v-for="dag in visibleDags"
            :key="dag.dag_id"
            @click="onSelect(dag)"
            class="cursor-pointer border-b border-border-default last:border-0 hover:bg-surface-muted"
          >
            <td class="px-4 py-3">
              <div class="flex items-center gap-2">
                <span
                  v-if="dag.is_paused"
                  class="inline-flex h-1.5 w-1.5 rounded-full bg-slate-400"
                  title="Paused"
                />
                <div class="min-w-0">
                  <p class="truncate font-medium text-fg">{{ dag.display_name }}</p>
                  <p class="truncate font-mono text-[11px] text-fg-subtle">{{ dag.dag_id }}</p>
                </div>
              </div>
            </td>
            <td class="px-4 py-3">
              <DagStatusBadge
                v-if="dag.last_run"
                :status="dag.last_run.status"
                density="compact"
              />
              <span v-else class="text-xs text-fg-subtle">Never ran</span>
            </td>
            <td class="px-4 py-3 text-xs text-fg-muted">
              <span v-if="dag.last_run">
                {{ formatRelative(dag.last_run.start_date || dag.last_run.logical_date) }}
              </span>
              <span v-else>—</span>
            </td>
            <td class="px-4 py-3 text-xs text-fg-muted">
              {{ formatDuration(dag.last_run?.duration_ms) }}
            </td>
            <td class="px-4 py-3">
              <span class="text-xs text-fg">
                {{ dag.stats_24h.success }}
                <span class="text-fg-subtle">/ {{ dag.stats_24h.total }}</span>
              </span>
            </td>
            <td class="px-4 py-3 text-right text-xs text-fg-muted">
              <span v-if="dag.schedule" class="font-mono">{{ dag.schedule }}</span>
              <span v-else>—</span>
            </td>
            <td class="px-4 py-3 text-right">
              <router-link
                :to="detailRoute(dag)"
                class="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-brand hover:bg-surface-muted"
              >
                Open
                <ChevronRight :size="14" />
              </router-link>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import { useRouter } from 'vue-router';
import { Search, RefreshCw, AlertCircle, ChevronRight } from 'lucide-vue-next';
import { Button } from '@/components/ui';
import DagStatusBadge from '../components/DagStatusBadge.vue';
import { useDagList } from '../composables/queries.js';

const router = useRouter();
const { data, isLoading, isError, error, isFetching, refetch } = useDagList();

const search = ref('');

const dags = computed(() => data.value ?? []);

const visibleDags = computed(() => {
  const q = search.value.trim().toLowerCase();
  if (!q) return dags.value;
  return dags.value.filter((dag) => {
    if (dag.dag_id.toLowerCase().includes(q)) return true;
    if (dag.display_name.toLowerCase().includes(q)) return true;
    if (dag.owners.some((o) => o.toLowerCase().includes(q))) return true;
    if (dag.tags.some((t) => t.toLowerCase().includes(q))) return true;
    return false;
  });
});

function detailRoute(dag) {
  return {
    name: 'DAG details',
    params: { dagId: dag.dag_id },
  };
}

function onSelect(dag) {
  router.push(detailRoute(dag));
}

function onRefresh() {
  refetch();
}

function formatDuration(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  const rest = Math.floor(s % 60);
  return `${m}m ${rest}s`;
}

function formatRelative(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const diff = Date.now() - d.getTime();
  if (diff < 0) return d.toLocaleString();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return d.toLocaleString();
}
</script>
