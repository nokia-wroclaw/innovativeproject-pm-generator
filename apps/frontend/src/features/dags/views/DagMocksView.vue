<!--
  DagMocksView — designerska strona-piaskownica.

  Route: /_dev/dag-mocks (rejestrowane tylko w `import.meta.env.DEV`).

  Pokazuje wszystkie warianty komponentów DAG management na mockowych
  danych. Pozwala oceniać czytelność statusów, animacje, panel boczny
  i layout grafu *zanim* wpieprzymy się w integrację z Airflow API.

  Jeśli ten widok wygląda dobrze — gotowe komponenty są gotowe na Fazę 3.
-->
<template>
  <div class="mx-auto max-w-6xl space-y-12 pb-20">
    <header class="space-y-2">
      <p class="text-xs font-semibold uppercase tracking-wider text-fg-subtle">
        Dev preview
      </p>
      <h1 class="text-2xl font-semibold text-fg">DAG management — komponenty</h1>
      <p class="max-w-2xl text-sm text-fg-muted">
        Strona piaskownica dla Fazy 2b. Mockowe dane. Po akceptacji wizualnej
        wpinamy komponenty w prawdziwy widok DAG-a (Faza 3).
      </p>
    </header>

    <!-- ── Section: Status badges ───────────────────────────────────── -->
    <section class="space-y-5">
      <h2 class="border-b border-border-default pb-2 text-sm font-semibold uppercase tracking-wider text-fg-subtle">
        Status badges
      </h2>

      <div class="grid grid-cols-1 gap-8 lg:grid-cols-2">
        <div class="space-y-3">
          <p class="text-xs font-medium text-fg-muted">TaskStatusBadge — 7 stanów</p>
          <div class="flex flex-wrap items-center gap-2 rounded-lg border border-border-default bg-surface p-4">
            <TaskStatusBadge v-for="s in taskStatuses" :key="s" :status="s" />
          </div>
          <div class="flex flex-wrap items-center gap-2 rounded-lg border border-border-default bg-surface p-4">
            <TaskStatusBadge v-for="s in taskStatuses" :key="s" :status="s" density="compact" />
          </div>
          <div class="flex flex-wrap items-center gap-2 rounded-lg border border-border-default bg-surface p-4">
            <TaskStatusBadge v-for="s in taskStatuses" :key="s" :status="s" density="icon-only" />
          </div>
        </div>

        <div class="space-y-3">
          <p class="text-xs font-medium text-fg-muted">DagStatusBadge — 4 stany</p>
          <div class="flex flex-wrap items-center gap-2 rounded-lg border border-border-default bg-surface p-4">
            <DagStatusBadge v-for="s in dagStatuses" :key="s" :status="s" />
          </div>
          <div class="flex flex-wrap items-center gap-2 rounded-lg border border-border-default bg-surface p-4">
            <DagStatusBadge v-for="s in dagStatuses" :key="s" :status="s" density="compact" />
          </div>
          <div class="flex flex-wrap items-center gap-2 rounded-lg border border-border-default bg-surface p-4">
            <DagStatusBadge v-for="s in dagStatuses" :key="s" :status="s" density="icon-only" />
          </div>
        </div>
      </div>
    </section>

    <!-- ── Section: Buttons ─────────────────────────────────────────── -->
    <section class="space-y-5">
      <h2 class="border-b border-border-default pb-2 text-sm font-semibold uppercase tracking-wider text-fg-subtle">
        Buttons
      </h2>
      <div class="flex flex-wrap items-center gap-3 rounded-lg border border-border-default bg-surface p-4">
        <Button variant="primary">Trigger DAG</Button>
        <Button variant="secondary">Open in Airflow</Button>
        <Button variant="ghost">Refresh</Button>
        <Button variant="danger">
          <Square :size="14" />
          Stop run
        </Button>
        <Button variant="primary" size="sm">
          <RotateCw :size="14" />
          Retry task
        </Button>
        <Button variant="secondary" disabled>Disabled</Button>
      </div>
    </section>

    <!-- ── Section: Graph + side panel ──────────────────────────────── -->
    <section class="space-y-5">
      <h2 class="border-b border-border-default pb-2 text-sm font-semibold uppercase tracking-wider text-fg-subtle">
        DAG graph + details panel
      </h2>
      <p class="text-sm text-fg-muted">
        Kliknij dowolny węzeł — wysunie się panel boczny ze szczegółami taska.
        Pozycje węzłów są tutaj zakodowane ręcznie; w prawdziwym widoku użyjemy
        auto-layoutu dagre.
      </p>

      <div class="h-[520px] overflow-hidden rounded-xl border border-border-default bg-surface-muted">
        <VueFlow
          :nodes="SAMPLE_GRAPH.nodes"
          :edges="SAMPLE_GRAPH.edges"
          :node-types="{ task: TaskNode }"
          :nodes-draggable="false"
          :nodes-connectable="false"
          :elements-selectable="true"
          :default-edge-options="defaultEdgeOptions"
          :fit-view-on-init="true"
          @node-click="handleNodeClick"
        >
          <Background :gap="20" :size="1" class="!bg-surface-muted" />
          <MiniMap
            pannable
            zoomable
            class="!bg-surface !border !border-border-default !rounded-lg"
          />
          <Controls
            position="bottom-right"
            class="!bg-surface !border !border-border-default !rounded-lg !shadow-sm"
          />
        </VueFlow>
      </div>
    </section>

    <TaskDetailsSheet
      v-model:open="sheetOpen"
      :task-instance="selectedTaskInstance"
      @retry-task="onRetry"
      @stop-run="onStop"
    />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import { VueFlow, MarkerType } from '@vue-flow/core';
import { Background } from '@vue-flow/background';
import { Controls } from '@vue-flow/controls';
import { MiniMap } from '@vue-flow/minimap';
import { RotateCw, Square } from 'lucide-vue-next';

import '@vue-flow/core/dist/style.css';
import '@vue-flow/core/dist/theme-default.css';
import '@vue-flow/controls/dist/style.css';
import '@vue-flow/minimap/dist/style.css';

import { Button } from '@/components/ui';
import TaskStatusBadge from '../components/TaskStatusBadge.vue';
import DagStatusBadge from '../components/DagStatusBadge.vue';
import TaskNode from '../components/TaskNode.vue';
import TaskDetailsSheet from '../components/TaskDetailsSheet.vue';
import { SAMPLE_GRAPH, SAMPLE_TASK_INSTANCES } from '../__mocks__/sample.js';

const taskStatuses = ['success', 'running', 'failed', 'up_for_retry', 'queued', 'skipped', 'none'];
const dagStatuses = ['success', 'running', 'failed', 'queued'];

const defaultEdgeOptions = {
  type: 'smoothstep',
  style: { stroke: '#94a3b8', strokeWidth: 1.5 },
  markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8', width: 16, height: 16 },
};

const sheetOpen = ref(false);
const selectedTaskId = ref(null);

const selectedTaskInstance = computed(() => {
  if (!selectedTaskId.value) return null;
  return SAMPLE_TASK_INSTANCES.find((ti) => ti.task_id === selectedTaskId.value) ?? null;
});

function handleNodeClick({ node }) {
  selectedTaskId.value = node.id;
  sheetOpen.value = true;
}

function onRetry() {
  // eslint-disable-next-line no-console
  console.info('[mock] Retry task', selectedTaskId.value);
}

function onStop() {
  // eslint-disable-next-line no-console
  console.info('[mock] Stop run for', selectedTaskId.value);
}
</script>
