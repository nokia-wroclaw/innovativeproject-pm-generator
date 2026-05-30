<template>
  <article class="rounded-xl border border-border-default bg-surface p-5 shadow-sm">
    <div class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 class="text-lg font-semibold text-fg">{{ process.title }}</h2>
        <p class="mt-1 text-sm text-fg-muted">{{ process.description }}</p>
        <p class="mt-2 text-xs text-fg-subtle">
          DAG: <span class="font-mono text-fg">{{ process.dagId }}</span>
        </p>
      </div>
      <div class="flex flex-wrap items-center justify-end gap-2">
        <RouterLink :to="`/dags/${process.dagId}`" class="inline-flex">
          <Button variant="secondary">
            DAG details
          </Button>
        </RouterLink>
        <Button @click="$emit('configure', process.processType)">
          <Play :size="14" />
          Configure and run
        </Button>
      </div>
    </div>

    <div class="mt-5 rounded-lg border border-border-default bg-surface-muted p-4">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p class="text-xs uppercase tracking-wide text-fg-subtle">Run status</p>
          <p v-if="process.runId" class="mt-2 break-all font-mono text-xs text-fg">
            {{ process.runId }}
          </p>
          <p v-else class="mt-2 text-sm text-fg-muted">{{ process.emptyText }}</p>
        </div>
        <div class="flex items-center gap-2">
          <DagStatusBadge
            v-if="process.statusData"
            :status="process.statusData.status"
            density="compact"
          />
          <span v-else class="text-xs text-fg-muted">
            {{ process.runId ? 'Queued' : 'Not started' }}
          </span>
          <Loader2
            v-if="process.isPolling"
            :size="14"
            class="animate-spin text-fg-subtle"
          />
        </div>
      </div>
      <Button
        variant="secondary"
        size="sm"
        class="mt-4"
        :disabled="!process.runId"
        @click="$emit('refresh', process)"
      >
        <RefreshCw :size="14" />
        Refresh
      </Button>
    </div>

    <div
      v-if="process.error"
      class="mt-4 flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
    >
      <AlertCircle :size="16" class="mt-0.5 shrink-0" />
      <div>
        <p class="font-semibold">Failed to read run status.</p>
        <p class="text-xs">{{ process.error }}</p>
      </div>
    </div>

    <div class="mt-5 space-y-4">
      <div class="rounded-lg border border-border-default">
        <div class="border-b border-border-default px-4 py-3">
          <h3 class="text-sm font-semibold text-fg">Artifacts</h3>
        </div>
        <div v-if="artifacts.length" class="divide-y divide-border-default">
          <div
            v-for="artifact in artifacts"
            :key="artifact.kind"
            class="flex items-start gap-3 px-4 py-3"
          >
            <FileCheck2
              :size="16"
              :class="artifact.status === 'saved' ? 'text-emerald-500' : 'text-fg-subtle'"
              class="mt-0.5 shrink-0"
            />
            <div class="min-w-0">
              <p class="text-sm font-medium text-fg">{{ artifactLabel(artifact.kind) }}</p>
              <p class="break-all font-mono text-[11px] text-fg-muted">{{ artifact.path }}</p>
            </div>
          </div>
        </div>
        <div v-else class="flex items-center gap-2 p-4 text-sm text-fg-muted">
          <XCircle :size="16" class="text-fg-subtle" />
          Artifacts will appear after the run starts.
        </div>
      </div>

      <div class="rounded-lg border border-border-default">
        <div class="border-b border-border-default px-4 py-3">
          <h3 class="text-sm font-semibold text-fg">Summary</h3>
        </div>
        <div v-if="process.statusData?.metrics" class="grid grid-cols-2 gap-3 p-4">
          <div
            v-for="(value, key) in process.statusData.metrics"
            :key="key"
            class="rounded-md bg-surface-muted p-3"
          >
            <p class="text-xs uppercase tracking-wide text-fg-subtle">{{ key }}</p>
            <p class="mt-1 text-lg font-semibold text-fg">{{ value }}</p>
          </div>
        </div>
        <div v-else class="flex items-center gap-2 p-4 text-sm text-fg-muted">
          <XCircle :size="16" class="text-fg-subtle" />
          Summary will appear after the status is Success.
        </div>
      </div>
    </div>
  </article>
</template>

<script setup>
import { computed } from 'vue';
import { AlertCircle, FileCheck2, Loader2, Play, RefreshCw, XCircle } from 'lucide-vue-next';

import { Button } from '@/components/ui';
import DagStatusBadge from '@/features/dags/components/DagStatusBadge.vue';

const props = defineProps({
  process: { type: Object, required: true },
  artifactLabels: { type: Object, default: () => ({}) },
});

defineEmits(['configure', 'refresh']);

const artifacts = computed(() => props.process.statusData?.artifacts ?? []);

function artifactLabel(kind) {
  return props.artifactLabels[kind] ?? kind;
}
</script>
