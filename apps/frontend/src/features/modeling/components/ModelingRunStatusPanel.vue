<template>
  <div class="space-y-3 rounded-lg border border-border-default bg-surface-muted p-4">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <p class="text-sm font-semibold text-fg">Run status</p>
      <div class="flex items-center gap-2">
        <DagStatusBadge v-if="statusData" :status="statusData.status" density="compact" />
        <span v-else-if="submitting" class="text-xs text-fg-muted">Triggering…</span>
        <Loader2 v-if="submitting || polling" :size="14" class="animate-spin text-fg-subtle" />
      </div>
    </div>

    <p v-if="submitting" class="text-sm text-fg-muted">
      Sending trigger to Airflow and waiting for the DAG run to appear…
    </p>

    <template v-else-if="runId">
      <p class="break-all font-mono text-xs text-fg">{{ runId }}</p>
      <p v-if="statusData?.raw_state" class="text-xs text-fg-muted">
        Airflow state: <span class="font-mono">{{ statusData.raw_state }}</span>
      </p>
      <p v-else-if="statusError" class="text-xs text-rose-600">{{ statusError }}</p>
      <p v-else class="text-xs text-fg-muted">Waiting for the first status update…</p>
    </template>
  </div>
</template>

<script setup>
import { Loader2 } from 'lucide-vue-next';

import DagStatusBadge from '@/features/dags/components/DagStatusBadge.vue';

defineProps({
  submitting: { type: Boolean, default: false },
  polling: { type: Boolean, default: false },
  runId: { type: String, default: null },
  statusData: { type: Object, default: null },
  statusError: { type: String, default: '' },
});
</script>
