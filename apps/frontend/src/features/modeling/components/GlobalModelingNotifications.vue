<template>
  <div class="fixed bottom-5 right-5 z-50 space-y-3">
    <div
      v-for="toast in state.toasts"
      :key="toast.id"
      class="w-[340px] rounded-xl border bg-surface p-4 shadow-lg"
      :class="toast.variant === 'success' ? 'border-emerald-200' : 'border-rose-200'"
    >
      <div class="flex items-start gap-3">
        <CheckCircle2
          v-if="toast.variant === 'success'"
          :size="18"
          class="mt-0.5 shrink-0 text-emerald-500"
        />
        <AlertCircle v-else :size="18" class="mt-0.5 shrink-0 text-rose-500" />
        <div class="min-w-0">
          <p class="text-sm font-semibold text-fg">{{ toast.title }}</p>
          <p class="mt-1 text-xs text-fg-muted">{{ toast.message }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { AlertCircle, CheckCircle2 } from 'lucide-vue-next';

import { modelingRunMonitorState as state } from '../services/modelingRunMonitor.js';
</script>
