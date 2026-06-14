<template>
  <div class="space-y-2">
    <span class="inline-flex items-center gap-1 text-sm font-medium text-fg">
      {{ label }}
      <span v-if="required" class="text-rose-600" aria-hidden="true">*</span>
      <ModelingFormHint v-if="hint" :text="hint" />
    </span>

    <div v-if="isLoading" class="flex items-center gap-2 text-sm text-fg-muted py-2">
      <Loader2 :size="16" class="animate-spin text-primary" />
      <span>Loading KPIs list...</span>
    </div>
    <div v-else-if="error" class="text-sm text-rose-600 py-2">
      Failed to load KPIs: {{ error.message || error || 'Unknown error' }}
    </div>
    <div v-else-if="kpis.length === 0" class="text-sm text-fg-muted py-2">
      No KPIs found in model config.
    </div>
    <div v-else class="space-y-3">
      <div class="relative">
        <input
          v-model="kpiSearchQuery"
          type="text"
          placeholder="Search KPIs..."
          class="w-full pl-9 pr-3 py-1.5 text-sm bg-bg-surface border border-border rounded-md placeholder-fg-muted focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-colors"
        />
        <span class="absolute left-3 top-1/2 -translate-y-1/2 text-fg-muted">
          <Search :size="14" />
        </span>
      </div>

      <div class="flex items-center justify-between border-b border-border pb-2">
        <span class="text-xs text-fg-muted">
          Selected: {{ modelValue.length }} / {{ kpis.length }}
        </span>
        <div class="flex gap-3 text-xs">
          <button
            type="button"
            class="text-primary hover:underline font-medium"
            @click="selectAllKpis"
          >
            Select all
          </button>
          <button
            type="button"
            class="text-primary hover:underline font-medium"
            @click="clearAllKpis"
          >
            Clear all
          </button>
        </div>
      </div>

      <div v-if="filteredKpis.length === 0" class="text-sm text-fg-muted py-4 text-center border border-dashed border-border rounded-md bg-bg-surface">
        No KPIs match your search.
      </div>
      <div v-else class="max-h-48 overflow-y-auto border border-border rounded-md p-3 grid grid-cols-2 gap-2 bg-bg-surface">
        <label
          v-for="kpi in filteredKpis"
          :key="kpi"
          class="flex items-center gap-2 text-sm text-fg cursor-pointer hover:bg-bg-hover p-1 rounded transition-colors"
        >
          <input
            type="checkbox"
            :value="kpi"
            :checked="modelValue.includes(kpi)"
            @change="toggleKpi(kpi)"
            class="rounded border-border text-primary focus:ring-primary h-4 w-4"
          />
          <span class="truncate" :title="kpi">{{ kpi }}</span>
        </label>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
import { Loader2, Search } from 'lucide-vue-next';
import ModelingFormHint from './ModelingFormHint.vue';

const props = defineProps({
  modelValue: { type: Array, required: true, default: () => [] },
  kpis: { type: Array, required: true, default: () => [] },
  isLoading: { type: Boolean, default: false },
  error: { type: [String, Object], default: null },
  label: { type: String, default: 'Select KPIs to generate' },
  hint: { type: String, default: '' },
  required: { type: Boolean, default: false },
});

const emit = defineEmits(['update:modelValue']);

const kpiSearchQuery = ref('');

const filteredKpis = computed(() => {
  const query = kpiSearchQuery.value.toLowerCase().trim();
  if (!query) return props.kpis;
  return props.kpis.filter((kpi) => kpi.toLowerCase().includes(query));
});

function toggleKpi(kpi) {
  const nextSelected = [...props.modelValue];
  const index = nextSelected.indexOf(kpi);
  if (index === -1) {
    nextSelected.push(kpi);
  } else {
    nextSelected.splice(index, 1);
  }
  emit('update:modelValue', nextSelected);
}

function selectAllKpis() {
  const currentSelected = new Set(props.modelValue);
  filteredKpis.value.forEach(kpi => currentSelected.add(kpi));
  emit('update:modelValue', Array.from(currentSelected));
}

function clearAllKpis() {
  const filteredSet = new Set(filteredKpis.value);
  const nextSelected = props.modelValue.filter(kpi => !filteredSet.has(kpi));
  emit('update:modelValue', nextSelected);
}

// Reset search query when the list of KPIs changes
watch(
  () => props.kpis,
  () => {
    kpiSearchQuery.value = '';
  }
);
</script>
