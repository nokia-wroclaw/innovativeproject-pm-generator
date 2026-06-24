<template>
  <div
    class="space-y-3 rounded-lg border border-dashed border-border-default bg-surface-muted/60 p-4"
    :class="disabled ? 'pointer-events-none opacity-60' : ''"
  >
    <div class="flex items-center gap-2">
      <h3 class="text-sm font-semibold text-fg">AI-assisted configuration</h3>
    </div>
    <p class="text-xs text-fg-subtle">
      Describe only what you want to change; other fields keep their current values.
    </p>

    <ModelingFormTextarea
      v-model="instruction"
      label="Your instruction"
      :placeholder="placeholder"
      :rows="3"
      @update:model-value="clearError"
    />

    <div class="flex flex-wrap items-center gap-2">
      <Button
        type="button"
        variant="secondary"
        size="sm"
        :disabled="disabled || isAutofilling || !instruction.trim()"
        @click="onAutofill"
      >
        <Loader2 v-if="isAutofilling" :size="14" class="animate-spin" />
        {{ isAutofilling ? 'Filling…' : 'Auto-fill form' }}
      </Button>
    </div>

    <div
      v-if="errorMessage"
      class="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700"
    >
      {{ errorMessage }}
    </div>
  </div>
</template>

<script setup>
import { toRaw } from 'vue';
import { Loader2, Sparkles } from 'lucide-vue-next';

import { Button } from '@/components/ui';
import { useModelingAutofill } from '../composables/useModelingAutofill.js';
import ModelingFormTextarea from './form-fields/ModelingFormTextarea.vue';

const PLACEHOLDERS = {
  preprocessing_feature_engineering:
    'e.g. Dataset #1, working days, 168h windows, enable imputation, output to s3://…/preprocessed…',
  training_dataset:
    'e.g. Dataset #2, weekends, target column "sales", 20% test split, shuffle on…',
  generate:
    'e.g. Use model #2, set encoder to models/enc.pkl, config to models/cfg.json, compare with dataset #1, KPIs: kpi_a kpi_b…',
};

const props = defineProps({
  processType: { type: String, required: true },
  currentValues: { type: Object, required: true },
  disabled: { type: Boolean, default: false },
});

const emit = defineEmits(['apply']);

const placeholder = PLACEHOLDERS[props.processType] ?? 'Describe how to configure this process…';

const { instruction, isAutofilling, errorMessage, runAutofill, clearError } =
  useModelingAutofill(props.processType);

async function onAutofill() {
  const values = await runAutofill(toRaw(props.currentValues));
  if (values) {
    emit('apply', values);
  }
}
</script>
