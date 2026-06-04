<template>
  <BaseModal
    :show="show"
    :title="process.title"
    width="680px"
    @close="onClose"
  >
    <div class="space-y-4">
      <p class="text-sm text-fg-muted">
        Process configuration
        <span class="font-mono text-fg">training_dataset</span>.
      </p>

      <ModelingRunStatusPanel
        v-if="phase !== 'form'"
        :submitting="phase === 'submitting'"
        :polling="isPolling"
        :run-id="startedRun?.run_id ?? null"
        :status-data="statusData"
        :status-error="statusQuery.error.value?.message ?? ''"
      />

      <form
        class="space-y-4"
        :class="phase !== 'form' && phase !== 'error' ? 'pointer-events-none opacity-60' : ''"
        @submit.prevent="submit"
      >
        <ModelingFormSelect
          v-model="form.dataset_id"
          label="Input dataset"
          hint="Only datasets with COMPLETED status are shown."
          placeholder="Select dataset"
          :options="datasetOptions"
          value-type="number"
        />

        <ModelingFormRadioGroup
          v-model="form.dataset_type"
          label="Dataset type"
          :options="datasetTypeOptions"
        />

        <div class="space-y-4 rounded-lg border border-border-default bg-surface-muted p-4">
          <h3 class="text-sm font-semibold text-fg">Training dataset creation</h3>

          <ModelingFormInput
            v-model="form.target_column"
            label="Target column name"
            type="text"
            placeholder="np. sales"
          />

          <ModelingFormInput
            v-model="form.test_size"
            label="Test set size"
            type="number"
            min="0.05"
            max="0.5"
            step="0.01"
            value-type="number"
          />

          <ModelingFormInput
            v-model="form.random_seed"
            label="Random seed"
            type="number"
            min="1"
            step="1"
            value-type="number"
          />

          <ModelingFormCalendar
            v-model="form.split_date"
            label="Split cutoff date (optional)"
            hint="If provided, the backend may use it for a time-based split."
          />

          <ModelingFormCheckbox
            v-model="form.shuffle"
            label="Shuffle samples before split"
          />

          <ModelingFormCheckbox
            v-model="form.stratify"
            label="Stratified split (when target is categorical)"
          />
        </div>

        <div
          v-if="formError"
          class="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
        >
          {{ formError }}
        </div>
      </form>
    </div>

    <template #footer>
      <Button variant="secondary" :disabled="isSubmitting" @click="onClose">
        {{ phase === 'running' ? 'Close' : 'Cancel' }}
      </Button>
      <Button
        v-if="phase === 'form' || phase === 'error'"
        :disabled="isSubmitting || isSubmitDisabled"
        @click="submit"
      >
        <Loader2 v-if="isSubmitting" :size="14" class="animate-spin" />
        {{ isSubmitting ? 'Triggering…' : 'Run DAG' }}
      </Button>
    </template>
  </BaseModal>
</template>

<script setup>
import { computed, reactive, watch } from 'vue';
import { Loader2 } from 'lucide-vue-next';

import BaseModal from '@/components/BaseModal.vue';
import { Button } from '@/components/ui';
import { useModelingProcessRun } from '../composables/useModelingProcessRun.js';
import ModelingFormCalendar from './form-fields/ModelingFormCalendar.vue';
import ModelingFormCheckbox from './form-fields/ModelingFormCheckbox.vue';
import ModelingFormInput from './form-fields/ModelingFormInput.vue';
import ModelingFormRadioGroup from './form-fields/ModelingFormRadioGroup.vue';
import ModelingFormSelect from './form-fields/ModelingFormSelect.vue';
import ModelingRunStatusPanel from './ModelingRunStatusPanel.vue';

const props = defineProps({
  show: { type: Boolean, required: true },
  process: { type: Object, required: true },
  datasets: { type: Array, default: () => [] },
});

const emit = defineEmits(['close', 'started']);

const {
  phase,
  formError,
  startedRun,
  statusData,
  statusQuery,
  isSubmitting,
  isPolling,
  triggerRun,
  reset,
} = useModelingProcessRun(props.process.processType, props.process.title);

const datasetTypeOptions = [
  { value: 'working_days', label: 'Working days' },
  { value: 'weekends', label: 'Weekends' },
];

const form = reactive({
  dataset_id: '',
  dataset_type: 'working_days',
  target_column: '',
  test_size: 0.2,
  random_seed: 42,
  split_date: '',
  shuffle: true,
  stratify: false,
});

const isSubmitDisabled = computed(() => !form.dataset_id || !form.target_column?.trim());
const datasetOptions = computed(() =>
  props.datasets.map((dataset) => ({
    value: dataset.id,
    label: `#${dataset.id} · ${dataset.file_name} · ${dataset.type} · ${dataset.status}`,
  })),
);

watch(
  () => props.show,
  (isOpen) => {
    if (!isOpen) {
      reset();
      return;
    }
    formError.value = '';
    form.dataset_id = props.datasets[0]?.id ?? '';
    form.dataset_type = 'working_days';
    form.target_column = '';
    form.test_size = 0.2;
    form.random_seed = 42;
    form.split_date = '';
    form.shuffle = true;
    form.stratify = false;
  },
  { immediate: true },
);

async function submit() {
  const targetColumn = form.target_column.trim();
  if (!targetColumn) {
    formError.value = 'The "Target column name" field is required.';
    phase.value = 'error';
    return;
  }

  await triggerRun(
    {
      dataset_id: Number(form.dataset_id),
      dataset_type: form.dataset_type,
      dag_args: {
        target_column: targetColumn,
        test_size: form.test_size,
        random_seed: form.random_seed,
        split_date: form.split_date || null,
        shuffle: form.shuffle,
        stratify: form.stratify,
      },
    },
    emit,
  );
}

function onClose() {
  emit('close');
}
</script>
