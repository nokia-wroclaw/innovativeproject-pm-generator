<template>
  <BaseModal
    :show="show"
    :title="process.title"
    width="680px"
    @close="close"
  >
    <div class="space-y-4">
      <p class="text-sm text-fg-muted">
        Process configuration
        <span class="font-mono text-fg">preprocessing_feature_engineering</span>.
      </p>

      <form class="space-y-4" @submit.prevent="submit">
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
          <h3 class="text-sm font-semibold text-fg">Preprocessing + Feature Engineering</h3>

          <ModelingFormSelect
            v-model="form.missing_value_strategy"
            label="Missing value strategy"
            :options="missingValueStrategyOptions"
          />

          <ModelingFormSelect
            v-model="form.scaling_method"
            label="Feature scaling"
            :options="scalingMethodOptions"
          />

          <ModelingFormSelect
            v-model="form.outlier_method"
            label="Outlier handling"
            :options="outlierMethodOptions"
          />

          <ModelingFormCheckbox
            v-model="form.enable_calendar_features"
            label="Add calendar features (day of week, month, holidays)"
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
      <Button variant="secondary" @click="close">Cancel</Button>
      <Button :disabled="isSubmitting || !form.dataset_id" @click="submit">
        <Loader2 v-if="isSubmitting" :size="14" class="animate-spin" />
        Run DAG
      </Button>
    </template>
  </BaseModal>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';
import { Loader2 } from 'lucide-vue-next';

import BaseModal from '@/components/BaseModal.vue';
import { Button } from '@/components/ui';
import { useTriggerModelingRun } from '../composables/queries.js';
import ModelingFormCheckbox from './form-fields/ModelingFormCheckbox.vue';
import ModelingFormRadioGroup from './form-fields/ModelingFormRadioGroup.vue';
import ModelingFormSelect from './form-fields/ModelingFormSelect.vue';

const props = defineProps({
  show: { type: Boolean, required: true },
  process: { type: Object, required: true },
  datasets: { type: Array, default: () => [] },
});

const emit = defineEmits(['close', 'started']);
const triggerMutation = useTriggerModelingRun();
const isSubmitting = computed(() => triggerMutation.isPending.value);
const formError = ref('');

const datasetTypeOptions = [
  { value: 'working_days', label: 'Working days' },
  { value: 'weekends', label: 'Weekends' },
];
const missingValueStrategyOptions = [
  { value: 'drop', label: 'Drop rows with missing values' },
  { value: 'median', label: 'Fill with median' },
  { value: 'mean', label: 'Fill with mean' },
];
const scalingMethodOptions = [
  { value: 'standard', label: 'StandardScaler' },
  { value: 'minmax', label: 'MinMaxScaler' },
  { value: 'none', label: 'No scaling' },
];
const outlierMethodOptions = [
  { value: 'none', label: 'None' },
  { value: 'iqr', label: 'IQR clipping' },
  { value: 'zscore', label: 'Z-score filtering' },
];

const form = reactive({
  dataset_id: '',
  dataset_type: 'working_days',
  missing_value_strategy: 'median',
  scaling_method: 'standard',
  outlier_method: 'iqr',
  enable_calendar_features: true,
});

const datasetOptions = computed(() =>
  props.datasets.map((dataset) => ({
    value: dataset.id,
    label: `#${dataset.id} · ${dataset.file_name} · ${dataset.status}`,
  })),
);

watch(
  () => props.show,
  (isOpen) => {
    if (!isOpen) return;
    formError.value = '';
    form.dataset_id = props.datasets[0]?.id ?? '';
    form.dataset_type = 'working_days';
    form.missing_value_strategy = 'median';
    form.scaling_method = 'standard';
    form.outlier_method = 'iqr';
    form.enable_calendar_features = true;
  },
  { immediate: true },
);

async function submit() {
  formError.value = '';
  try {
    const response = await triggerMutation.mutateAsync({
      processType: props.process.processType,
      body: {
        dataset_id: Number(form.dataset_id),
        dataset_type: form.dataset_type,
        dag_args: {
          missing_value_strategy: form.missing_value_strategy,
          scaling_method: form.scaling_method,
          outlier_method: form.outlier_method,
          enable_calendar_features: form.enable_calendar_features,
        },
      },
    });
    emit('started', response);
    close();
  } catch (error) {
    formError.value = error?.message ?? 'Failed to trigger DAG.';
  }
}

function close() {
  emit('close');
}
</script>
