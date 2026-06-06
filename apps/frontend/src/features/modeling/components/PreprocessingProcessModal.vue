<template>
  <BaseModal
    :show="show"
    :title="process.title"
    width="760px"
    @close="onClose"
  >
    <div class="space-y-4">
      <p class="text-sm text-fg-muted">
        Process configuration
        <span class="font-mono text-fg">preprocessing_feature_engineering</span>.
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
          required
        />

        <ModelingFormRadioGroup
          v-model="form.dataset_type"
          label="Dataset type"
          :options="datasetTypeOptions"
        />

        <div
          v-for="section in sections"
          :key="section.title"
          class="space-y-4 rounded-lg border border-border-default bg-surface-muted p-4"
        >
          <h3 class="text-sm font-semibold text-fg">{{ section.title }}</h3>
          <template v-for="field in section.fields" :key="field.key">
            <ModelingFormCheckbox
              v-if="field.type === 'checkbox'"
              v-model="form[field.key]"
              :label="field.label"
            />
            <ModelingFormInput
              v-else
              v-model="form[field.key]"
              :label="field.label"
              :type="field.inputType ?? 'text'"
              :placeholder="field.placeholder"
              :min="field.min"
              :max="field.max"
              :step="field.step"
              :value-type="field.valueType"
              :required="field.required"
            />
          </template>
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

const num = (extra = {}) => ({ valueType: 'number', inputType: 'number', ...extra });
const frac = (extra = {}) => num({ step: 0.01, min: 0, max: 1, required: true, ...extra });

const sections = [
  {
    title: 'RAW paths',
    fields: [
      { key: 'kpi_definitions_raw_path', label: 'KPI definitions raw path', required: true, placeholder: '.../raw_data/kpis_definitions.parquet' },
      { key: 'simple_reports_raw_path', label: 'Simple reports raw path', required: true, placeholder: '.../raw_data/simple_reports.parquet' },
    ],
  },
  {
    title: 'Output paths',
    fields: [
      { key: 'output_path_prefix', label: 'Output path prefix', required: true, placeholder: '.../preprocessed_dataset' },
    ],
  },
  {
    title: 'KPI coverage',
    fields: [
      frac({ key: 'kpi_min_global_density', label: 'KPI min global density' }),
      frac({ key: 'kpi_global_min_frac_cells_passing', label: 'KPI global min frac cells passing' }),
      frac({ key: 'kpi_window_coverage_frac', label: 'KPI window coverage frac' }),
    ],
  },
  {
    title: 'Max gap filtering',
    fields: [frac({ key: 'min_imputable_gap_frac', label: 'Min imputable gap frac' })],
  },
  {
    title: 'Stale KPIs filtering',
    fields: [
      num({ key: 'kpi_min_std_val', label: 'KPI min std val', min: 0, step: 0.01, required: true }),
      frac({ key: 'max_zero_frac', label: 'Max zero frac' }),
    ],
  },
  {
    title: 'Training data windows',
    fields: [
      num({ key: 'window_width_hours', label: 'Window width (hours)', min: 1, step: 1, default: 168 }),
      num({ key: 'stride_hours', label: 'Stride (hours)', min: 1, step: 1, default: 24 }),
      num({ key: 'max_gap_hours', label: 'Max gap (hours)', min: 1, step: 1, default: 6 }),
      num({ key: 'min_joint_windows_abs', label: 'Min joint windows (optional)', min: 1, step: 1, placeholder: 'empty = null' }),
    ],
  },
  {
    title: 'Imputation',
    fields: [{ key: 'impute', label: 'Enable imputation', type: 'checkbox', default: true }],
  },
];

const preprocessingDefaults = Object.fromEntries(
  sections.flatMap((section) => section.fields).map((field) => [field.key, field.default ?? '']),
);

const form = reactive({
  dataset_id: '',
  dataset_type: 'working_days',
  ...preprocessingDefaults,
});

const allFields = sections.flatMap((section) => section.fields);

const datasetOptions = computed(() =>
  props.datasets.map((dataset) => ({
    value: dataset.id,
    label: `#${dataset.id} · ${dataset.file_name} · ${dataset.type} · ${dataset.status}`,
  })),
);

function getFormError() {
  for (const field of allFields) {
    const value = form[field.key];
    if (field.required) {
      const isEmpty = field.valueType === 'number' ? value === '' : !String(value).trim();
      if (isEmpty) return `"${field.label}" is required.`;
    }
    if (field.valueType !== 'number' || value === '' || value === '.' || (typeof value === 'string' && value.endsWith('.'))) {
      continue;
    }
    const numValue = Number(value);
    if (Number.isNaN(numValue)) continue;
    if (field.min != null && numValue < field.min) {
      return field.max != null
        ? `"${field.label}" must be between ${field.min} and ${field.max}.`
        : `"${field.label}" must be at least ${field.min}.`;
    }
    if (field.max != null && numValue > field.max) {
      return `"${field.label}" must be at most ${field.max}.`;
    }
  }
  return '';
}

const isSubmitDisabled = computed(() => !form.dataset_id || Boolean(getFormError()));

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
    Object.assign(form, preprocessingDefaults);
  },
  { immediate: true },
);

function buildDagArgs() {
  return Object.fromEntries(
    allFields.map((field) => {
      const value = form[field.key];
      if (field.key === 'min_joint_windows_abs') return [field.key, value === '' ? null : value];
      if (field.type === 'checkbox' || field.valueType === 'number') return [field.key, value];
      return [field.key, String(value).trim()];
    }),
  );
}

async function submit() {
  const error = getFormError();
  if (error) {
    formError.value = error;
    phase.value = 'error';
    return;
  }

  await triggerRun(
    {
      dataset_id: Number(form.dataset_id),
      dataset_type: form.dataset_type,
      dag_args: buildDagArgs(),
    },
    emit,
  );
}

function onClose() {
  emit('close');
}
</script>
