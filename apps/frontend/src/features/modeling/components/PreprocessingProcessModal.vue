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
        <span class="font-mono text-fg">preprocessing_feature_engineering</span>
        · DAG
        <span class="font-mono text-fg">preprocessing_pipeline</span>.
        Output path is assigned automatically by the backend.
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
          :options="rawDatasetOptions"
          value-type="number"
          required
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
              :hint="field.hint"
            />
            <ModelingFormSelect
              v-else-if="field.type === 'select'"
              v-model="form[field.key]"
              :label="field.label"
              :hint="field.hint"
              :placeholder="field.placeholder"
              :options="field.key === 'kpi_definitions_raw_path' ? kpiDefinitionsOptions : simpleReportsOptions"
              :required="field.required"
            />
            <ModelingFormInput
              v-else
              v-model="form[field.key]"
              :label="field.label"
              :hint="field.hint"
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

const num = (extra = {}) => ({ valueType: 'number', inputType: 'number', ...extra });
const frac = (extra = {}) => num({ step: 0.01, min: 0, max: 1, required: true, ...extra });

const sections = [
  {
    title: 'RAW paths',
    fields: [
      { key: 'kpi_definitions_raw_path', label: 'KPI definitions raw path', type: 'select', required: true, placeholder: 'Select KPI definitions parquet' },
      { key: 'simple_reports_raw_path', label: 'Simple reports raw path', type: 'select', required: true, placeholder: 'Select simple reports parquet' },
    ],
  },
  {
    title: 'KPI coverage',
    fields: [
      frac({
        key: 'kpi_min_global_density',
        label: 'KPI min global density',
        default: 0.5,
        hint: 'Min. share of non-null hours in a KPI series active range per cell.',
      }),
      frac({
        key: 'kpi_global_min_frac_cells_passing',
        label: 'KPI global min frac cells passing',
        default: 0.8,
        hint: 'Min. share of a KPI cells that must meet the density threshold.',
      }),
    ],
  },
  {
    title: 'Max gap filtering',
    fields: [
      frac({
        key: 'min_imputable_gap_frac',
        label: 'Min imputable gap frac',
        default: 0.8,
        hint: 'Min. share of null runs short enough to impute (≤ max gap hours).',
      }),
    ],
  },
  {
    title: 'Stale KPIs filtering',
    fields: [
      num({
        key: 'kpi_min_std_val',
        label: 'KPI min std val',
        min: 0,
        step: 0.01,
        default: 0.01,
        required: true,
        hint: 'Reject KPIs with near-zero variance in good-window values.',
      }),
      frac({
        key: 'max_zero_frac',
        label: 'Max zero frac',
        default: 0.95,
        hint: 'Reject KPIs where at least this share of values is zero.',
      }),
    ],
  },
  {
    title: 'Training data windows',
    fields: [
      num({
        key: 'window_width_hours',
        label: 'Window width (hours)',
        min: 1,
        step: 1,
        default: 168,
        required: true,
        hint: 'Window length. Valid windows need all W contiguous hours (0..W-1).',
      }),
      num({
        key: 'stride_hours',
        label: 'Stride (hours)',
        min: 1,
        step: 1,
        default: 24,
        required: true,
        hint: 'Hours between stride-aligned window anchors.',
      }),
      num({
        key: 'max_gap_hours',
        label: 'Max gap (hours)',
        min: 1,
        step: 1,
        default: 24,
        required: true,
        hint: 'Max consecutive null hours per window and safe imputation limit.',
      }),
      num({
        key: 'min_joint_windows_abs',
        label: 'Min joint windows (optional)',
        min: 1,
        step: 1,
        placeholder: 'empty = null',
        hint: 'Min. joint (distname, anchor) pairs for the selected KPI set. Empty = elbow method.',
      }),
    ],
  },
  {
    title: 'Imputation',
    fields: [{
      key: 'impute',
      label: 'Enable imputation',
      type: 'checkbox',
      default: true,
      hint: 'Forward-fill / interpolate gaps up to max gap hours before window validation.',
    }],
  },
];

const preprocessingDefaults = Object.fromEntries(
  sections.flatMap((section) => section.fields).map((field) => [field.key, field.default ?? '']),
);

const form = reactive({
  dataset_id: '',
  ...preprocessingDefaults,
});

const allFields = sections.flatMap((section) => section.fields);

const rawDatasetOptions = computed(() =>
  props.datasets
    .filter((dataset) => dataset.type === 'RAW')
    .map((dataset) => ({
      value: dataset.id,
      label: `#${dataset.id} · ${dataset.file_name} · ${dataset.status}`,
    })),
);

const kpiDefinitionsOptions = computed(() =>
  props.datasets
    .filter((dataset) => dataset.type === 'KPI_DEFINITIONS')
    .map((dataset) => ({
      value: dataset.s3_key,
      label: `#${dataset.id} · ${dataset.file_name} · ${dataset.s3_key}`,
    })),
);

const simpleReportsOptions = computed(() =>
  props.datasets
    .filter((dataset) => dataset.type === 'SIMPLE_REPORTS')
    .map((dataset) => ({
      value: dataset.s3_key,
      label: `#${dataset.id} · ${dataset.file_name} · ${dataset.s3_key}`,
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
    form.dataset_id = props.datasets.find((dataset) => dataset.type === 'RAW')?.id ?? '';
    Object.assign(form, preprocessingDefaults);
  },
  { immediate: true },
);

function buildDagArgs() {
  return Object.fromEntries(
    allFields
      .map((field) => {
        const value = form[field.key];
        if (field.key === 'min_joint_windows_abs') {
          return [field.key, value === '' ? null : value];
        }
        if (field.type === 'checkbox' || field.valueType === 'number') {
          return [field.key, value];
        }
        return [field.key, String(value).trim()];
      })
      .filter(([, value]) => value !== '' && value !== null),
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
      dag_args: buildDagArgs(),
    },
    emit,
  );
}

function onClose() {
  emit('close');
}
</script>
