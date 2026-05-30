<template>
  <BaseModal
    :show="show"
    :title="process.title"
    width="680px"
    @close="close"
  >
    <div class="space-y-4">
      <p class="text-sm text-fg-muted">
        Dedicated configuration form for process
        <span class="font-mono text-fg">{{ process.processType }}</span>.
      </p>

      <form class="space-y-4" @submit.prevent="submit">
        <label class="block space-y-2">
          <span class="text-sm font-medium text-fg">Input dataset</span>
          <select
            v-model.number="form.dataset_id"
            class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
          >
            <option value="" disabled>Select dataset</option>
            <option v-for="dataset in datasets" :key="dataset.id" :value="dataset.id">
              #{{ dataset.id }} · {{ dataset.file_name }} · {{ dataset.type }} · {{ dataset.status }}
            </option>
          </select>
          <span class="text-xs text-fg-subtle">Only datasets with COMPLETED status are shown.</span>
        </label>

        <fieldset class="space-y-2">
          <legend class="text-sm font-medium text-fg">Dataset type</legend>
          <div class="grid gap-2 sm:grid-cols-2">
            <label class="flex cursor-pointer items-center gap-2 rounded-md border border-border-default p-3 text-sm">
              <input v-model="form.dataset_type" type="radio" value="working_days" />
              <span>Working days</span>
            </label>
            <label class="flex cursor-pointer items-center gap-2 rounded-md border border-border-default p-3 text-sm">
              <input v-model="form.dataset_type" type="radio" value="weekends" />
              <span>Weekends</span>
            </label>
          </div>
        </fieldset>

        <div
          v-if="isPreprocessingProcess"
          class="space-y-4 rounded-lg border border-border-default bg-surface-muted p-4"
        >
          <h3 class="text-sm font-semibold text-fg">Preprocessing + Feature Engineering</h3>

          <label class="block space-y-2">
            <span class="text-sm font-medium text-fg">Missing value strategy</span>
            <select
              v-model="form.missing_value_strategy"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            >
              <option value="drop">Drop rows with missing values</option>
              <option value="median">Fill with median</option>
              <option value="mean">Fill with mean</option>
            </select>
          </label>

          <label class="block space-y-2">
            <span class="text-sm font-medium text-fg">Feature scaling</span>
            <select
              v-model="form.scaling_method"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            >
              <option value="standard">StandardScaler</option>
              <option value="minmax">MinMaxScaler</option>
              <option value="none">No scaling</option>
            </select>
          </label>

          <label class="block space-y-2">
            <span class="text-sm font-medium text-fg">Outlier handling</span>
            <select
              v-model="form.outlier_method"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            >
              <option value="none">None</option>
              <option value="iqr">IQR clipping</option>
              <option value="zscore">Z-score filtering</option>
            </select>
          </label>

          <label class="flex items-center gap-2 text-sm text-fg">
            <input v-model="form.enable_calendar_features" type="checkbox" />
            Add calendar features (day of week, month, holidays)
          </label>
        </div>

        <div
          v-else
          class="space-y-4 rounded-lg border border-border-default bg-surface-muted p-4"
        >
          <h3 class="text-sm font-semibold text-fg">Training dataset creation</h3>

          <label class="block space-y-2">
            <span class="text-sm font-medium text-fg">Target column name</span>
            <input
              v-model.trim="form.target_column"
              type="text"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
              placeholder="np. sales"
            />
          </label>

          <label class="block space-y-2">
            <span class="text-sm font-medium text-fg">Test set size</span>
            <input
              v-model.number="form.test_size"
              type="number"
              min="0.05"
              max="0.5"
              step="0.01"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            />
          </label>

          <label class="block space-y-2">
            <span class="text-sm font-medium text-fg">Random seed</span>
            <input
              v-model.number="form.random_seed"
              type="number"
              min="1"
              step="1"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            />
          </label>

          <label class="flex items-center gap-2 text-sm text-fg">
            <input v-model="form.shuffle" type="checkbox" />
            Shuffle samples before split
          </label>

          <label class="flex items-center gap-2 text-sm text-fg">
            <input v-model="form.stratify" type="checkbox" />
            Stratified split (when target is categorical)
          </label>
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
      <Button :disabled="isSubmitting || isSubmitDisabled" @click="submit">
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

const PREPROCESSING_PROCESS = 'preprocessing_feature_engineering';
const TRAINING_DATASET_PROCESS = 'training_dataset';

const props = defineProps({
  show: { type: Boolean, required: true },
  process: { type: Object, required: true },
  datasets: { type: Array, default: () => [] },
});

const emit = defineEmits(['close', 'started']);

const processType = computed(() => props.process.processType);
const triggerMutation = useTriggerModelingRun();

const form = reactive({
  dataset_id: '',
  dataset_type: 'working_days',
  missing_value_strategy: 'median',
  scaling_method: 'standard',
  outlier_method: 'iqr',
  enable_calendar_features: true,
  target_column: '',
  test_size: 0.2,
  random_seed: 42,
  shuffle: true,
  stratify: false,
});
const formError = ref('');

const isPreprocessingProcess = computed(() => processType.value === PREPROCESSING_PROCESS);
const isSubmitting = computed(() => triggerMutation.isPending.value);
const isSubmitDisabled = computed(() => {
  if (!form.dataset_id) return true;
  if (processType.value === TRAINING_DATASET_PROCESS && !form.target_column?.trim()) return true;
  return false;
});

watch(
  [processType, () => props.show],
  ([nextType, isOpen]) => {
    if (!isOpen) return;
    resetForm(nextType);
  },
  { immediate: true },
);

function resetForm(nextType) {
  formError.value = '';
  form.dataset_id = props.datasets[0]?.id ?? '';
  form.dataset_type = 'working_days';
  form.missing_value_strategy = 'median';
  form.scaling_method = 'standard';
  form.outlier_method = 'iqr';
  form.enable_calendar_features = true;
  form.target_column = '';
  form.test_size = 0.2;
  form.random_seed = 42;
  form.shuffle = true;
  form.stratify = false;

  if (nextType === PREPROCESSING_PROCESS) return;
}

async function submit() {
  formError.value = '';
  try {
    const body = buildPayload();
    const response = await triggerMutation.mutateAsync({
      processType: props.process.processType,
      body,
    });
    emit('started', response);
    close();
  } catch (error) {
    formError.value = error?.message ?? 'Failed to trigger DAG.';
  }
}

function buildPayload() {
  const dagArgs = {};

  if (processType.value === PREPROCESSING_PROCESS) {
    dagArgs.missing_value_strategy = form.missing_value_strategy;
    dagArgs.scaling_method = form.scaling_method;
    dagArgs.outlier_method = form.outlier_method;
    dagArgs.enable_calendar_features = form.enable_calendar_features;
  } else if (processType.value === TRAINING_DATASET_PROCESS) {
    if (!form.target_column?.trim()) {
      throw new Error('The "Target column name" field is required.');
    }
    dagArgs.target_column = form.target_column.trim();
    dagArgs.test_size = form.test_size;
    dagArgs.random_seed = form.random_seed;
    dagArgs.shuffle = form.shuffle;
    dagArgs.stratify = form.stratify;
  }

  return {
    dataset_id: Number(form.dataset_id),
    dataset_type: form.dataset_type,
    dag_args: dagArgs,
  };
}

function close() {
  emit('close');
}
</script>
