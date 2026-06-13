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
        <span class="font-mono text-fg">generate</span>.
      </p>

      <ModelingRunStatusPanel
        v-if="phase !== 'form'"
        :submitting="phase === 'submitting'"
        :polling="isPolling"
        :run-id="startedRun?.run_id ?? null"
        :status-data="statusData"
        :status-error="statusQuery.error.value?.message ?? ''"
      />

      <div
        v-if="modelsError"
        class="flex items-start gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
      >
        <p>Failed to load trained models: {{ modelsError.message }}</p>
      </div>

      <form
        class="space-y-4"
        :class="phase !== 'form' && phase !== 'error' ? 'pointer-events-none opacity-60' : ''"
        @submit.prevent="submit"
      >
        <ModelingFormSelect
          v-model="form.model_id"
          label="Trained model"
          hint="Models from completed training runs."
          placeholder="Select model"
          :options="modelOptions"
        />

        <ModelingFormSelect
          v-if="isAdmin && form.model_id"
          v-model="form.comparison_dataset_id"
          label="Comparison dataset"
          hint="Dataset used to compare generated results."
          placeholder="Select dataset"
          :options="datasetOptions"
          value-type="number"
        />

        <ModelingFormInput
          v-if="isAdmin && form.model_id"
          v-model="form.encoder_s3_key"
          label="Encoder S3 Key"
          placeholder="e.g. models/my_encoder.pkl"
          required
        />

        <ModelingFormInput
          v-if="isAdmin && form.model_id"
          v-model="form.config_s3_key"
          label="Config S3 Key"
          placeholder="e.g. models/my_config.json"
          required
        />

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
import { useModelingModels, useModelingDatasets } from '../composables/queries.js';
import { useModelingProcessRun } from '../composables/useModelingProcessRun.js';
import ModelingFormInput from './form-fields/ModelingFormInput.vue';
import ModelingFormSelect from './form-fields/ModelingFormSelect.vue';
import ModelingRunStatusPanel from './ModelingRunStatusPanel.vue';
import { isAdmin } from '@/auth/keycloak';

const props = defineProps({
  show: { type: Boolean, required: true },
  process: { type: Object, required: true },
});

const emit = defineEmits(['close', 'started']);

const modelsQuery = useModelingModels();
const models = computed(() => modelsQuery.data.value ?? []);
const modelsError = computed(() => modelsQuery.error.value);
const isModelsLoading = computed(() => modelsQuery.isLoading.value);

const datasetsQuery = useModelingDatasets({
  enabled: computed(() => isAdmin.value),
});
const datasets = computed(() => datasetsQuery.data.value ?? []);
const datasetOptions = computed(() => {
  if (!isAdmin.value) return [];
  return datasets.value.map((dataset) => ({
    value: dataset.id,
    label: `#${dataset.id} · ${dataset.file_name} (${dataset.type})`,
  }));
});

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

const form = reactive({
  model_id: '',
  comparison_dataset_id: '',
  encoder_s3_key: '',
  config_s3_key: '',
});

const modelOptions = computed(() =>
  models.value.map((model) => ({
    value: model.id,
    label: model.name,
  })),
);

const isSubmitDisabled = computed(
  () =>
    !form.model_id ||
    !form.encoder_s3_key.trim() ||
    !form.config_s3_key.trim() ||
    isModelsLoading.value,
);

function updateFormFromModel(modelId, list) {
  if (modelId && list && list.length) {
    const selectedModel = list.find((m) => String(m.id) === String(modelId));
    if (selectedModel) {
      form.comparison_dataset_id = selectedModel.dataset_id || '';
      form.encoder_s3_key = selectedModel.encoder_s3_key || '';
      form.config_s3_key = selectedModel.config_s3_key || '';
      return;
    }
  }
  form.comparison_dataset_id = '';
  form.encoder_s3_key = '';
  form.config_s3_key = '';
}

watch(
  () => props.show,
  (isOpen) => {
    if (!isOpen) {
      reset();
      return;
    }
    formError.value = '';
    form.model_id = models.value[0]?.id ?? '';
    updateFormFromModel(form.model_id, models.value);
  },
  { immediate: true },
);

watch(
  () => models.value,
  (list) => {
    if (props.show && !form.model_id && list.length) {
      form.model_id = list[0].id;
    }
  }
);

watch(
  [() => form.model_id, () => models.value],
  ([nextModelId, list]) => {
    updateFormFromModel(nextModelId, list);
  },
  { immediate: true },
);

async function submit() {
  if (!form.model_id) {
    formError.value = 'Select a trained model.';
    phase.value = 'error';
    return;
  }
  const encoderKey = form.encoder_s3_key.trim();
  const configKey = form.config_s3_key.trim();
  if (!encoderKey) {
    formError.value = 'The Encoder S3 Key is required.';
    phase.value = 'error';
    return;
  }
  if (!configKey) {
    formError.value = 'The Config S3 Key is required.';
    phase.value = 'error';
    return;
  }

  await triggerRun(
    {
      model_id: String(form.model_id),
      prompt: '',
      comparison_dataset_id: form.comparison_dataset_id ? Number(form.comparison_dataset_id) : null,
      encoder_s3_key: encoderKey,
      config_s3_key: configKey,
      dag_args: {},
    },
    emit,
  );
}

function onClose() {
  emit('close');
}
</script>

