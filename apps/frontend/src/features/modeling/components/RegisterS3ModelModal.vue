<template>
  <BaseModal
    :show="show"
    :title="isEditMode ? 'Edit S3 Model' : 'Register S3 Model'"
    width="580px"
    @close="onClose"
  >
    <div class="space-y-4">
      <p class="text-sm text-fg-muted">
        {{ isEditMode ? 'Edit the metadata of the registered Keras model.' : 'Register a Keras model file that is already stored in your S3/MinIO bucket.' }}
      </p>

      <form class="space-y-4" @submit.prevent="submit">
        <ModelingFormInput
          v-model="form.name"
          label="Model name"
          placeholder="e.g. Random Forest V1"
          required
        />

        <ModelingFormInput
          v-model="form.s3_key"
          label="S3 Key"
          placeholder="e.g. models/my_model.weights.h5"
          required
        />

        <ModelingFormInput
          v-model="form.encoder_s3_key"
          label="Encoder S3 Key"
          placeholder="e.g. models/my_encoder.pkl"
          required
        />

        <ModelingFormInput
          v-model="form.config_s3_key"
          label="Config S3 Key"
          placeholder="e.g. models/my_config.json"
          required
        />

        <ModelingFormSelect
          v-model="form.dataset_id"
          label="Dataset used for training"
          placeholder="Select dataset"
          :options="datasetOptions"
          value-type="number"
        />

        <div
          v-if="errorMsg"
          class="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
        >
          {{ errorMsg }}
        </div>
      </form>
    </div>

    <template #footer>
      <Button variant="secondary" :disabled="isPending" @click="onClose">
        Cancel
      </Button>
      <Button :disabled="isPending || isSubmitDisabled" @click="submit">
        <Loader2 v-if="isPending" :size="14" class="animate-spin" />
        {{ isEditMode ? (isPending ? 'Saving…' : 'Save Changes') : (isPending ? 'Registering…' : 'Register') }}
      </Button>
    </template>
  </BaseModal>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';
import { Loader2 } from 'lucide-vue-next';

import BaseModal from '@/components/BaseModal.vue';
import { Button } from '@/components/ui';
import ModelingFormInput from './form-fields/ModelingFormInput.vue';
import ModelingFormSelect from './form-fields/ModelingFormSelect.vue';
import { useCreateTrainedModel, useUpdateTrainedModel, useModelingDatasets } from '../composables/queries.js';

const props = defineProps({
  show: { type: Boolean, required: true },
  model: { type: Object, default: null },
});

const emit = defineEmits(['close', 'registered', 'updated']);

const datasetsQuery = useModelingDatasets();
const datasets = computed(() => datasetsQuery.data.value ?? []);

const datasetOptions = computed(() => {
  return datasets.value
    .filter((dataset) => dataset.type === 'PREPROCESSED')
    .map((dataset) => ({
      value: dataset.id,
      label: `#${dataset.id} · ${dataset.file_name} (${dataset.type})`,
    }));
});

const createMutation = useCreateTrainedModel();
const updateMutation = useUpdateTrainedModel();
const isPending = computed(() => createMutation.isPending.value || updateMutation.isPending.value);
const errorMsg = ref('');

const form = reactive({
  name: '',
  s3_key: '',
  encoder_s3_key: '',
  config_s3_key: '',
  dataset_id: '',
});

const isEditMode = computed(() => Boolean(props.model));
const isSubmitDisabled = computed(
  () =>
    !form.name.trim() ||
    !form.s3_key.trim() ||
    !form.encoder_s3_key.trim() ||
    !form.config_s3_key.trim() ||
    !form.dataset_id,
);

watch(
  [() => props.show, () => props.model],
  ([isOpen, modelVal]) => {
    if (isOpen) {
      if (modelVal) {
        form.name = modelVal.name || '';
        form.s3_key = modelVal.s3_key || '';
        form.encoder_s3_key = modelVal.encoder_s3_key || '';
        form.config_s3_key = modelVal.config_s3_key || '';
        form.dataset_id = modelVal.dataset_id || '';
      } else {
        form.name = '';
        form.s3_key = '';
        form.encoder_s3_key = '';
        form.config_s3_key = '';
        form.dataset_id = '';
      }
      errorMsg.value = '';
    }
  },
  { immediate: true }
);

async function submit() {
  const name = form.name.trim();
  const s3Key = form.s3_key.trim();
  if (!name || !s3Key || !form.dataset_id) return;

  errorMsg.value = '';
  if (!s3Key.toLowerCase().endsWith('.weights.h5')) {
    errorMsg.value = 'Model file must have .weights.h5 extension.';
    return;
  }

  const payload = {
    name,
    s3_key: s3Key,
    encoder_s3_key: form.encoder_s3_key.trim() || null,
    config_s3_key: form.config_s3_key.trim() || null,
    dataset_id: Number(form.dataset_id),
  };

  try {
    if (isEditMode.value) {
      await updateMutation.mutateAsync({ id: props.model.id, body: payload });
      emit('updated');
    } else {
      await createMutation.mutateAsync(payload);
      emit('registered');
    }
    emit('close');
  } catch (err) {
    errorMsg.value = err.message || (isEditMode.value ? 'Failed to update the model.' : 'Failed to register the model.');
  }
}

function onClose() {
  emit('close');
}
</script>
