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

        <ModelingFormTextarea
          v-model="form.prompt"
          label="Generation prompt"
          placeholder="Describe the synthetic event log to generate…"
          hint="Instructions passed to the generation DAG."
          :rows="5"
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
import { useModelingModels } from '../composables/queries.js';
import { useModelingProcessRun } from '../composables/useModelingProcessRun.js';
import ModelingFormSelect from './form-fields/ModelingFormSelect.vue';
import ModelingFormTextarea from './form-fields/ModelingFormTextarea.vue';
import ModelingRunStatusPanel from './ModelingRunStatusPanel.vue';

const props = defineProps({
  show: { type: Boolean, required: true },
  process: { type: Object, required: true },
});

const emit = defineEmits(['close', 'started']);

const modelsQuery = useModelingModels();
const models = computed(() => modelsQuery.data.value ?? []);
const modelsError = computed(() => modelsQuery.error.value);
const isModelsLoading = computed(() => modelsQuery.isLoading.value);

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
  prompt: '',
});

const modelOptions = computed(() =>
  models.value.map((model) => ({
    value: model.id,
    label: model.name,
  })),
);

const isSubmitDisabled = computed(
  () => !form.model_id || !form.prompt.trim() || isModelsLoading.value,
);

watch(
  () => props.show,
  (isOpen) => {
    if (!isOpen) {
      reset();
      return;
    }
    formError.value = '';
    form.model_id = models.value[0]?.id ?? '';
    form.prompt = '';
  },
  { immediate: true },
);

watch(models, (list) => {
  if (props.show && !form.model_id && list.length) {
    form.model_id = list[0].id;
  }
});

async function submit() {
  const prompt = form.prompt.trim();
  if (!form.model_id) {
    formError.value = 'Select a trained model.';
    phase.value = 'error';
    return;
  }
  if (!prompt) {
    formError.value = 'The generation prompt is required.';
    phase.value = 'error';
    return;
  }

  await triggerRun(
    {
      model_id: form.model_id,
      prompt,
      dag_args: {},
    },
    emit,
  );
}

function onClose() {
  emit('close');
}
</script>
