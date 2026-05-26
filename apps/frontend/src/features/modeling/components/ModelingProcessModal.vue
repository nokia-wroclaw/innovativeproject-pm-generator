<template>
  <BaseModal
    :show="show"
    :title="process.title"
    width="680px"
    @close="close"
  >
    <div class="space-y-4">
      <p class="text-sm text-fg-muted">
        Formularz jest budowany na podstawie schematu zwróconego dla procesu
        <span class="font-mono text-fg">{{ process.processType }}</span>.
      </p>

      <div v-if="isSchemaLoading" class="flex items-center gap-2 text-sm text-fg-muted">
        <Loader2 :size="16" class="animate-spin" />
        Ładowanie formularza...
      </div>

      <div
        v-else-if="schemaError"
        class="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
      >
        <p>{{ schemaErrorMessage }}</p>
        <Button variant="secondary" size="sm" class="mt-3" @click="schemaQuery.refetch()">
          Spróbuj ponownie
        </Button>
      </div>

      <form v-else class="space-y-4" @submit.prevent="submit">
        <template v-for="field in fields" :key="field.name">
          <label v-if="field.type === 'dataset_select'" class="block space-y-2">
            <span class="text-sm font-medium text-fg">{{ field.label }}</span>
            <select
              v-model.number="form[field.name]"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            >
              <option value="" disabled>Wybierz dataset</option>
              <option v-for="dataset in datasets" :key="dataset.id" :value="dataset.id">
                #{{ dataset.id }} · {{ dataset.file_name }} · {{ dataset.status }}
              </option>
            </select>
            <span v-if="field.help" class="text-xs text-fg-subtle">{{ field.help }}</span>
          </label>

          <fieldset v-else-if="field.type === 'radio'" class="space-y-2">
            <legend class="text-sm font-medium text-fg">{{ field.label }}</legend>
            <div class="grid gap-2 sm:grid-cols-2">
              <label
                v-for="option in field.options"
                :key="option.value"
                class="flex cursor-pointer items-center gap-2 rounded-md border border-border-default p-3 text-sm"
              >
                <input v-model="form[field.name]" type="radio" :value="option.value" />
                <span>{{ option.label }}</span>
              </label>
            </div>
          </fieldset>

          <label v-else-if="field.type === 'select'" class="block space-y-2">
            <span class="text-sm font-medium text-fg">{{ field.label }}</span>
            <select
              v-model.number="form[field.name]"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            >
              <option v-for="option in field.options" :key="option.value" :value="option.value">
                {{ option.label }}
              </option>
            </select>
          </label>

          <label v-else-if="field.type === 'text'" class="block space-y-2">
            <span class="text-sm font-medium text-fg">{{ field.label }}</span>
            <input
              v-model="form[field.name]"
              type="text"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            />
            <span v-if="field.help" class="text-xs text-fg-subtle">{{ field.help }}</span>
          </label>

          <label v-else-if="field.type === 'integer' || field.type === 'float'" class="block space-y-2">
            <span class="text-sm font-medium text-fg">{{ field.label }}</span>
            <input
              v-model.number="form[field.name]"
              type="number"
              :min="field.min ?? undefined"
              :max="field.max ?? undefined"
              :step="field.step ?? (field.type === 'integer' ? 1 : 'any')"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
            />
          </label>

          <label v-else-if="field.type === 'json'" class="block space-y-2">
            <span class="text-sm font-medium text-fg">{{ field.label }}</span>
            <textarea
              v-model="jsonFields[field.name]"
              rows="6"
              class="w-full rounded-md border border-border-default bg-surface px-3 py-2 font-mono text-xs text-fg"
            />
            <span v-if="field.help" class="text-xs text-fg-subtle">{{ field.help }}</span>
          </label>
        </template>

        <div
          v-if="formError"
          class="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
        >
          {{ formError }}
        </div>
      </form>
    </div>

    <template #footer>
      <Button variant="secondary" @click="close">Anuluj</Button>
      <Button :disabled="isSubmitting || isSchemaLoading || Boolean(schemaError) || !schema" @click="submit">
        <Loader2 v-if="isSubmitting" :size="14" class="animate-spin" />
        Uruchom DAG
      </Button>
    </template>
  </BaseModal>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';
import { Loader2 } from 'lucide-vue-next';

import BaseModal from '@/components/BaseModal.vue';
import { Button } from '@/components/ui';
import {
  useModelingFormSchema,
  useTriggerModelingRun,
} from '../composables/queries.js';

const props = defineProps({
  show: { type: Boolean, required: true },
  process: { type: Object, required: true },
  datasets: { type: Array, default: () => [] },
});

const emit = defineEmits(['close', 'started']);

const processType = computed(() => props.process.processType);
const schemaQuery = useModelingFormSchema(processType);
const triggerMutation = useTriggerModelingRun();

const form = reactive({});
const jsonFields = reactive({});
const formError = ref('');

const schema = computed(() => schemaQuery.data.value ?? null);
const fields = computed(() => schema.value?.fields ?? []);
const isSchemaLoading = computed(() => schemaQuery.isLoading.value);
const schemaError = computed(() => schemaQuery.error.value);
const schemaErrorMessage = computed(() => {
  if (schemaError.value?.code === 'REQUEST_TIMEOUT') {
    return 'Nie udało się pobrać formularza w wyznaczonym czasie. Sprawdź backend albo spróbuj ponownie.';
  }
  return schemaError.value?.message ?? 'Nie udało się pobrać formularza.';
});
const isSubmitting = computed(() => triggerMutation.isPending.value);

watch(
  [schema, () => props.show],
  ([currentSchema, isOpen]) => {
    if (!currentSchema || !isOpen) return;
    resetForm(currentSchema.fields);
  },
  { immediate: true },
);

function resetForm(schemaFields) {
  formError.value = '';
  for (const field of schemaFields) {
    if (field.type === 'json') {
      jsonFields[field.name] = JSON.stringify(field.default ?? {}, null, 2);
    } else {
      form[field.name] = field.default ?? '';
    }
  }
  if (!form.dataset_id && props.datasets.length) {
    form.dataset_id = props.datasets[0].id;
  }
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
    formError.value = error?.message ?? 'Nie udało się uruchomić DAG-a.';
  }
}

function buildPayload() {
  const payload = {};
  for (const field of fields.value) {
    if (field.type === 'json') {
      const raw = (jsonFields[field.name] ?? '').trim();
      payload[field.name] = raw ? JSON.parse(raw) : {};
      if (payload[field.name] === null || Array.isArray(payload[field.name])) {
        throw new Error(`${field.label} musi być obiektem JSON.`);
      }
    } else {
      payload[field.name] = form[field.name];
    }
  }
  return payload;
}

function close() {
  emit('close');
}
</script>
