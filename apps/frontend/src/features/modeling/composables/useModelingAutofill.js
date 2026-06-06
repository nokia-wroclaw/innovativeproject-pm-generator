import { computed, ref } from 'vue';
import { useMutation } from '@tanstack/vue-query';

import { autofillModelingForm } from '../services/modelingApi.js';

export function useModelingAutofill(processType) {
  const instruction = ref('');
  const localError = ref('');

  const mutation = useMutation({
    mutationFn: ({ instruction: text, currentValues }) =>
      autofillModelingForm(processType, {
        instruction: text,
        current_values: currentValues ?? {},
      }),
  });

  const isAutofilling = computed(() => mutation.isPending.value);
  const errorMessage = computed(
    () => localError.value || mutation.error.value?.message || '',
  );

  async function runAutofill(currentValues) {
    const text = instruction.value.trim();
    if (!text) {
      localError.value = 'Describe what you want to configure before auto-filling.';
      return null;
    }
    localError.value = '';
    try {
      const result = await mutation.mutateAsync({ instruction: text, currentValues });
      return result?.values ?? null;
    } catch {
      return null;
    }
  }

  function clearError() {
    localError.value = '';
    mutation.reset();
  }

  return {
    instruction,
    isAutofilling,
    errorMessage,
    runAutofill,
    clearError,
  };
}
