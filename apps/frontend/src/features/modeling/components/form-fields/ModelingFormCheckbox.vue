<template>
  <label class="block space-y-1">
    <span class="flex items-center gap-2 text-sm text-fg">
      <input
        :checked="modelValue"
        type="checkbox"
        @change="emit('update:modelValue', $event.target.checked)"
      />
      <span class="inline-flex items-center gap-1">
        {{ label }}
        <ModelingFormHint v-if="hint" :text="hint" />
      </span>
    </span>
  </label>
</template>

<script setup>
import ModelingFormHint from './ModelingFormHint.vue';
defineProps({
  modelValue: { type: Boolean, default: false },
  label: { type: String, required: true },
  hint: { type: String, default: '' },
});

const emit = defineEmits(['update:modelValue']);
</script>
