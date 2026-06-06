<template>
  <label class="block space-y-2">
    <span class="inline-flex items-center gap-1 text-sm font-medium text-fg">
      {{ label }}
      <ModelingFormHint v-if="hint" :text="hint" />
    </span>
    <select
      :value="selectValue"
      class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
      @change="handleChange"
    >
      <option v-if="placeholder" value="" disabled>{{ placeholder }}</option>
      <option
        v-for="option in options"
        :key="String(option.value)"
        :value="optionValue(option)"
      >
        {{ option.label }}
      </option>
    </select>
    <span v-if="hint" class="text-xs text-fg-subtle">{{ hint }}</span>
  </label>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  modelValue: { type: [String, Number], default: '' },
  label: { type: String, required: true },
  hint: { type: String, default: '' },
  placeholder: { type: String, default: '' },
  options: { type: Array, default: () => [] },
  valueType: { type: String, default: 'string' },
});

const emit = defineEmits(['update:modelValue']);

// HTML select compares string values; normalize so programmatic updates show correctly.
const selectValue = computed(() => {
  if (props.modelValue === '' || props.modelValue == null) return '';
  if (props.valueType === 'number') return String(props.modelValue);
  return props.modelValue;
});

function optionValue(option) {
  if (props.valueType === 'number') return String(option.value);
  return option.value;
}

function handleChange(event) {
  const rawValue = event.target.value;
  if (props.valueType === 'number') {
    emit('update:modelValue', rawValue === '' ? '' : Number(rawValue));
    return;
  }
  emit('update:modelValue', rawValue);
}
</script>
