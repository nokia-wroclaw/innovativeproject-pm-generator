<template>
  <label class="block space-y-2">
    <span class="text-sm font-medium text-fg">{{ label }}</span>
    <select
      :value="modelValue"
      class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
      @change="handleChange"
    >
      <option v-if="placeholder" value="" disabled>{{ placeholder }}</option>
      <option v-for="option in options" :key="String(option.value)" :value="option.value">
        {{ option.label }}
      </option>
    </select>
    <span v-if="hint" class="text-xs text-fg-subtle">{{ hint }}</span>
  </label>
</template>

<script setup>
const props = defineProps({
  modelValue: { type: [String, Number], default: '' },
  label: { type: String, required: true },
  hint: { type: String, default: '' },
  placeholder: { type: String, default: '' },
  options: { type: Array, default: () => [] },
  valueType: { type: String, default: 'string' },
});

const emit = defineEmits(['update:modelValue']);

function handleChange(event) {
  const rawValue = event.target.value;
  if (props.valueType === 'number') {
    emit('update:modelValue', rawValue === '' ? '' : Number(rawValue));
    return;
  }
  emit('update:modelValue', rawValue);
}
</script>
