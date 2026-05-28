<template>
  <label class="block space-y-2">
    <span class="text-sm font-medium text-fg">{{ label }}</span>
    <input
      :value="modelValue"
      :type="type"
      :min="min"
      :max="max"
      :step="step"
      :placeholder="placeholder"
      class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
      @input="handleInput"
    />
  </label>
</template>

<script setup>
const props = defineProps({
  modelValue: { type: [String, Number], default: '' },
  label: { type: String, required: true },
  type: { type: String, default: 'text' },
  placeholder: { type: String, default: '' },
  min: { type: [String, Number], default: undefined },
  max: { type: [String, Number], default: undefined },
  step: { type: [String, Number], default: undefined },
  valueType: { type: String, default: 'string' },
});

const emit = defineEmits(['update:modelValue']);

function handleInput(event) {
  const rawValue = event.target.value;
  if (props.valueType === 'number') {
    emit('update:modelValue', rawValue === '' ? '' : Number(rawValue));
    return;
  }
  emit('update:modelValue', rawValue);
}
</script>
