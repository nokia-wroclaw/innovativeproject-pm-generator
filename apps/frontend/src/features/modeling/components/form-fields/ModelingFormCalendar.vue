<template>
  <label class="block space-y-2">
    <span class="text-sm font-medium text-fg">{{ label }}</span>
    <input
      :value="modelValue"
      type="date"
      class="w-full rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
      @input="emit('update:modelValue', $event.target.value)"
    />
    <span v-if="hint" class="text-xs text-fg-subtle">{{ hint }}</span>
  </label>
</template>

<script setup>
defineProps({
  modelValue: { type: String, default: '' },
  label: { type: String, required: true },
  hint: { type: String, default: '' },
});

const emit = defineEmits(['update:modelValue']);
</script>
