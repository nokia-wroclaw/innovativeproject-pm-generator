<template>
  <label class="block space-y-2">
    <span class="text-sm font-medium text-fg">{{ label }}</span>
    <textarea
      :value="modelValue"
      :rows="rows"
      :placeholder="placeholder"
      class="w-full resize-y rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-fg"
      @input="$emit('update:modelValue', $event.target.value)"
    />
    <p v-if="hint" class="text-xs text-fg-subtle">{{ hint }}</p>
  </label>
</template>

<script setup>
defineProps({
  modelValue: { type: String, default: '' },
  label: { type: String, required: true },
  placeholder: { type: String, default: '' },
  hint: { type: String, default: '' },
  rows: { type: Number, default: 4 },
});

defineEmits(['update:modelValue']);
</script>
