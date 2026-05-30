<template>
  <label class="flex items-center gap-2 text-sm text-fg">
    <input
      :checked="modelValue"
      type="checkbox"
      @change="emit('update:modelValue', $event.target.checked)"
    />
    {{ label }}
  </label>
</template>

<script setup>
defineProps({
  modelValue: { type: Boolean, default: false },
  label: { type: String, required: true },
});

const emit = defineEmits(['update:modelValue']);
</script>
