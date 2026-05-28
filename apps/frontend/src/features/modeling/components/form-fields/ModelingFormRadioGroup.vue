<template>
  <fieldset class="space-y-2">
    <legend class="text-sm font-medium text-fg">{{ label }}</legend>
    <div class="grid gap-2 sm:grid-cols-2">
      <label
        v-for="option in options"
        :key="String(option.value)"
        class="flex cursor-pointer items-center gap-2 rounded-md border border-border-default p-3 text-sm"
      >
        <input
          :checked="modelValue === option.value"
          type="radio"
          :value="option.value"
          @change="emit('update:modelValue', option.value)"
        />
        <span>{{ option.label }}</span>
      </label>
    </div>
  </fieldset>
</template>

<script setup>
defineProps({
  modelValue: { type: [String, Number, Boolean], default: '' },
  label: { type: String, required: true },
  options: { type: Array, default: () => [] },
});

const emit = defineEmits(['update:modelValue']);
</script>
