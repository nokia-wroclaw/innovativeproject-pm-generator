<template>
  <label class="block space-y-2">
    <span class="text-sm font-medium text-fg">
      {{ label }}
      <span v-if="required" class="ml-0.5 text-rose-600" aria-hidden="true">*</span>
    </span>
    <input
      :value="modelValue"
      :type="isNumeric ? 'text' : type"
      :inputmode="isNumeric ? numericInputMode : undefined"
      :placeholder="placeholder"
      :aria-required="required"
      class="w-full rounded-md border bg-surface px-3 py-2 text-sm text-fg"
      :class="hasError ? 'border-rose-300' : 'border-border-default'"
      :aria-invalid="hasError"
      @keydown="handleKeydown"
      @paste="handlePaste"
      @input="handleInput"
    />
    <span v-if="rangeError" class="text-xs text-rose-600">{{ rangeError }}</span>
  </label>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  modelValue: { type: [String, Number], default: '' },
  label: { type: String, required: true },
  type: { type: String, default: 'text' },
  placeholder: { type: String, default: '' },
  min: { type: [String, Number], default: undefined },
  max: { type: [String, Number], default: undefined },
  step: { type: [String, Number], default: undefined },
  valueType: { type: String, default: 'string' },
  required: { type: Boolean, default: false },
});

const emit = defineEmits(['update:modelValue']);

const isNumeric = computed(() => props.valueType === 'number' || props.type === 'number');
const allowsDecimal = computed(
  () => props.step != null && !Number.isInteger(Number(props.step)),
);
const numericInputMode = computed(() => (allowsDecimal.value ? 'decimal' : 'numeric'));

const rangeError = computed(() => {
  if (!isNumeric.value || props.modelValue === '' || props.modelValue === '.') return '';
  if (typeof props.modelValue === 'string' && props.modelValue.endsWith('.')) return '';

  const num = Number(props.modelValue);
  if (Number.isNaN(num)) return '';

  if (props.min != null && num < Number(props.min)) {
    return `Value must be at least ${props.min}.`;
  }
  if (props.max != null && num > Number(props.max)) {
    return `Value must be at most ${props.max}.`;
  }
  return '';
});

const hasError = computed(
  () => (props.required && props.modelValue === '') || Boolean(rangeError.value),
);

const NAVIGATION_KEYS = new Set([
  'Backspace',
  'Delete',
  'Tab',
  'Escape',
  'Enter',
  'ArrowLeft',
  'ArrowRight',
  'Home',
  'End',
]);

function sanitizeNumeric(rawValue) {
  if (rawValue === '') return '';

  if (allowsDecimal.value) {
    let cleaned = rawValue.replace(/[^\d.]/g, '');
    const dotIndex = cleaned.indexOf('.');
    if (dotIndex !== -1) {
      cleaned = cleaned.slice(0, dotIndex + 1) + cleaned.slice(dotIndex + 1).replace(/\./g, '');
    }
    return cleaned;
  }

  return rawValue.replace(/\D/g, '');
}

function emitNumericValue(value) {
  if (value === '' || value === '.') {
    emit('update:modelValue', '');
    return;
  }

  if (allowsDecimal.value && value.endsWith('.')) {
    emit('update:modelValue', value);
    return;
  }

  const parsed = Number(value);
  emit('update:modelValue', Number.isNaN(parsed) ? '' : parsed);
}

function handleKeydown(event) {
  if (!isNumeric.value) return;
  if (NAVIGATION_KEYS.has(event.key) || event.ctrlKey || event.metaKey) return;

  if (allowsDecimal.value && event.key === '.') {
    if (String(event.target.value).includes('.')) event.preventDefault();
    return;
  }

  if (!/^\d$/.test(event.key)) event.preventDefault();
}

function applyNumericValue(event, rawValue) {
  const sanitized = sanitizeNumeric(rawValue);
  if (sanitized !== rawValue) event.target.value = sanitized;
  emitNumericValue(sanitized);
}

function handlePaste(event) {
  if (!isNumeric.value) return;
  event.preventDefault();
  applyNumericValue(event, (event.clipboardData?.getData('text') ?? '').trim());
}

function handleInput(event) {
  if (isNumeric.value) {
    applyNumericValue(event, event.target.value);
    return;
  }
  emit('update:modelValue', event.target.value);
}
</script>
