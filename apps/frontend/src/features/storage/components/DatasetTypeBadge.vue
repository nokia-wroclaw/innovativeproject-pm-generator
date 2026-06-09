<script setup>
import { computed } from 'vue';

const props = defineProps({
  type: {
    type: String,
    default: '',
  },
});

const TYPE_META = {
  RAW: { label: 'Raw', className: 's3-type-raw' },
  PREPROCESSED: { label: 'Preprocessed', className: 's3-type-preprocessed' },
  GENERATED: { label: 'Generated', className: 's3-type-generated' },
};

const meta = computed(() => {
  const key = String(props.type || '').toUpperCase();
  return TYPE_META[key] ?? { label: props.type || 'Unknown', className: 's3-type-unknown' };
});
</script>

<template>
  <span :class="['s3-type-badge', meta.className]">{{ meta.label }}</span>
</template>
