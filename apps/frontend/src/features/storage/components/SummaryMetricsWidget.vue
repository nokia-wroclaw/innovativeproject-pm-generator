<script setup>
import { computed } from 'vue';

const props = defineProps({
  basicInfo: {
    type: Object,
    default: null,
  },
  sparkVersion: {
    type: String,
    default: '',
  },
});

const metrics = computed(() => {
  const info = props.basicInfo ?? {};
  return [
    { label: 'Rows', value: info.rows_count ?? '—' },
    { label: 'KPIs', value: info.kpi_count ?? '—' },
    { label: 'BTS', value: info.bts_count ?? '—' },
    { label: 'Distnames', value: info.distname_count ?? '—' },
    { label: 'Start date', value: info.start_date ?? '—' },
    { label: 'End date', value: info.end_date ?? '—' },
  ];
});
</script>

<template>
  <div class="s3-summary-metrics">
    <div v-if="sparkVersion" class="s3-summary-metrics-spark">
      Spark <strong>{{ sparkVersion }}</strong>
    </div>
    <div class="s3-summary-metrics-grid">
      <div v-for="item in metrics" :key="item.label" class="s3-summary-metric-card">
        <span class="s3-summary-metric-label">{{ item.label }}</span>
        <span class="s3-summary-metric-value">{{ item.value }}</span>
      </div>
    </div>
  </div>
</template>
