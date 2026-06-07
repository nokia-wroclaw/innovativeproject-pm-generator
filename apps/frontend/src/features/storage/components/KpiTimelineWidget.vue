<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import Plotly from 'plotly.js-dist-min';

const props = defineProps({
  analysis: {
    type: Object,
    default: null,
  },
});

const containerRef = ref(null);
const selectedKpi = ref('');

const kpiList = computed(() => props.analysis?.kpi_list ?? []);

const kpiPlots = computed(() => props.analysis?.kpi_plots ?? {});

const activeFigure = computed(() => {
  const kpi = selectedKpi.value;
  if (!kpi) return null;
  const plot = kpiPlots.value[kpi];
  if (!plot || plot.error) return null;
  if (plot.data && plot.layout) return plot;
  return null;
});

function render() {
  const el = containerRef.value;
  const fig = activeFigure.value;
  if (!el || !fig) return;

  Plotly.react(el, fig.data, fig.layout, {
    responsive: true,
    displayModeBar: true,
  });
}

watch(kpiList, (list) => {
  if (list.length && !list.includes(selectedKpi.value)) {
    selectedKpi.value = list[0];
  }
}, { immediate: true });

onMounted(render);
watch([activeFigure, selectedKpi], render);

onBeforeUnmount(() => {
  if (containerRef.value) {
    Plotly.purge(containerRef.value);
  }
});

const plotError = computed(() => {
  const kpi = selectedKpi.value;
  if (!kpi) return null;
  return kpiPlots.value[kpi]?.error ?? null;
});
</script>

<template>
  <div v-if="kpiList.length" class="s3-kpi-timeline">
    <label class="s3-kpi-timeline-label" for="kpi-timeline-select">KPI</label>
    <select
      id="kpi-timeline-select"
      v-model="selectedKpi"
      class="s3-kpi-timeline-select"
    >
      <option v-for="kpi in kpiList" :key="kpi" :value="kpi">
        {{ kpi }}
      </option>
    </select>
    <p v-if="plotError" class="s3-kpi-timeline-error">{{ plotError }}</p>
    <div v-else ref="containerRef" class="s3-kpi-timeline-plot" />
  </div>
  <p v-else class="s3-kpi-timeline-empty">
    KPI timeline plots are generated on the next visualization run.
  </p>
</template>

<style scoped>
.s3-kpi-timeline {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.s3-kpi-timeline-label {
  font-size: 0.8rem;
  font-weight: 600;
  color: #374151;
}

.s3-kpi-timeline-select {
  max-width: 280px;
  padding: 8px 10px;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  font-size: 0.875rem;
}

.s3-kpi-timeline-plot {
  width: 100%;
  min-height: 400px;
}

.s3-kpi-timeline-error,
.s3-kpi-timeline-empty {
  margin: 0;
  color: #6b7280;
  font-size: 0.875rem;
}
</style>
