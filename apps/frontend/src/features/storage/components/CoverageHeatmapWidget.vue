<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import Plotly from 'plotly.js-dist-min';

const props = defineProps({
  coverage: {
    type: Object,
    default: null,
  },
  height: {
    type: Number,
    default: 520,
  },
});

const containerRef = ref(null);

function normalizePlotlyHeatmap(raw) {
  if (!raw || typeof raw !== 'object') return null;
  if (raw.data && raw.layout) {
    const trace = raw.data.find((t) => t.type === 'heatmap') || raw.data[0];
    if (trace?.z) {
      return {
        mode: 'figure',
        data: raw.data,
        layout: raw.layout,
      };
    }
  }
  return null;
}

function normalizeCoverage(raw) {
  if (!raw) return null;

  if (typeof raw === 'string') {
    try {
      return normalizeCoverage(JSON.parse(raw));
    } catch {
      return null;
    }
  }

  if (typeof raw !== 'object') return null;

  const plotly = normalizePlotlyHeatmap(raw);
  if (plotly) return plotly;

  if (raw.z && raw.x && raw.y) {
    return {
      mode: 'matrix',
      z: raw.z,
      x: raw.x,
      y: raw.y,
    };
  }

  return null;
}

function render() {
  const el = containerRef.value;
  const data = normalizeCoverage(props.coverage);
  if (!el || !data) return;

  if (data.mode === 'figure') {
    const layout = { ...data.layout, height: props.height, autosize: true };
    Plotly.react(el, data.data, layout, { responsive: true, displayModeBar: true });
    return;
  }

  Plotly.react(
    el,
    [
      {
        type: 'heatmap',
        z: data.z,
        x: data.x,
        y: data.y,
        colorscale: [
          [0, '#ef4444'],
          [1, '#22c55e'],
        ],
        showscale: false,
        hovertemplate: 'BTS: %{y}<br>KPI: %{x}<br>present: %{z}<extra></extra>',
      },
    ],
    {
      title: 'KPI Coverage per BTS (green = present, red = missing)',
      xaxis: { title: 'KPI ID' },
      yaxis: { title: 'BTS ID' },
      autosize: true,
      height: props.height,
      template: 'plotly_dark',
      paper_bgcolor: '#0F172A',
      plot_bgcolor: '#1E293B',
    },
    { responsive: true, displayModeBar: true },
  );
}

onMounted(render);
watch(() => [props.coverage, props.height], render);

onBeforeUnmount(() => {
  if (containerRef.value) {
    Plotly.purge(containerRef.value);
  }
});
</script>

<template>
  <div ref="containerRef" class="s3-coverage-heatmap" />
</template>

<style scoped>
.s3-coverage-heatmap {
  width: 100%;
  min-height: 320px;
}
</style>
