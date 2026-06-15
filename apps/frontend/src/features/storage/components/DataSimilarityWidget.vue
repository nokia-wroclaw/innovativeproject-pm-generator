<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import Plotly from 'plotly.js-dist-min';

const props = defineProps({
  summary: {
    type: Object,
    default: null,
  },
});

const singleKpiRef = ref(null);
const multiKpiRef = ref(null);

const singleKpis = computed(() => Object.keys(props.summary?.single_kpi ?? {}));
const selectedKpi = ref('');

watch(singleKpis, (list) => {
  if (list.length && !list.includes(selectedKpi.value)) {
    selectedKpi.value = list[0];
  }
}, { immediate: true });

const activeSingleFigure = computed(() => {
  const kpi = selectedKpi.value;
  if (!kpi) return null;
  return props.summary?.single_kpi?.[kpi]?.figure ?? null;
});

const activeMultiFigure = computed(() => props.summary?.multi_kpi?.figure ?? null);

const SINGLE_METRIC_COLS = [
  { key: 'wasserstein_1d',      label: 'Wasserstein' },
  { key: 'jensen_shannon',      label: 'Jensen–Shannon' },
  { key: 'mmd_rbf',             label: 'MMD (RBF)' },
  { key: 'ls_spectrum_distance',label: 'LS spectrum' },
  { key: 'acf_distance',        label: 'ACF dist.' },
  { key: 'hourly_profile_rmse', label: 'Hourly RMSE' },
];

const singleSummaryRows = computed(() =>
  singleKpis.value.map((kpi) => {
    const e = props.summary?.single_kpi?.[kpi] ?? {};
    return {
      kpi,
      values: SINGLE_METRIC_COLS.map(({ key }) =>
        e[key] != null ? Number(e[key]).toFixed(4) : '—',
      ),
    };
  }),
);

const multiMetricRows = computed(() => {
  const m = props.summary?.multi_kpi;
  if (!m) return null;
  return [
    { label: 'Sliced Wasserstein', val: m.sliced_wasserstein?.toFixed(4) },
    { label: 'MMD (multivariate)', val: m.mmd_multivariate?.toFixed(4) },
    { label: 'Pairwise corr dist.', val: m.pairwise_corr_distance?.toFixed(4) },
    { label: 'Partial corr dist.', val: m.partial_corr_distance?.toFixed(4) },
  ];
});

function renderFigure(containerRef, fig) {
  const el = containerRef.value;
  if (!el || !fig) return;
  Plotly.react(el, fig.data, fig.layout, { responsive: true, displayModeBar: true });
}

watch([activeSingleFigure, singleKpiRef], () => renderFigure(singleKpiRef, activeSingleFigure.value));
watch([activeMultiFigure, multiKpiRef], () => renderFigure(multiKpiRef, activeMultiFigure.value));

onBeforeUnmount(() => {
  if (singleKpiRef.value) Plotly.purge(singleKpiRef.value);
  if (multiKpiRef.value) Plotly.purge(multiKpiRef.value);
});
</script>

<template>
  <div v-if="summary" class="ds-widget">

    <!-- ── Summary table: all KPIs × all metrics, visible without scrolling ── -->
    <div v-if="singleKpis.length" class="s3-dataset-viz-block">
      <h3 class="s3-dataset-viz-subtitle">Metrics summary</h3>
      <div class="ds-widget-table-wrap">
        <table class="ds-widget-summary">
          <thead>
            <tr>
              <th class="ds-widget-summary-kpi-head">KPI</th>
              <th v-for="col in SINGLE_METRIC_COLS" :key="col.key">{{ col.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in singleSummaryRows"
              :key="row.kpi"
              class="ds-widget-summary-row"
              :class="{ 'ds-widget-summary-row--active': row.kpi === selectedKpi }"
              @click="selectedKpi = row.kpi"
            >
              <td class="ds-widget-summary-kpi">{{ row.kpi }}</td>
              <td v-for="(val, i) in row.values" :key="i" class="ds-widget-summary-val">{{ val }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- ── Single-KPI plot ── -->
    <div v-if="singleKpis.length" class="s3-dataset-viz-block">
      <h3 class="s3-dataset-viz-subtitle">Single-KPI plot</h3>
      <div class="ds-widget-kpi-row">
        <label class="ds-widget-label" for="ds-kpi-select">KPI</label>
        <select id="ds-kpi-select" v-model="selectedKpi" class="ds-widget-select">
          <option v-for="kpi in singleKpis" :key="kpi" :value="kpi">{{ kpi }}</option>
        </select>
      </div>
      <div ref="singleKpiRef" class="ds-widget-plot" />
    </div>

    <!-- ── Multi-KPI section ── -->
    <div v-if="activeMultiFigure" class="s3-dataset-viz-block">
      <h3 class="s3-dataset-viz-subtitle">Multi-KPI similarity</h3>
      <table v-if="multiMetricRows" class="ds-widget-multi-metrics">
        <tbody>
          <tr v-for="row in multiMetricRows" :key="row.label">
            <td class="ds-widget-multi-label">{{ row.label }}</td>
            <td class="ds-widget-multi-val">{{ row.val }}</td>
          </tr>
        </tbody>
      </table>
      <div ref="multiKpiRef" class="ds-widget-plot" />
    </div>

  </div>
</template>

<style scoped>
.ds-widget {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* scrollable wrapper so the table never forces the page to be wider */
.ds-widget-table-wrap {
  overflow-x: auto;
}

.ds-widget-summary {
  border-collapse: collapse;
  font-size: 0.83rem;
  white-space: nowrap;
  width: 100%;
}

.ds-widget-summary thead th {
  padding: 6px 14px 6px 0;
  font-weight: 600;
  color: #374151;
  border-bottom: 2px solid #e5e7eb;
  text-align: left;
}

.ds-widget-summary-kpi-head {
  padding-right: 20px !important;
}

.ds-widget-summary-row {
  cursor: pointer;
  transition: background 0.1s;
}

.ds-widget-summary-row:hover {
  background: #f3f4f6;
}

.ds-widget-summary-row--active {
  background: #eff6ff;
}

.ds-widget-summary-row--active .ds-widget-summary-kpi {
  font-weight: 700;
  color: #1d4ed8;
}

.ds-widget-summary-kpi {
  padding: 5px 20px 5px 0;
  font-weight: 500;
  color: #111827;
}

.ds-widget-summary-val {
  padding: 5px 14px 5px 0;
  font-family: monospace;
  color: #374151;
  text-align: right;
}

.ds-widget-kpi-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ds-widget-label {
  font-size: 0.8rem;
  font-weight: 600;
  color: #374151;
}

.ds-widget-select {
  max-width: 240px;
  padding: 6px 10px;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  font-size: 0.875rem;
}

.ds-widget-plot {
  width: 100%;
  min-height: 500px;
}

.ds-widget-multi-metrics {
  border-collapse: collapse;
  margin-bottom: 12px;
  font-size: 0.85rem;
}

.ds-widget-multi-metrics td {
  padding: 4px 16px 4px 0;
}

.ds-widget-multi-label {
  color: #6b7280;
  white-space: nowrap;
}

.ds-widget-multi-val {
  font-family: monospace;
  font-weight: 600;
  color: #111827;
}
</style>
