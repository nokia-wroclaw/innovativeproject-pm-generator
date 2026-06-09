<script setup>
defineProps({
  rows: {
    type: Array,
    default: () => [],
  },
});
</script>

<template>
  <div class="s3-kpi-catalog-wrap">
    <table v-if="rows.length" class="s3-kpi-catalog-table">
      <thead>
        <tr>
          <th>KPI ID</th>
          <th>Records</th>
          <th>BTS</th>
          <th>Days</th>
          <th>Mean</th>
          <th>Min</th>
          <th>Max</th>
          <th>Null %</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.kpi_id">
          <td>{{ row.kpi_id }}</td>
          <td>{{ row.records_count }}</td>
          <td>{{ row.bts_count }}</td>
          <td>{{ row.day_count }}</td>
          <td>{{ row.kpi_value_mean }}</td>
          <td>{{ row.kpi_value_min }}</td>
          <td>{{ row.kpi_value_max }}</td>
          <td>{{ row.null_pct }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else class="s3-kpi-catalog-empty">No KPI catalog data.</p>
  </div>
</template>
