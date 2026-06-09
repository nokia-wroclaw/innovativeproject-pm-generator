<script setup>
defineProps({
  rows: {
    type: Array,
    default: () => [],
  },
});
</script>

<template>
  <div class="s3-schema-table-wrap">
    <table v-if="rows.length" class="s3-schema-table">
      <thead>
        <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Nullable</th>
          <th>Null %</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.column">
          <td><code>{{ row.column }}</code></td>
          <td>{{ row.type }}</td>
          <td>{{ row['nullable?'] ?? row.nullable }}</td>
          <td>{{ row.null_pct }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else class="s3-schema-table-empty">No schema data.</p>
  </div>
</template>
