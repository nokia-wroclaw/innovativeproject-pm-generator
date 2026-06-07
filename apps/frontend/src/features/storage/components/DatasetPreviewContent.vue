<script setup>
import { formatPreviewValue } from '../utils/formatPreviewValue';

defineProps({
  loading: { type: Boolean, default: false },
  error: { type: String, default: '' },
  previewData: { type: Object, default: null },
  scrollable: { type: Boolean, default: true },
});
</script>

<template>
  <div :class="['s3-preview-content', { 's3-preview-content--scroll': scrollable }]">
    <p v-if="loading" class="s3-preview-status">Loading preview...</p>
    <p v-else-if="error" class="s3-preview-error">{{ error }}</p>

    <template v-else-if="previewData">
      <p class="s3-preview-meta">
        Tables found: <strong>{{ previewData.tables?.length || 0 }}</strong>
      </p>

      <section
        v-for="table in previewData.tables"
        :key="table.name"
        class="s3-preview-table-section"
      >
        <h3 class="s3-preview-table-title">{{ table.name }}</h3>
        <p class="s3-preview-columns">
          Columns:
          <span>{{ table.columns?.join(', ') || '—' }}</span>
        </p>

        <div class="s3-preview-table-wrap">
          <table class="s3-preview-table">
            <thead>
              <tr>
                <th v-for="column in table.columns" :key="column">{{ column }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, rowIndex) in table.rows" :key="rowIndex">
                <td v-for="column in table.columns" :key="column">
                  {{ formatPreviewValue(row[column]) }}
                </td>
              </tr>
              <tr v-if="!table.rows?.length">
                <td :colspan="table.columns?.length || 1" class="s3-preview-empty">
                  No rows available
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </template>
  </div>
</template>
