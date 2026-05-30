<template>
  <div class="s3-tab-body">
    <p v-if="description" class="s3-tab-description">{{ description }}</p>

    <div v-if="showAddButton" class="s3-toolbar">
      <button type="button" class="btn-primary" @click="$emit('add-dataset')">
        Add dataset
      </button>
    </div>

    <DataTable
      ref="tableRef"
      :columns="columns"
      :provider="provider"
      :per-page="10"
    >
      <template #cell-status="{ row }">
        <div v-if="row.status === uploadingStatus" class="s3-upload-cell">
          <div class="s3-progress-bar">
            <div class="s3-progress-fill" :style="{ width: `${row.progress}%` }" />
          </div>
          <span class="s3-progress-text">{{ row.progress }}%</span>
        </div>

        <span
          v-else-if="row.status"
          :class="['s3-status-badge', `s3-status-${row.status.toLowerCase()}`]"
        >
          {{ row.status }}
        </span>

        <span v-else class="s3-status-badge s3-status-unknown">No data</span>
      </template>

      <template #cell-actions="{ row }">
        <DynamicActions
          v-if="row.status === completedStatus"
          :row="row"
          :actions="rowActions"
          @action="$emit('row-action', $event)"
        />
        <span v-else class="s3-status-waiting">Uploading...</span>
      </template>
    </DataTable>
  </div>
</template>

<script setup>
import { ref } from 'vue';

import DataTable from './DataTable.vue';
import DynamicActions from './TableActions.vue';

defineProps({
  description: { type: String, default: '' },
  showAddButton: { type: Boolean, default: false },
  columns: { type: Array, required: true },
  provider: { type: Function, required: true },
  rowActions: { type: Array, required: true },
  uploadingStatus: { type: String, required: true },
  completedStatus: { type: String, required: true },
});

defineEmits(['add-dataset', 'row-action']);

const tableRef = ref(null);

function refresh() {
  tableRef.value?.refresh();
}

defineExpose({ refresh });
</script>
