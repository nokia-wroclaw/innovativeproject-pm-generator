<script setup>
import { ref } from 'vue';
import DataTable from '../components/DataTable.vue';
import BaseModal from '../components/BaseModal.vue';

import { fetchS3DatasetsPage, createS3Dataset } from '../services/s3'; 

const tableRef = ref(null);
const isModalOpen = ref(false);
const isSubmitting = ref(false);

const formData = ref({
  file_name: '',
  s3_key: ''
});

const tableColumns = [
  { key: 'id', label: 'ID' },
  { key: 'file_name', label: 'Name' },
  { key: 's3_bucket', label: 'Bucket' },
  { key: 's3_key', label: 'Path' },
  { key: 'status', label: 'Status' }
];

const openModal = () => isModalOpen.value = true;

const closeModal = () => {
  isModalOpen.value = false;
  formData.value = { file_name: '', s3_key: '' };
};

const submitDataset = async () => {
  if (!formData.value.file_name || !formData.value.s3_key) return;

  isSubmitting.value = true;
  try {
    await createS3Dataset(formData.value);

    closeModal();
    if (tableRef.value) {
      tableRef.value.refresh();
    }
  } catch (error) {
    console.error('Error during creation of dataset:', error);
  } finally {
    isSubmitting.value = false;
  }
};
</script>

<template>
  <div >
    <div class="page-header">
      <h2>Manage datasets</h2>
      <button class="primary-button" @click="openModal">
        Add dataset
      </button>
    </div>

    <DataTable
      ref="tableRef"
      :columns="tableColumns"
      :provider="fetchS3DatasetsPage"
      :per-page="10"
    >
      <template #cell-status="{ row }">
        <span 
          v-if="row.status"
          :class="['status-badge', `status-${row.status.toLowerCase()}`]"
        >
          {{ row.status }}
        </span>
        <span v-else class="status-badge status-unknown">No data</span>
      </template>
    </DataTable>

    <BaseModal
      :show="isModalOpen"
      title="Add new S3 dataset"
      width="500px"
      @close="closeModal"
    >
      <form @submit.prevent="submitDataset" class="dataset-form">
        <div class="form-group">
          <label for="fileName">File name:</label>
          <input
            id="fileName"
            v-model="formData.file_name"
            type="text"
            required
            class="form-input"
          />
        </div>
        <div class="form-group">
          <label for="s3Bucket">Bucket</label>
          <input
            id="s3Bucket"
            v-model="formData.s3_bucket"
            type="text"
            required
            class="form-input"
          />
        </div>
        <div class="form-group">
          <label for="s3Key">Path</label>
          <input
            id="s3Key"
            v-model="formData.s3_key"
            type="text"
            required
            class="form-input"
          />
        </div>
      </form>

      <template #footer>
        <button
          type="button"
          class="secondary-button"
          @click="closeModal"
          :disabled="isSubmitting"
        >
          Cancel
        </button>
        <button
          type="button"
          class="primary-button"
          @click="submitDataset"
          :disabled="isSubmitting || !formData.file_name || !formData.s3_key"
        >
          Confirm
        </button>
      </template>
    </BaseModal>
  </div>
</template>

<style scoped>
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  color: #111827;
  font-size: 1.5rem;
}

.dataset-form {
  display: flex;
  flex-direction: column;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.form-group label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.form-input {
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.95rem;
  color: #1f2937;
  outline: none;
  transition: border-color 0.2s;
}

.form-input:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.primary-button {
  background-color: #3b82f6;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

.primary-button:hover:not(:disabled) {
  background-color: #2563eb;
}

.primary-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.secondary-button {
  background-color: white;
  color: #374151;
  border: 1px solid #d1d5db;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.secondary-button:hover:not(:disabled) {
  background-color: #f9fafb;
  color: #111827;
}

.status-badge {
  padding: 4px 10px;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.status-pending { background-color: #fef08a; color: #854d0e; }
.status-processing { background-color: #dbeafe; color: #1e40af; }
.status-completed { background-color: #dcfce7; color: #166534; }
.status-failed { background-color: #fee2e2; color: #991b1b; }
.status-unknown { background-color: #f3f4f6; color: #4b5563; }
</style>