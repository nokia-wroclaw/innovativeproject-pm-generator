<script setup>
import '../assets/S3.css';
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue';
import DataTable from '../components/DataTable.vue';
import BaseModal from '../components/BaseModal.vue';
import DynamicActions from '../components/TableActions.vue';
import DeleteAction from '../components/DeleteAction.vue';
import {
  fetchS3DatasetsPage,
  createS3Dataset,
  deleteS3Dataset,
  updateS3Status,
  DatasetStatus,
  initiateMultipartUpload,
  getPresignedPartUrl,
  completeMultipartUpload,
  abortMultipartUpload,
  registerExistingS3Dataset,
  fetchDatasetPreview,
} from '../services/s3';
import { isAdmin } from '../auth/keycloak';

const CHUNK_SIZE = 5 * 1024 * 1024;
const STORAGE_KEY = 's3_pending_upload';

const TABLE_COLUMNS = [
  { key: 'id', label: 'ID' },
  { key: 'file_name', label: 'Name' },
  { key: 's3_key', label: 'Path' },
  { key: 'status', label: 'Status' },
  { key: 'actions', label: 'Actions' },
];

const rowActions = computed(() => {
  const actions = [
    { id: 'analyze', label: 'Preview dataset', class: 's3-action-analyze' },
  ];
  if (isAdmin.value) {
    actions.push({ id: 'delete', label: 'Delete', class: 's3-action-delete' });
  }
  return actions;
});

const tableRef = ref(null);
const isAddModalOpen = ref(false);
const isPreparing = ref(false);
const activeRow = ref(null);
const showDeleteModal = ref(false);
const isPreviewModalOpen = ref(false);
const isPreviewLoading = ref(false);
const previewError = ref('');
const previewData = ref(null);
const activeUploads = ref([]);

const uploadMode = ref('upload');
const formData = ref({ file_name: '', s3_key: '', file: null });

const pendingResumeState = ref(null);
const isResumeModalOpen = ref(false);
const resumeFile = ref(null);
const resumeError = ref('');

const isUploadMode = computed(() => uploadMode.value === 'upload');
const isRegisterMode = computed(() => uploadMode.value === 'register');

const isSubmitDisabled = computed(() => {
  if (isPreparing.value || !formData.value.s3_key) return true;
  if (isUploadMode.value) {
    return !formData.value.file_name || !formData.value.file;
  }
  return false;
});

const submitButtonLabel = computed(() => {
  if (isPreparing.value) return 'Processing...';
  if (isRegisterMode.value) return 'Register file';
  return 'Start upload';
});

const hasActiveUpload = computed(() =>
  activeUploads.value.some((task) => task.status === DatasetStatus.UPLOADING)
);

function resetForm() {
  formData.value = { file_name: '', s3_key: '', file: null };
  uploadMode.value = 'upload';
}

function openAddModal() {
  isAddModalOpen.value = true;
}

function closeAddModal() {
  isAddModalOpen.value = false;
  resetForm();
}

function refreshTable() {
  tableRef.value?.refresh();
}

function createUploadTask({ id, file_name, s3_key }) {
  return reactive({
    id,
    file_name,
    s3_key,
    status: DatasetStatus.UPLOADING,
    progress: 0,
  });
}

function persistUploadState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function clearPersistedUpload() {
  localStorage.removeItem(STORAGE_KEY);
}

function handleBeforeUnload(event) {
  if (!hasActiveUpload.value) return;
  event.preventDefault();
  event.returnValue = '';
}

function closePreviewModal() {
  isPreviewModalOpen.value = false;
  previewError.value = '';
  previewData.value = null;
}

async function openDatasetPreview(row) {
  activeRow.value = row;
  isPreviewModalOpen.value = true;
  isPreviewLoading.value = true;
  previewError.value = '';
  previewData.value = null;

  try {
    previewData.value = await fetchDatasetPreview(row.id);
  } catch (error) {
    previewError.value = error.message || 'Failed to load dataset preview';
  } finally {
    isPreviewLoading.value = false;
  }
}

function formatPreviewValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function handleRowAction({ type, row }) {
  activeRow.value = row;
  if (type === 'analyze') {
    openDatasetPreview(row);
  } else if (type === 'delete') {
    showDeleteModal.value = true;
  }
}

function handleFileChange(event) {
  const selectedFile = event.target.files?.[0];
  if (!selectedFile) return;

  formData.value.file = selectedFile;
  if (!formData.value.file_name) {
    formData.value.file_name = selectedFile.name;
  }
}

function handleResumeFileChange(event) {
  resumeError.value = '';
  resumeFile.value = event.target.files?.[0] ?? null;
}

async function uploadChunk(url, chunkData) {
  const response = await fetch(url, { method: 'PUT', body: chunkData });
  if (!response.ok) {
    throw new Error(`Chunk upload failed with status ${response.status}`);
  }

  const etag = response.headers.get('ETag');
  if (!etag) {
    throw new Error('ETag not found. Check S3 CORS ExposeHeaders configuration.');
  }
  return etag;
}

async function processUploadLoop(file, uploadState, uploadTask) {
  try {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

    for (let partNumber = 1; partNumber <= totalChunks; partNumber += 1) {
      const alreadyUploaded = uploadState.uploadedParts.some(
        (part) => part.PartNumber === partNumber
      );

      if (alreadyUploaded) {
        uploadTask.progress = Math.round(
          (uploadState.uploadedParts.length / totalChunks) * 100
        );
        continue;
      }

      const start = (partNumber - 1) * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const { url } = await getPresignedPartUrl(
        uploadState.datasetId,
        uploadState.s3UploadId,
        partNumber
      );
      const etag = await uploadChunk(url, chunk);

      uploadState.uploadedParts.push({ PartNumber: partNumber, ETag: etag });
      persistUploadState(uploadState);

      uploadTask.progress = Math.round(
        (uploadState.uploadedParts.length / totalChunks) * 100
      );
    }

    await completeMultipartUpload(
      uploadState.datasetId,
      uploadState.s3UploadId,
      uploadState.uploadedParts
    );
    await updateS3Status(uploadState.datasetId, DatasetStatus.COMPLETED);

    clearPersistedUpload();
    uploadTask.status = DatasetStatus.COMPLETED;

    setTimeout(() => {
      activeUploads.value = activeUploads.value.filter((task) => task.id !== uploadTask.id);
      refreshTable();
    }, 1500);
  } catch (error) {
    console.error('Multipart upload failed:', error);
    uploadTask.status = DatasetStatus.FAILED;
  }
}

async function startMultipartUpload(file, fileName, s3Key) {
  const dataset = await createS3Dataset({ file_name: fileName, s3_key: s3Key });
  const initResponse = await initiateMultipartUpload(dataset.id);

  const uploadState = {
    datasetId: dataset.id,
    s3UploadId: initResponse.upload_id,
    s3_key: initResponse.s3_key,
    fileName: file.name,
    fileSize: file.size,
    uploadedParts: [],
  };

  persistUploadState(uploadState);

  const uploadTask = createUploadTask({
    id: `upload_${Date.now()}`,
    file_name: fileName,
    s3_key: initResponse.s3_key,
  });

  activeUploads.value.unshift(uploadTask);
  closeAddModal();
  refreshTable();

  await processUploadLoop(file, uploadState, uploadTask);
}

async function submitDataset() {
  const { file_name, s3_key, file } = formData.value;
  if (!s3_key) return;

  isPreparing.value = true;

  try {
    if (isRegisterMode.value) {
      await registerExistingS3Dataset({ file_name, s3_key });
      closeAddModal();
      refreshTable();
      return;
    }

    if (!file_name || !file) return;

    await startMultipartUpload(file, file_name, s3_key);
  } catch (error) {
    console.error('Failed to process dataset:', error);
    alert(error.message || 'Error processing dataset');
  } finally {
    isPreparing.value = false;
  }
}

async function confirmResume() {
  if (!resumeFile.value || !pendingResumeState.value) return;

  const { fileName, fileSize, datasetId, s3_key } = pendingResumeState.value;

  if (resumeFile.value.name !== fileName || resumeFile.value.size !== fileSize) {
    resumeError.value =
      'Selected file does not match the interrupted upload. Please select the correct file.';
    return;
  }

  isResumeModalOpen.value = false;

  const uploadTask = createUploadTask({
    id: `upload_resumed_${Date.now()}`,
    file_name: fileName,
    s3_key,
  });

  activeUploads.value.unshift(uploadTask);
  refreshTable();

  await updateS3Status(datasetId, DatasetStatus.UPLOADING);
  await processUploadLoop(resumeFile.value, pendingResumeState.value, uploadTask);
}

async function cancelResumeUpload() {
  if (!pendingResumeState.value) return;

  try {
    const { datasetId, s3UploadId } = pendingResumeState.value;
    await abortMultipartUpload(datasetId, s3UploadId);
    await updateS3Status(datasetId, DatasetStatus.FAILED);
  } catch (error) {
    console.error('Failed to abort upload on backend:', error);
  }

  clearPersistedUpload();
  pendingResumeState.value = null;
  isResumeModalOpen.value = false;
  refreshTable();
}

function onDeleteSuccess() {
  showDeleteModal.value = false;
  refreshTable();
}

async function tableProviderWrapper(params) {
  const serverData = await fetchS3DatasetsPage(params);
  const items = Array.isArray(serverData)
    ? serverData
    : serverData.data || serverData.items || [];

  const filteredItems = items.filter((item) => item.status !== DatasetStatus.PENDING);
  const mergedItems = [...activeUploads.value, ...filteredItems];

  if (Array.isArray(serverData)) return mergedItems;
  return { ...serverData, data: mergedItems, items: mergedItems };
}

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload);

  const savedState = localStorage.getItem(STORAGE_KEY);
  if (savedState) {
    pendingResumeState.value = JSON.parse(savedState);
    isResumeModalOpen.value = true;
  }
});

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload);
});
</script>

<template>
  <div class="s3-page">
    <div class="s3-toolbar">
      <button type="button" class="btn-primary" @click="openAddModal">
        Add dataset
      </button>
    </div>

    <DataTable
      ref="tableRef"
      :columns="TABLE_COLUMNS"
      :provider="tableProviderWrapper"
      :per-page="10"
    >
      <template #cell-status="{ row }">
        <div v-if="row.status === DatasetStatus.UPLOADING" class="s3-upload-cell">
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
          v-if="row.status === DatasetStatus.COMPLETED"
          :row="row"
          :actions="rowActions"
          @action="handleRowAction"
        />
        <span v-else class="s3-status-waiting">Uploading...</span>
      </template>
    </DataTable>

    <BaseModal
      :show="isAddModalOpen"
      title="Add S3 dataset"
      width="500px"
      @close="closeAddModal"
    >
      <form class="s3-form" @submit.prevent="submitDataset">
        <div class="form-group">
          <label>Action type</label>
          <div class="s3-radio-group">
            <label class="s3-radio-label">
              <input
                v-model="uploadMode"
                type="radio"
                value="upload"
                :disabled="isPreparing"
              />
              Upload local file
            </label>
            <label class="s3-radio-label">
              <input
                v-model="uploadMode"
                type="radio"
                value="register"
                :disabled="isPreparing"
              />
              Register existing S3 file
            </label>
          </div>
        </div>

        <div class="form-group">
          <label for="fileName">
            File name
            <span v-if="isRegisterMode" class="s3-optional">(optional)</span>
          </label>
          <input
            id="fileName"
            v-model="formData.file_name"
            type="text"
            class="form-input"
            :required="isUploadMode"
            :disabled="isPreparing"
            :placeholder="isRegisterMode ? 'Leave empty to use S3 key name' : undefined"
          />
        </div>

        <div class="form-group">
          <label for="s3Key">Path (S3 key)</label>
          <input
            id="s3Key"
            v-model="formData.s3_key"
            type="text"
            class="form-input"
            required
            :disabled="isPreparing"
            placeholder="e.g. data/my-file.csv"
          />
        </div>

        <div v-if="isUploadMode" class="form-group">
          <label for="fileUpload">Select file</label>
          <input
            id="fileUpload"
            type="file"
            class="s3-form-input-file"
            required
            :disabled="isPreparing"
            @change="handleFileChange"
          />
        </div>
      </form>

      <template #footer>
        <button
          type="button"
          class="btn-secondary"
          :disabled="isPreparing"
          @click="closeAddModal"
        >
          Cancel
        </button>
        <button
          type="button"
          class="btn-primary"
          :disabled="isSubmitDisabled"
          @click="submitDataset"
        >
          {{ submitButtonLabel }}
        </button>
      </template>
    </BaseModal>

    <BaseModal
      :show="isResumeModalOpen"
      title="Resume interrupted upload"
      width="500px"
      @close="cancelResumeUpload"
    >
      <div class="s3-resume-content">
        <p>
          An interrupted upload was detected for:
          <strong>{{ pendingResumeState?.fileName }}</strong>
        </p>
        <p class="s3-resume-hint">
          To resume from where it stopped, select the exact same file from your computer.
        </p>

        <div class="form-group s3-resume-file">
          <label for="resumeFile">Select file</label>
          <input
            id="resumeFile"
            type="file"
            class="s3-form-input-file"
            @change="handleResumeFileChange"
          />
        </div>

        <p v-if="resumeError" class="s3-resume-error">{{ resumeError }}</p>
      </div>

      <template #footer>
        <button type="button" class="btn-danger" @click="cancelResumeUpload">
          Cancel upload
        </button>
        <button
          type="button"
          class="btn-primary"
          :disabled="!resumeFile"
          @click="confirmResume"
        >
          Resume upload
        </button>
      </template>
    </BaseModal>

    <BaseModal
      :show="isPreviewModalOpen"
      :title="`Dataset preview: ${activeRow?.file_name || ''}`"
      width="900px"
      @close="closePreviewModal"
    >
      <div class="s3-preview-content">
        <p v-if="isPreviewLoading" class="s3-preview-status">Loading preview...</p>
        <p v-else-if="previewError" class="s3-preview-error">{{ previewError }}</p>

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

      <template #footer>
        <button type="button" class="btn-secondary" @click="closePreviewModal">
          Close
        </button>
      </template>
    </BaseModal>

    <DeleteAction
      v-if="showDeleteModal"
      :item="activeRow"
      :delete-service="deleteS3Dataset"
      @success="onDeleteSuccess"
      @close="showDeleteModal = false"
    />
  </div>
</template>
