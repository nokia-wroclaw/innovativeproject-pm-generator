<script setup>
import { ref, reactive, onMounted, onBeforeUnmount } from 'vue';
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
  registerExistingS3Dataset
} from '../services/s3';

const CHUNK_SIZE = 5 * 1024 * 1024;
const STORAGE_KEY = 's3_pending_upload';

const tableRef = ref(null);
const isModalOpen = ref(false);
const isPreparing = ref(false);
const activeRow = ref(null);
const showDeleteComponent = ref(false);
const activeUploads = ref([]);

const uploadMode = ref('upload');

const formData = ref({ file_name: '', s3_key: '', file: null });

const pendingResumeState = ref(null);
const isResumeModalOpen = ref(false);
const resumeFile = ref(null);
const resumeError = ref('');

const tableColumns = [
  { key: 'id', label: 'ID' },
  { key: 'file_name', label: 'Name' },
  { key: 's3_key', label: 'Path' },
  { key: 'status', label: 'Status' },
  { key: 'actions', label: 'Actions' }
];

const rowActions = [
  { id: 'analyze', label: 'Analyze Dataset', class: 'text-blue' },
  { id: 'delete', label: 'Delete', class: 'text-red' }
];

const handleBeforeUnload = (event) => {
  const isUploading = activeUploads.value.some(task => task.status === DatasetStatus.UPLOADING);
  if (isUploading) {
    event.preventDefault();
    event.returnValue = '';
  }
};

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

const openModal = () => { isModalOpen.value = true; };
const closeModal = () => {
  isModalOpen.value = false;
  formData.value = { file_name: '', s3_key: '', file: null };
  uploadMode.value = 'upload';
};

const handleRowAction = ({ type, row }) => {
  activeRow.value = row;
  if (type === 'analyze') console.log('Initiating analysis for:', row.file_name);
  else if (type === 'delete') showDeleteComponent.value = true;
};

const handleFileChange = (event) => {
  const selectedFile = event.target.files[0];
  if (selectedFile) {
    formData.value.file = selectedFile;
    if (!formData.value.file_name) formData.value.file_name = selectedFile.name;
  }
};

const uploadChunk = async (url, chunkData) => {
  const response = await fetch(url, { method: 'PUT', body: chunkData });
  if (!response.ok) throw new Error(`Chunk upload failed with status ${response.status}`);

  const etag = response.headers.get('ETag');
  if (!etag) throw new Error('ETag not found. Check S3 CORS ExposeHeaders configuration.');
  return etag;
};

const processUploadLoop = async (file, uploadState, uploadTask) => {
  isPreparing.value = true;

  try {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);

    for (let partNumber = 1; partNumber <= totalChunks; partNumber++) {
      if (uploadState.uploadedParts.some(p => p.PartNumber === partNumber)) {
        uploadTask.progress = Math.round((uploadState.uploadedParts.length / totalChunks) * 100);
        continue;
      }

      const start = (partNumber - 1) * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const partUrlData = await getPresignedPartUrl(uploadState.datasetId, uploadState.s3UploadId, partNumber);
      const etag = await uploadChunk(partUrlData.url, chunk);

      uploadState.uploadedParts.push({ PartNumber: partNumber, ETag: etag });
      localStorage.setItem(STORAGE_KEY, JSON.stringify(uploadState));

      uploadTask.progress = Math.round((uploadState.uploadedParts.length / totalChunks) * 100);
    }

    await completeMultipartUpload(uploadState.datasetId, uploadState.s3UploadId, uploadState.uploadedParts);
    await updateS3Status(uploadState.datasetId, DatasetStatus.COMPLETED);

    localStorage.removeItem(STORAGE_KEY);
    uploadTask.status = DatasetStatus.COMPLETED;

    setTimeout(() => {
      activeUploads.value = activeUploads.value.filter(t => t.id !== uploadTask.id);
      if (tableRef.value) tableRef.value.refresh();
    }, 1500);

  } catch (error) {
    console.error('Multipart Upload process failed:', error);
    uploadTask.status = DatasetStatus.FAILED;
  } finally {
    isPreparing.value = false;
  }
};

const submitDataset = async () => {
  const { file_name, s3_key, file } = formData.value;

  if (!s3_key) return;

  isPreparing.value = true;

  try {
    if (uploadMode.value === 'register') {
      await registerExistingS3Dataset({ file_name, s3_key });
      closeModal();
      if (tableRef.value) tableRef.value.refresh();
      isPreparing.value = false;
      return;
    }

    if (!file_name || !file) {
      isPreparing.value = false;
      return;
    }

    const dataset = await createS3Dataset({ file_name, s3_key });
    const initResponse = await initiateMultipartUpload(dataset.id);

    const uploadState = {
      datasetId: dataset.id,
      s3UploadId: initResponse.upload_id,
      s3_key: initResponse.s3_key,
      fileName: file.name,
      fileSize: file.size,
      uploadedParts: []
    };

    localStorage.setItem(STORAGE_KEY, JSON.stringify(uploadState));

    const tempId = 'upload_' + Date.now();
    const uploadTask = reactive({
      id: tempId,
      file_name,
      s3_key: initResponse.s3_key,
      status: DatasetStatus.UPLOADING,
      progress: 0
    });

    activeUploads.value.unshift(uploadTask);
    closeModal();
    if (tableRef.value) tableRef.value.refresh();

    await processUploadLoop(file, uploadState, uploadTask);

  } catch (error) {
    console.log(error)
    console.error('Failed to process dataset:', error);
    alert(error.message || 'Error processing dataset');
    isPreparing.value = false;
  }
};

const handleResumeFileChange = (event) => {
  resumeError.value = '';
  resumeFile.value = event.target.files[0];
};

const confirmResume = async () => {
  if (!resumeFile.value || !pendingResumeState.value) return;

  if (
    resumeFile.value.name !== pendingResumeState.value.fileName ||
    resumeFile.value.size !== pendingResumeState.value.fileSize
  ) {
    resumeError.value = "Selected file does not match the interrupted upload. Please select the correct file.";
    return;
  }

  isResumeModalOpen.value = false;

  const tempId = 'upload_resumed_' + Date.now();
  const uploadTask = reactive({
    id: tempId,
    file_name: pendingResumeState.value.fileName,
    s3_key: pendingResumeState.value.s3_key,
    status: DatasetStatus.UPLOADING,
    progress: 0
  });

  activeUploads.value.unshift(uploadTask);
  if (tableRef.value) tableRef.value.refresh();
  await updateS3Status(pendingResumeState.value.datasetId, DatasetStatus.UPLOADING);
  await processUploadLoop(resumeFile.value, pendingResumeState.value, uploadTask);
};

const cancelResumeUpload = async () => {
  if (!pendingResumeState.value) return;

  try {
    await abortMultipartUpload(pendingResumeState.value.datasetId, pendingResumeState.value.s3UploadId);
    await updateS3Status(pendingResumeState.value.datasetId, DatasetStatus.FAILED);
  } catch (e) {
    console.error('Failed to abort upload on backend', e);
  }

  localStorage.removeItem(STORAGE_KEY);
  pendingResumeState.value = null;
  isResumeModalOpen.value = false;
  if (tableRef.value) tableRef.value.refresh();
};

const tableProviderWrapper = async (params) => {
  const serverData = await fetchS3DatasetsPage(params);
  const items = Array.isArray(serverData) ? serverData : (serverData.data || serverData.items || []);

  const filteredItems = items.filter(item =>
    item.status !== DatasetStatus.PENDING
  );

  const mergedItems = [...activeUploads.value, ...filteredItems];

  if (Array.isArray(serverData)) return mergedItems;
  return { ...serverData, data: mergedItems, items: mergedItems };
};
</script>

<template>
  <div class="s3-page-container">
    <div class="page-header">
      <h2>Manage S3 Datasets</h2>
      <button class="primary-button" @click="openModal">Add dataset</button>
    </div>

    <DataTable
      ref="tableRef"
      :columns="tableColumns"
      :provider="tableProviderWrapper"
      :per-page="10"
    >
      <template #cell-status="{ row }">
        <div v-if="row.status === DatasetStatus.UPLOADING" class="table-upload-cell">
          <div class="table-progress-bar">
            <div class="table-progress-fill" :style="{ width: row.progress + '%' }"></div>
          </div>
          <span class="table-progress-text">{{ row.progress }}%</span>
        </div>
        <span v-else-if="row.status" :class="['status-badge', `status-${row.status.toLowerCase()}`]">
          {{ row.status }}
        </span>
        <span v-else class="status-badge status-unknown">No data</span>
      </template>

      <template #cell-actions="{ row }">
        <DynamicActions
          v-if="row.status !== DatasetStatus.UPLOADING"
          :row="row"
          :actions="rowActions"
          @action="handleRowAction"
        />
        <span v-else class="status-waiting">Uploading...</span>
      </template>
    </DataTable>

    <BaseModal :show="isModalOpen" title="Add S3 dataset" width="500px" @close="closeModal">
      <form @submit.prevent="submitDataset" class="dataset-form">

        <div class="form-group">
          <label>Action type:</label>
          <div class="radio-group">
            <label class="radio-label">
              <input type="radio" value="upload" v-model="uploadMode" :disabled="isPreparing" />
              Upload local file
            </label>
            <label class="radio-label">
              <input type="radio" value="register" v-model="uploadMode" :disabled="isPreparing" />
              Register existing S3 file
            </label>
          </div>
        </div>

        <div class="form-group">
          <label for="fileName">File name <span v-if="uploadMode === 'register'">(Optional)</span>:</label>
          <input
            id="fileName"
            v-model="formData.file_name"
            type="text"
            :required="uploadMode === 'upload'"
            class="form-input"
            :disabled="isPreparing"
            :placeholder="uploadMode === 'register' ? 'Leave empty to use S3 key name' : ''"
          />
        </div>
        <div class="form-group">
          <label for="s3Key">Path (S3 Key):</label>
          <input id="s3Key" v-model="formData.s3_key" type="text" required class="form-input" :disabled="isPreparing" placeholder="e.g. data/my-file.csv" />
        </div>

        <div class="form-group file-upload-section" v-if="uploadMode === 'upload'">
          <label for="fileUpload">Select file:</label>
          <input id="fileUpload" type="file" @change="handleFileChange" required class="form-input-file" :disabled="isPreparing" />
        </div>

      </form>
      <template #footer>
        <button type="button" class="secondary-button" @click="closeModal" :disabled="isPreparing">Cancel</button>
        <button
          type="button"
          class="primary-button"
          @click="submitDataset"
          :disabled="isPreparing || !formData.s3_key || (uploadMode === 'upload' && (!formData.file_name || !formData.file))"
        >
          <span v-if="isPreparing">Processing...</span>
          <span v-else-if="uploadMode === 'register'">Register File</span>
          <span v-else>Start Upload</span>
        </button>
      </template>
    </BaseModal>

    <BaseModal :show="isResumeModalOpen" title="Resume Interrupted Upload" width="500px" @close="cancelResumeUpload">
      <div class="resume-content">
        <p>An interrupted upload was detected for the file: <strong>{{ pendingResumeState?.fileName }}</strong></p>
        <p style="font-size: 0.9rem; color: #4b5563;">To resume from where it stopped, please select the exact same file from your computer.</p>

        <div class="form-group mt-3">
          <input type="file" @change="handleResumeFileChange" class="form-input-file" />
        </div>

        <p v-if="resumeError" class="text-red mt-2" style="font-size: 0.85rem;">{{ resumeError }}</p>
      </div>

      <template #footer>
        <button type="button" class="danger-button" @click="cancelResumeUpload">Cancel Upload</button>
        <button type="button" class="primary-button" @click="confirmResume" :disabled="!resumeFile">
          Resume Upload
        </button>
      </template>
    </BaseModal>

    <DeleteAction
      v-if="showDeleteComponent"
      :item="activeRow"
      :delete-service="deleteS3Dataset"
      @success="() => { showDeleteComponent = false; tableRef.refresh(); }"
      @close="showDeleteComponent = false"
    />
  </div>
</template>

<style scoped>
.s3-page-container { padding: 20px; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
.page-header h2 { margin: 0; color: #111827; font-size: 1.5rem; }
.dataset-form { display: flex; flex-direction: column; gap: 16px; }
.form-group { display: flex; flex-direction: column; gap: 4px; }
.form-group label { font-size: 0.875rem; font-weight: 500; color: #374151; }
.form-input { padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.95rem; outline: none; transition: border-color 0.2s; }
.form-input:focus { border-color: #3b82f6; }
.file-upload-section { margin-top: 10px; }

.radio-group { display: flex; gap: 16px; align-items: center; padding: 6px 0; }
.radio-label { display: flex; align-items: center; gap: 8px; font-size: 0.9rem; cursor: pointer; color: #4b5563; font-weight: normal !important; }
.radio-label input[type="radio"] { cursor: pointer; }

.table-upload-cell { display: flex; align-items: center; gap: 10px; min-width: 140px; }
.table-progress-bar { flex-grow: 1; height: 8px; background-color: #e5e7eb; border-radius: 10px; overflow: hidden; }
.table-progress-fill { height: 100%; background-color: #3b82f6; transition: width 0.3s ease; }
.table-progress-text { font-size: 0.75rem; font-weight: 600; color: #3b82f6; }
.status-badge { padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
.status-completed { background-color: #dcfce7; color: #166534; }
.status-uploading { background-color: #e0e7ff; color: #4338ca; }
.status-failed { background-color: #fee2e2; color: #991b1b; }
.status-pending { background-color: #fef08a; color: #854d0e; }
.status-waiting { color: #9ca3af; font-size: 0.85rem; font-style: italic; }
.status-unknown { background-color: #f3f4f6; color: #4b5563; }

.primary-button { background-color: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; }
.primary-button:hover:not(:disabled) { background-color: #2563eb; }
.primary-button:disabled { opacity: 0.5; cursor: not-allowed; }

.secondary-button { background-color: white; color: #374151; border: 1px solid #d1d5db; padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; }
.secondary-button:hover:not(:disabled) { background-color: #f9fafb; color: #111827; }

.danger-button { background-color: #ef4444; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: 600; cursor: pointer; transition: background-color 0.2s; }
.danger-button:hover { background-color: #dc2626; }

.text-blue { color: #3b82f6; }
.text-red { color: #ef4444; }
.mt-2 { margin-top: 8px; }
.mt-3 { margin-top: 12px; }
.resume-content p { margin: 0 0 10px 0; }
</style>