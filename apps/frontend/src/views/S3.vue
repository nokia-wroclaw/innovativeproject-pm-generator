<script setup>
import '../assets/S3.css';
import { ref, reactive, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import BaseModal from '../components/BaseModal.vue';
import S3StorageTabBody from '../components/S3StorageTabBody.vue';
import DeleteAction from '../components/DeleteAction.vue';
import DatasetPreviewContent from '../features/storage/components/DatasetPreviewContent.vue';
import { Tabs } from '@/components/ui';
import {
  fetchS3DatasetsPage,
  createS3Dataset,
  deleteS3Dataset,
  updateS3Status,
  DatasetStatus,
  DatasetType,
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

const TAB_RAW = 'raw';
const TAB_KPI_DEFINITIONS = 'kpi_definitions';
const TAB_SIMPLE_REPORTS = 'simple_reports';
const TAB_PREPROCESSED = 'preprocessed';
const TAB_GENERATED = 'generated';

const TAB_TO_DATASET_TYPE = {
  [TAB_RAW]: DatasetType.RAW,
  [TAB_KPI_DEFINITIONS]: DatasetType.KPI_DEFINITIONS,
  [TAB_SIMPLE_REPORTS]: DatasetType.SIMPLE_REPORTS,
  [TAB_PREPROCESSED]: DatasetType.PREPROCESSED,
  [TAB_GENERATED]: DatasetType.GENERATED,
};

const ADMIN_TABS = [
  { value: TAB_RAW, label: 'Raw' },
  { value: TAB_KPI_DEFINITIONS, label: 'KPI Definitions' },
  { value: TAB_SIMPLE_REPORTS, label: 'Simple Reports' },
  { value: TAB_PREPROCESSED, label: 'Preprocessed' },
  { value: TAB_GENERATED, label: 'Generated' },
];

const USER_TABS = [{ value: TAB_GENERATED, label: 'Generated' }];

const TAB_DESCRIPTIONS = {
  [TAB_RAW]: 'Upload and manage raw datasets before preprocessing.',
  [TAB_KPI_DEFINITIONS]: 'Upload and manage KPI definition parquet files.',
  [TAB_SIMPLE_REPORTS]: 'Upload and manage simple report parquet files.',
  [TAB_PREPROCESSED]: 'Datasets produced by preprocessing pipelines.',
  [TAB_GENERATED]: 'Synthetic event logs available for download and preview.',
};

const TABLE_COLUMNS = [
  { key: 'id', label: 'ID' },
  { key: 'file_name', label: 'Name' },
  { key: 's3_key', label: 'Path' },
  { key: 'status', label: 'Status' },
  { key: 'actions', label: 'Actions' },
];

const route = useRoute();
const router = useRouter();

const storageTabItems = computed(() => (isAdmin.value ? ADMIN_TABS : USER_TABS));

const defaultTab = computed(() => storageTabItems.value[0]?.value ?? TAB_GENERATED);

const activeTab = ref(
  storageTabItems.value.some((tab) => tab.value === route.query.tab)
    ? route.query.tab
    : defaultTab.value,
);

const canAddDataset = computed(
  () =>
    isAdmin.value &&
    (activeTab.value === TAB_RAW ||
      activeTab.value === TAB_KPI_DEFINITIONS ||
      activeTab.value === TAB_SIMPLE_REPORTS),
);

const rowActions = computed(() => {
  const actions = [
    { id: 'analyze', label: 'Preview dataset', class: 's3-action-analyze' },
  ];
  if (isAdmin.value) {
    actions.push({ id: 'delete', label: 'Delete', class: 's3-action-delete' });
  }
  return actions;
});

const rawTabBodyRef = ref(null);
const kpiDefinitionsTabBodyRef = ref(null);
const simpleReportsTabBodyRef = ref(null);
const preprocessedTabBodyRef = ref(null);
const generatedTabBodyRef = ref(null);

const tabBodyRefs = {
  [TAB_RAW]: rawTabBodyRef,
  [TAB_KPI_DEFINITIONS]: kpiDefinitionsTabBodyRef,
  [TAB_SIMPLE_REPORTS]: simpleReportsTabBodyRef,
  [TAB_PREPROCESSED]: preprocessedTabBodyRef,
  [TAB_GENERATED]: generatedTabBodyRef,
};

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
  activeUploads.value.some((task) => task.status === DatasetStatus.UPLOADING),
);

watch(storageTabItems, (tabs) => {
  if (!tabs.some((tab) => tab.value === activeTab.value)) {
    activeTab.value = tabs[0]?.value ?? TAB_GENERATED;
  }
});

watch(activeTab, (tab) => {
  const allowed = new Set(storageTabItems.value.map((item) => item.value));
  if (!allowed.has(tab)) {
    activeTab.value = defaultTab.value;
    return;
  }
  const nextQuery = { ...route.query, tab };
  router.replace({ query: nextQuery });
  refreshActiveTable();
});

watch(
  () => route.query.tab,
  (tab) => {
    if (typeof tab === 'string' && storageTabItems.value.some((item) => item.value === tab)) {
      activeTab.value = tab;
    }
  },
);

function refreshActiveTable() {
  tabBodyRefs[activeTab.value]?.value?.refresh();
}

function resetForm() {
  formData.value = { file_name: '', s3_key: '', file: null };
  uploadMode.value = 'upload';
}

function openAddModal() {
  if (!canAddDataset.value) return;
  isAddModalOpen.value = true;
}

function closeAddModal() {
  isAddModalOpen.value = false;
  resetForm();
}

function createUploadTask({ id, file_name, s3_key, type }) {
  return reactive({
    id,
    file_name,
    s3_key,
    status: DatasetStatus.UPLOADING,
    progress: 0,
    type,
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

function openDatasetDetail(row) {
  if (!row?.id || row.status !== DatasetStatus.COMPLETED) return;
  router.push({
    path: `/s3/datasets/${row.id}`,
    query: { tab: activeTab.value },
  });
}

function openFullViewFromPreview() {
  if (!activeRow.value?.id) return;
  const row = activeRow.value;
  closePreviewModal();
  openDatasetDetail(row);
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
        (part) => part.PartNumber === partNumber,
      );

      if (alreadyUploaded) {
        uploadTask.progress = Math.round(
          (uploadState.uploadedParts.length / totalChunks) * 100,
        );
        continue;
      }

      const start = (partNumber - 1) * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const { url } = await getPresignedPartUrl(
        uploadState.datasetId,
        uploadState.s3UploadId,
        partNumber,
      );
      const etag = await uploadChunk(url, chunk);

      uploadState.uploadedParts.push({ PartNumber: partNumber, ETag: etag });
      persistUploadState(uploadState);

      uploadTask.progress = Math.round(
        (uploadState.uploadedParts.length / totalChunks) * 100,
      );
    }

    await completeMultipartUpload(
      uploadState.datasetId,
      uploadState.s3UploadId,
      uploadState.uploadedParts,
    );
    await updateS3Status(uploadState.datasetId, DatasetStatus.COMPLETED);

    clearPersistedUpload();
    uploadTask.status = DatasetStatus.COMPLETED;

    setTimeout(() => {
      activeUploads.value = activeUploads.value.filter((task) => task.id !== uploadTask.id);
      refreshActiveTable();
    }, 1500);
  } catch (error) {
    console.error('Multipart upload failed:', error);
    try {
      await updateS3Status(uploadState.datasetId, DatasetStatus.FAILED);
    } catch (statusError) {
      console.error('Failed to mark dataset as FAILED:', statusError);
    }
    uploadTask.status = DatasetStatus.FAILED;
  }
}

async function startMultipartUpload(file, fileName, s3Key, type) {
  const dataset = await createS3Dataset({ file_name: fileName, s3_key: s3Key, type });
  const initResponse = await initiateMultipartUpload(dataset.id);

  const uploadState = {
    datasetId: dataset.id,
    s3UploadId: initResponse.upload_id,
    s3_key: initResponse.s3_key,
    fileName: file.name,
    fileSize: file.size,
    uploadedParts: [],
    type,
  };

  persistUploadState(uploadState);

  const uploadTask = createUploadTask({
    id: dataset.id,
    file_name: fileName,
    s3_key: initResponse.s3_key,
    type,
  });

  activeUploads.value.unshift(uploadTask);
  closeAddModal();
  refreshActiveTable();

  await processUploadLoop(file, uploadState, uploadTask);
}

async function submitDataset() {
  if (!canAddDataset.value) return;

  const { file_name, s3_key, file } = formData.value;
  if (!s3_key) return;

  isPreparing.value = true;

  try {
    const type = TAB_TO_DATASET_TYPE[activeTab.value];
    if (isRegisterMode.value) {
      await registerExistingS3Dataset({ file_name, s3_key, type });
      closeAddModal();
      refreshActiveTable();
      return;
    }

    if (!file_name || !file) return;

    await startMultipartUpload(file, file_name, s3_key, type);
  } catch (error) {
    console.error('Failed to process dataset:', error);
    alert(error.message || 'Error processing dataset');
  } finally {
    isPreparing.value = false;
  }
}

async function confirmResume() {
  if (!isAdmin.value || !resumeFile.value || !pendingResumeState.value) return;

  const { fileName, fileSize, datasetId, s3_key, type } = pendingResumeState.value;

  if (resumeFile.value.name !== fileName || resumeFile.value.size !== fileSize) {
    resumeError.value =
      'Selected file does not match the interrupted upload. Please select the correct file.';
    return;
  }

  isResumeModalOpen.value = false;

  const uploadTask = createUploadTask({
    id: datasetId,
    file_name: fileName,
    s3_key,
    type: type || DatasetType.RAW,
  });

  activeUploads.value.unshift(uploadTask);
  refreshActiveTable();

  await updateS3Status(datasetId, DatasetStatus.UPLOADING);
  await processUploadLoop(resumeFile.value, pendingResumeState.value, uploadTask);
}

async function cancelResumeUpload() {
  if (!isAdmin.value || !pendingResumeState.value) return;

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
  refreshActiveTable();
}

function onDeleteSuccess(item) {
  showDeleteModal.value = false;
  if (item?.id != null) {
    activeUploads.value = activeUploads.value.filter((task) => task.id !== item.id);
  }
  refreshActiveTable();
}

function makeTableProvider(tabKey) {
  const datasetType = TAB_TO_DATASET_TYPE[tabKey];
  return async (params) => {
    const serverData = await fetchS3DatasetsPage({ ...params, type: datasetType });
    const items = Array.isArray(serverData)
      ? serverData
      : serverData.data || serverData.items || [];

    const filteredItems = items.filter((item) => item.status !== DatasetStatus.PENDING);
    const uploadsForTab = isAdmin.value
      ? activeUploads.value.filter((task) => task.type === datasetType)
      : [];
    const mergedItems = [...uploadsForTab, ...filteredItems];

    if (Array.isArray(serverData)) return mergedItems;
    return { ...serverData, data: mergedItems, items: mergedItems };
  };
}

const tableProviders = {
  [TAB_RAW]: makeTableProvider(TAB_RAW),
  [TAB_KPI_DEFINITIONS]: makeTableProvider(TAB_KPI_DEFINITIONS),
  [TAB_SIMPLE_REPORTS]: makeTableProvider(TAB_SIMPLE_REPORTS),
  [TAB_PREPROCESSED]: makeTableProvider(TAB_PREPROCESSED),
  [TAB_GENERATED]: makeTableProvider(TAB_GENERATED),
};

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload);

  if (isAdmin.value) {
    const savedState = localStorage.getItem(STORAGE_KEY);
    if (savedState) {
      pendingResumeState.value = JSON.parse(savedState);
      isResumeModalOpen.value = true;
    }
  }
});

onBeforeUnmount(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload);
});
</script>

<template>
  <div class="s3-page-wrapper">
    <Tabs
      v-model="activeTab"
      class="s3-tabs"
      :items="storageTabItems"
    >
      <template #raw>
        <S3StorageTabBody
          ref="rawTabBodyRef"
          :description="TAB_DESCRIPTIONS[TAB_RAW]"
          :show-add-button="canAddDataset"
          :columns="TABLE_COLUMNS"
          :provider="tableProviders[TAB_RAW]"
          :row-actions="rowActions"
          :uploading-status="DatasetStatus.UPLOADING"
          :completed-status="DatasetStatus.COMPLETED"
          :failed-status="DatasetStatus.FAILED"
          :allow-failed-delete="isAdmin"
          @add-dataset="openAddModal"
          @row-action="handleRowAction"
          @open-dataset="openDatasetDetail"
        />
      </template>

      <template #kpi_definitions>
        <S3StorageTabBody
          ref="kpiDefinitionsTabBodyRef"
          :description="TAB_DESCRIPTIONS[TAB_KPI_DEFINITIONS]"
          :show-add-button="canAddDataset"
          :columns="TABLE_COLUMNS"
          :provider="tableProviders[TAB_KPI_DEFINITIONS]"
          :row-actions="rowActions"
          :uploading-status="DatasetStatus.UPLOADING"
          :completed-status="DatasetStatus.COMPLETED"
          @add-dataset="openAddModal"
          @row-action="handleRowAction"
        />
      </template>

      <template #simple_reports>
        <S3StorageTabBody
          ref="simpleReportsTabBodyRef"
          :description="TAB_DESCRIPTIONS[TAB_SIMPLE_REPORTS]"
          :show-add-button="canAddDataset"
          :columns="TABLE_COLUMNS"
          :provider="tableProviders[TAB_SIMPLE_REPORTS]"
          :row-actions="rowActions"
          :uploading-status="DatasetStatus.UPLOADING"
          :completed-status="DatasetStatus.COMPLETED"
          @add-dataset="openAddModal"
          @row-action="handleRowAction"
        />
      </template>

      <template #preprocessed>
        <S3StorageTabBody
          ref="preprocessedTabBodyRef"
          :description="TAB_DESCRIPTIONS[TAB_PREPROCESSED]"
          :show-add-button="false"
          :columns="TABLE_COLUMNS"
          :provider="tableProviders[TAB_PREPROCESSED]"
          :row-actions="rowActions"
          :uploading-status="DatasetStatus.UPLOADING"
          :completed-status="DatasetStatus.COMPLETED"
          :failed-status="DatasetStatus.FAILED"
          :allow-failed-delete="isAdmin"
          @row-action="handleRowAction"
          @open-dataset="openDatasetDetail"
        />
      </template>

      <template #generated>
        <S3StorageTabBody
          ref="generatedTabBodyRef"
          :description="TAB_DESCRIPTIONS[TAB_GENERATED]"
          :show-add-button="false"
          :columns="TABLE_COLUMNS"
          :provider="tableProviders[TAB_GENERATED]"
          :row-actions="rowActions"
          :uploading-status="DatasetStatus.UPLOADING"
          :completed-status="DatasetStatus.COMPLETED"
          :failed-status="DatasetStatus.FAILED"
          :allow-failed-delete="isAdmin"
          @row-action="handleRowAction"
          @open-dataset="openDatasetDetail"
        />
      </template>
    </Tabs>

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
      v-if="isAdmin"
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
      <DatasetPreviewContent
        :loading="isPreviewLoading"
        :error="previewError"
        :preview-data="previewData"
      />

      <template #footer>
        <button type="button" class="btn-secondary" @click="closePreviewModal">
          Close
        </button>
        <button
          v-if="previewData && !previewError"
          type="button"
          class="btn-primary"
          @click="openFullViewFromPreview"
        >
          Full view
        </button>
      </template>
    </BaseModal>

    <DeleteAction
      v-if="showDeleteModal && isAdmin"
      :item="activeRow"
      :delete-service="deleteS3Dataset"
      :ask-storage-scope="activeRow?.status === DatasetStatus.COMPLETED"
      @success="onDeleteSuccess"
      @close="showDeleteModal = false"
    />
  </div>
</template>
