<script setup>
import { ref } from 'vue';
import BaseModal from './BaseModal.vue';

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
  deleteService: {
    type: Function,
    required: true,
  },
  title: {
    type: String,
    default: 'Confirm Delete',
  },
  askStorageScope: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(['success', 'error', 'close']);

const isModalOpen = ref(true);
const isDeleting = ref(false);
const deleteFromS3 = ref(false);

const handleClose = () => {
  if (isDeleting.value) return;
  isModalOpen.value = false;
  emit('close');
};

const handleConfirm = async () => {
  isDeleting.value = true;
  try {
    if (props.askStorageScope) {
      await props.deleteService(props.item.id, { deleteFromS3: deleteFromS3.value });
    } else {
      await props.deleteService(props.item.id);
    }
    emit('success', props.item);
    isModalOpen.value = false;
  } catch (error) {
    console.error('Delete failed:', error);
    emit('error', error);
  } finally {
    isDeleting.value = false;
  }
};
</script>

<template>
  <BaseModal
    :show="isModalOpen"
    :title="title"
    :width="askStorageScope ? '480px' : '400px'"
    @close="handleClose"
  >
    <p class="delete-warning">
      Are you sure you want to delete <strong>{{ item.file_name || item.name }}</strong>?
      This action cannot be undone.
    </p>

    <fieldset v-if="askStorageScope" class="delete-scope">
      <legend class="delete-scope-legend">What should be removed?</legend>
      <label class="delete-scope-option">
        <input v-model="deleteFromS3" type="radio" :value="false" :disabled="isDeleting" />
        <span>
          <strong>Database only</strong>
          <span class="delete-scope-hint">
            Remove the dataset record locally. The file stays in S3.
          </span>
        </span>
      </label>
      <label class="delete-scope-option">
        <input v-model="deleteFromS3" type="radio" :value="true" :disabled="isDeleting" />
        <span>
          <strong>Database and S3</strong>
          <span class="delete-scope-hint">
            Remove the record and delete the object at
            <code>{{ item.s3_key }}</code>.
          </span>
        </span>
      </label>
    </fieldset>

    <template #footer>
      <button
        type="button"
        class="secondary-button"
        :disabled="isDeleting"
        @click="handleClose"
      >
        Cancel
      </button>
      <button
        type="button"
        class="danger-button"
        :disabled="isDeleting"
        @click="handleConfirm"
      >
        {{ isDeleting ? 'Deleting...' : 'Yes, delete' }}
      </button>
    </template>
  </BaseModal>
</template>

<style scoped>
.delete-warning {
  margin: 0 0 16px;
  color: #4b5563;
  line-height: 1.5;
}

.delete-scope {
  margin: 0;
  padding: 0;
  border: none;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.delete-scope-legend {
  font-size: 0.9rem;
  font-weight: 600;
  color: #374151;
  margin-bottom: 4px;
}

.delete-scope-option {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  cursor: pointer;
}

.delete-scope-option:has(input:checked) {
  border-color: #93c5fd;
  background-color: #eff6ff;
}

.delete-scope-option input {
  margin-top: 3px;
}

.delete-scope-option span {
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: #111827;
}

.delete-scope-hint {
  font-size: 0.85rem;
  color: #6b7280;
  font-weight: 400;
  line-height: 1.4;
}

.delete-scope-hint code {
  font-size: 0.8rem;
  word-break: break-all;
}

.danger-button {
  background-color: #ef4444;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
}

.danger-button:hover:not(:disabled) {
  background-color: #dc2626;
}

.danger-button:disabled {
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
}
</style>
