<script setup>
import { ref } from 'vue';
import BaseModal from './BaseModal.vue';

const props = defineProps({
  item: {
    type: Object,
    required: true
  },
  deleteService: {
    type: Function,
    required: true
  },
  title: {
    type: String,
    default: 'Confirm Delete'
  }
});

const emit = defineEmits(['success', 'error', 'close']);

const isModalOpen = ref(true);
const isDeleting = ref(false);

const handleClose = () => {
  if (isDeleting.value) return;
  isModalOpen.value = false;
  emit('close');
};

const handleConfirm = async () => {
  isDeleting.value = true;
  try {
    await props.deleteService(props.item.id);
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
    width="400px"
    @close="handleClose"
  >
    <p class="delete-warning">
      Are you sure you want to delete <strong>{{ item.file_name }}</strong>? This action cannot be undone.
    </p>

    <template #footer>
      <button
        type="button"
        class="secondary-button"
        @click="handleClose"
        :disabled="isDeleting"
      >
        Cancel
      </button>
      <button
        type="button"
        class="danger-button"
        @click="handleConfirm"
        :disabled="isDeleting"
      >
        {{ isDeleting ? 'Deleting...' : 'Yes, delete' }}
      </button>
    </template>
  </BaseModal>
</template>

<style scoped>
.delete-warning {
  margin: 0;
  color: #4b5563;
  line-height: 1.5;
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