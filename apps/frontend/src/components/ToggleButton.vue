<template>
  <button @click="$emit('toggle')" class="toggle-btn" aria-label="Toggle Sidebar">
    <component :is="isCollapsed ? ChevronRight : ChevronLeft" :size="18" />
  </button>
</template>

<script setup>
import { ChevronRight, ChevronLeft } from 'lucide-vue-next';

defineProps({
  isCollapsed: {
    type: Boolean,
    required: true
  }
});

defineEmits(['toggle']);
</script>

<style scoped>
.toggle-btn {
  background: transparent;
  border: 1px solid #e5e7eb;
  color: #4b5563;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  transition: all 0.2s ease;
}

.toggle-btn:hover {
  background-color: #f4f8ff;
  border-color: #005AFF;
  color: #005AFF;
}
</style>