<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';

const props = defineProps({
  row: {
    type: Object,
    required: true
  },
  // [{ id: 'analyze', label: 'Analyze Dataset', class: 'text-blue' }]
  actions: {
    type: Array,
    required: true
  }
});

const emit = defineEmits(['action']);

const isOpen = ref(false);
const dropdownRef = ref(null);

const toggleDropdown = () => {
  isOpen.value = !isOpen.value;
};

const handleAction = (action) => {
  emit('action', { type: action.id, row: props.row });
  isOpen.value = false;
};

const handleClickOutside = (event) => {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target)) {
    isOpen.value = false;
  }
};

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside);
});
</script>

<template>
  <div class="dropdown-container" ref="dropdownRef">
    <button class="trigger-btn" @click="toggleDropdown" :class="{ 'active': isOpen }">
      Actions
    </button>

    <transition name="dropdown-fade">
      <div v-if="isOpen" class="dropdown-menu">
        <button
          v-for="action in actions"
          :key="action.id"
          :class="['dropdown-item', action.class]"
          @click="handleAction(action)"
          :title="action.tooltip || action.label"
        >
          {{ action.label }}
        </button>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.dropdown-container {
  position: relative;
  display: inline-block;
}

.trigger-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  background-color: white;
  border: 1px solid #d1d5db;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s;
}

.trigger-btn:hover, .trigger-btn.active {
  background-color: #f3f4f6;
  border-color: #9ca3af;
}

.chevron {
  width: 16px;
  height: 16px;
  transition: transform 0.2s ease;
}

.chevron-up {
  transform: rotate(180deg);
}

.dropdown-menu {
  position: absolute;
  right: 0;
  top: calc(100% + 4px);
  width: max-content;
  min-width: 140px;
  background-color: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  padding: 4px 0;
  z-index: 50;
}

.dropdown-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 8px 16px;
  font-size: 0.875rem;
  color: #374151;
  cursor: pointer;
  transition: background-color 0.15s;
}

.dropdown-item:hover {
  background-color: #f3f4f6;
}

.dropdown-fade-leave-active {
  transition: opacity 0.15s ease, transform 0.15s ease;
}

.dropdown-fade-enter-from,
.dropdown-fade-leave-to {
  opacity: 0;
  transform: translateY(-5px);
}

.text-blue { color: #2563eb; }
.text-red { color: #dc2626; }
.text-red:hover { background-color: #fee2e2; }
.text-green { color: #16a34a; }
.text-gray { color: #4b5563; }
</style>