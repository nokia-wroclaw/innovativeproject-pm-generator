<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';

const props = defineProps({
  row: {
    type: Object,
    required: true,
  },
  actions: {
    type: Array,
    required: true,
  },
});

const emit = defineEmits(['action']);

const isOpen = ref(false);
const triggerRef = ref(null);
const menuStyle = ref({});

const updateMenuPosition = () => {
  const trigger = triggerRef.value;
  if (!trigger) return;

  const rect = trigger.getBoundingClientRect();
  menuStyle.value = {
    position: 'fixed',
    top: `${rect.bottom + 4}px`,
    left: `${rect.right}px`,
    transform: 'translateX(-100%)',
    zIndex: '999',
  };
};

const openDropdown = async () => {
  isOpen.value = true;
  await nextTick();
  updateMenuPosition();
};

const closeDropdown = () => {
  isOpen.value = false;
};

const toggleDropdown = async () => {
  if (isOpen.value) {
    closeDropdown();
  } else {
    await openDropdown();
  }
};

const handleAction = (action) => {
  emit('action', { type: action.id, row: props.row });
  closeDropdown();
};

const handleClickOutside = (event) => {
  const trigger = triggerRef.value;
  const menu = document.getElementById(`actions-menu-${props.row.id}`);

  if (trigger?.contains(event.target) || menu?.contains(event.target)) {
    return;
  }

  closeDropdown();
};

const handleScrollOrResize = () => {
  if (isOpen.value) updateMenuPosition();
};

watch(isOpen, (open) => {
  if (open) {
    window.addEventListener('scroll', handleScrollOrResize, true);
    window.addEventListener('resize', handleScrollOrResize);
  } else {
    window.removeEventListener('scroll', handleScrollOrResize, true);
    window.removeEventListener('resize', handleScrollOrResize);
  }
});

onMounted(() => {
  document.addEventListener('click', handleClickOutside);
});

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside);
  window.removeEventListener('scroll', handleScrollOrResize, true);
  window.removeEventListener('resize', handleScrollOrResize);
});
</script>

<template>
  <div class="dropdown-container">
    <button
      ref="triggerRef"
      type="button"
      class="trigger-btn"
      :class="{ active: isOpen }"
      @click.stop="toggleDropdown"
    >
      Actions
    </button>

    <Teleport to="body">
      <transition name="dropdown-fade">
        <div
          v-if="isOpen"
          :id="`actions-menu-${row.id}`"
          class="dropdown-menu"
          :style="menuStyle"
          @click.stop
        >
          <button
            v-for="action in actions"
            :key="action.id"
            type="button"
            :class="['dropdown-item', action.class]"
            :title="action.tooltip || action.label"
            @click="handleAction(action)"
          >
            {{ action.label }}
          </button>
        </div>
      </transition>
    </Teleport>
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

.trigger-btn:hover,
.trigger-btn.active {
  background-color: #f3f4f6;
  border-color: #9ca3af;
}

.dropdown-menu {
  width: max-content;
  min-width: 140px;
  background-color: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
  padding: 4px 0;
}

.dropdown-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 8px 16px;
  font-size: 0.875rem;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
}

.dropdown-fade-enter-active,
.dropdown-fade-leave-active {
  transition: opacity 0.15s ease;
}

.dropdown-fade-enter-from,
.dropdown-fade-leave-to {
  opacity: 0;
}
</style>
