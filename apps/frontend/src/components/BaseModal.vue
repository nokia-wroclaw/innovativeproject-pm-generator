<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="show" class="modal-overlay" @click.self="close" @wheel.self.prevent>
        <Transition name="scale">
          <div v-if="show" class="modal-container" :style="{ maxWidth: width }">
            <div class="modal-header">
              <slot name="header">
                <h3 class="modal-title">{{ title }}</h3>
              </slot>
              <button class="close-button" @click="close">
                <X :size="24" />
              </button>
            </div>

            <div class="modal-body">
              <slot></slot>
            </div>

            <div v-if="$slots.footer" class="modal-footer">
              <slot name="footer"></slot>
            </div>
          </div>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { onMounted, onUnmounted, watch } from 'vue';
import { X } from 'lucide-vue-next';

let bodyScrollLockCount = 0;

function lockBodyScroll() {
  if (bodyScrollLockCount === 0) document.body.classList.add('modal-open');
  bodyScrollLockCount += 1;
}

function unlockBodyScroll() {
  if (bodyScrollLockCount <= 0) return;
  bodyScrollLockCount -= 1;
  if (bodyScrollLockCount === 0) document.body.classList.remove('modal-open');
}

const props = defineProps({
  show: {
    type: Boolean,
    default: false
  },
  title: {
    type: String,
    default: ''
  },
  width: {
    type: String,
    default: '500px'
  }
});

const emit = defineEmits(['close']);

const close = () => {
  emit('close');
};

const handleEsc = (e) => {
  if (e.key === 'Escape' && props.show) close();
};

watch(
  () => props.show,
  (isShown) => {
    if (isShown) lockBodyScroll();
    else unlockBodyScroll();
  },
  { immediate: true },
);

onMounted(() => window.addEventListener('keydown', handleEsc));
onUnmounted(() => {
  if (props.show) unlockBodyScroll();
  window.removeEventListener('keydown', handleEsc);
});
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(2px);
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 1.5rem;
  z-index: 1000;
  overscroll-behavior: contain;
}

.modal-container {
  background: white;
  width: 90%;
  max-height: min(90dvh, calc(100vh - 3rem));
  border-radius: 12px;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  padding: 16px 20px;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #f9fafb;
}

.modal-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #111827;
}

.close-button {
  background: none;
  border: none;
  color: #9ca3af;
  cursor: pointer;
  padding: 4px;
  border-radius: 6px;
  display: flex;
  transition: all 0.2s;
}

.close-button:hover {
  background-color: #f3f4f6;
  color: #4b5563;
}

.modal-body {
  padding: 20px;
  font-size: 0.95rem;
  color: #374151;
  line-height: 1.5;
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.modal-footer {
  padding: 12px 20px;
  background-color: #f9fafb;
  border-top: 1px solid #e5e7eb;
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

.fade-enter-active, .fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.scale-enter-active, .scale-leave-active {
  transition: transform 0.2s ease, opacity 0.2s ease;
}
.scale-enter-from, .scale-leave-to {
  transform: scale(0.95);
  opacity: 0;
}
</style>
