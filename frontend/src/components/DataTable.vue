<template>
  <div class="table-container">
    <div class="table-wrapper">
      <table class="custom-table">
        <thead>
          <tr>
            <th v-for="col in columns" :key="col.key" class="table-header">
              {{ col.label }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in paginatedData" :key="index" class="table-row">
            <td v-for="col in columns" :key="col.key" class="table-cell">
              <slot :name="`cell-${col.key}`" :row="row">
                {{ row[col.key] }}
              </slot>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="table-footer">
      <div class="footer-left">
        <button
          @click="loadData"
          class="refresh-button"
          :disabled="isLoading"
          title="Refresh Data"
        >
          <RefreshCw :class="{ 'spinning': isLoading }" :size="16" />
        </button>
      </div>

      <div class="pagination" v-if="totalPages > 1">
        <button
          class="pag-btn"
          :disabled="currentPage === 1 || isLoading"
          @click="currentPage--"
        >
          <ChevronLeft :size="18" />
        </button>

        <span class="page-info">
          Page {{ currentPage }} of {{ totalPages }}
        </span>

        <button
          class="pag-btn"
          :disabled="currentPage === totalPages || isLoading"
          @click="currentPage++"
        >
          <ChevronRight :size="18" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import { RefreshCw, ChevronLeft, ChevronRight } from 'lucide-vue-next';

const props = defineProps({
  columns: {
    type: Array,
    required: true,
  },
  provider: {
    type: Function,
    required: true,
  },
  perPage: {
    type: Number,
    default: 10
  }
});

const data = ref([]);
const isLoading = ref(true);
const currentPage = ref(1);

const totalPages = computed(() => Math.ceil(data.value.length / props.perPage));

const paginatedData = computed(() => {
  const start = (currentPage.value - 1) * props.perPage;
  const end = start + props.perPage;
  return data.value.slice(start, end);
});

const loadData = async () => {
  isLoading.value = true;
  try {
    data.value = await props.provider();
    currentPage.value = 1;
  } catch (error) {
    data.value = [];
  } finally {
    isLoading.value = false;
  }
};

onMounted(() => {
  loadData();
});
</script>

<style scoped>
.table-container {
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
}

.table-wrapper {
  overflow-x: auto;
}

.custom-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}

.table-header {
  padding: 14px 20px;
  background-color: #f9fafb;
  color: #4b5563;
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid #e5e7eb;
}

.table-cell {
  padding: 14px 20px;
  color: #1f2937;
  font-size: 0.9rem;
  border-bottom: 1px solid #e5e7eb;
}

.table-row:last-child .table-cell {
  border-bottom: none;
}

.table-footer {
  padding: 12px 16px;
  background-color: #f9fafb;
  border-top: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.refresh-button {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background-color: transparent;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  color: #4b5563;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}

.refresh-button:hover:not(:disabled) {
  background-color: #f9fafb;
  border-color: #9ca3af;
}

.pagination {
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-info {
  font-size: 0.85rem;
  color: #6b7280;
}

.page-info strong {
  color: #111827;
}

.pag-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 4px;
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  color: #4b5563;
  cursor: pointer;
  transition: all 0.2s;
}

.pag-btn:hover:not(:disabled) {
  background-color: #f3f4f6;
  color: #111827;
}

.pag-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.loading-state,
.empty-state {
  padding: 40px;
  text-align: center;
  color: #6b7280;
  font-style: italic;
}
</style>