<template>
  <div class="app-wrapper">
    <SideBar />

    <main class="content">
      <header class="content-header">
        <div>
          <h1>Aplikacja jest robiona tu</h1>
          <p>test.</p>
        </div>

        <button @click="isModalOpen = true" class="btn-primary">
          <Plus :size="18" />
          test
        </button>
      </header>

      <DataTable :columns="tableColumns" :provider="fetchModelsProvider" :per-page="10"/>

      <BaseModal
        :show="isModalOpen"
        @close="isModalOpen = false"
        title="test"
      >
        <div class="form-group">
          <label>Name</label>
          <input
            v-model="formData.name"
            type="text"
            placeholder="test"
            class="form-input"
          />
        </div>

        <div class="form-group">
          <label>number</label>
          <input
            v-model="formData.accuracy"
            type="text"
            placeholder="67"
            class="form-input"
          />
        </div>

        <template #footer>
          <button @click="isModalOpen = false" class="btn-secondary">Cancel</button>
          <button @click="handleSave" class="btn-primary">Save model</button>
        </template>
      </BaseModal>
    </main>
  </div>
</template>

<script setup>
import {reactive, ref} from 'vue';
import { Plus } from 'lucide-vue-next';
import SideBar from "./components/SideBar.vue";
import DataTable from "./components/DataTable.vue";
import BaseModal from "./components/BaseModal.vue";

const isModalOpen = ref(false);

const tableColumns = [
  { key: 'id', label: 'ID' },
  { key: 'name', label: 'Nazwa' },
  { key: 'accuracy', label: 'Dokładność' },
  { key: 'status', label: 'Status' },
  { key: 'createdAt', label: 'Data utworzenia' },
];

const formData = reactive({
  name: '',
  accuracy: ''
});

const resetForm = () => {
  formData.name = '';
  formData.accuracy = '';
};

const handleSave = () => {
  console.log("Save:", formData.name);

  isModalOpen.value = false;
  resetForm();
};

const fetchModelsProvider = async ({ page, limit }) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Just a symulation there will be a real API call here
      const allData = [
        { id: '1', name: 'AAA', accuracy: '94.2%', status: 'aaa', createdAt: '2026-03-15' },
        { id: '2', name: 'BBB', accuracy: '89.1%', status: 'as', createdAt: '2026-04-01' },
        { id: '3', name: 'CCC', accuracy: '91.5%', status: 'sfasd', createdAt: '2026-04-02' },
        { id: '4', name: 'DD1', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '5', name: 'DD2', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '6', name: 'DD3', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '7', name: 'DD4', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '8', name: 'DD5', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '9', name: 'DD6', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '10', name: 'DD7', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '11', name: 'DD8', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '12', name: 'DD9', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '13', name: 'DD10', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '14', name: 'DD11', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
        { id: '15', name: 'DD12', accuracy: '96.8%', status: 'sfasda', createdAt: '2026-04-03' },
      ];

      const start = (page - 1) * limit;
      const end = start + limit;
      const paginatedData = allData.slice(start, end);

      resolve({
        data: paginatedData,
        total: allData.length
      });
    }, 1500);
  });
};
</script>

<style>
*, *::before, *::after {
  box-sizing: border-box;
}

body, html {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
}

.app-wrapper {
  display: flex;
  flex-direction: row;
  min-height: 100vh;
  width: 100%;
  background-color: #ffffff;
}

.content {
  flex: 1;
  padding: 40px;
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 32px;
}

h1 {
  color: #111827;
  margin-bottom: 8px;
  margin-top: 0;
  font-size: 1.875rem;
}

p {
  color: #6b7280;
  margin: 0;
  line-height: 1.5;
}

.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  background-color: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 500;
  font-size: 0.9rem;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-primary:hover {
  background-color: #1d4ed8;
}

.btn-secondary {
  padding: 10px 18px;
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  color: #4b5563;
  font-weight: 500;
  cursor: pointer;
}

.btn-secondary:hover {
  background-color: #f9fafb;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 6px;
}

.form-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s;
}

.form-input:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}
</style>