<template>
  <div class="app-wrapper">
    <SideBar />

    <main class="content">
      <header class="content-header">
        <div>
          <h1>Aplikacja jest robiona tu</h1>
          <p>test.</p>
        </div>

        <div class="header-actions">
          <span class="user-pill">{{ userDisplayName }}</span>

          <button @click="handleLogout" class="btn-secondary">Logout</button>

          <button @click="isModalOpen = true" class="btn-primary">
            <Plus :size="18" />
            test
          </button>
        </div>
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
import { computed, onMounted, reactive, ref } from 'vue';
import { Plus } from 'lucide-vue-next';
import SideBar from "./components/SideBar.vue";
import DataTable from "./components/DataTable.vue";
import BaseModal from "./components/BaseModal.vue";
import { fetchGenerationPage } from './services/api';
import { getAuthProfile, logout } from './auth/keycloak';

const isModalOpen = ref(false);
const userProfile = ref({ username: '', fullName: '' });

const tableColumns = [
  { key: 'id', label: 'ID' },
  { key: 'name', label: 'Nazwa' },
  { key: 'number', label: 'Number' },
];

const userDisplayName = computed(
  () => userProfile.value.fullName || userProfile.value.username || 'Authenticated user'
);

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
  return fetchGenerationPage({ page, limit });
};

const handleLogout = async () => {
  await logout();
};

onMounted(() => {
  userProfile.value = getAuthProfile();
});
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

.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.user-pill {
  display: inline-flex;
  align-items: center;
  padding: 8px 12px;
  border-radius: 999px;
  background-color: #eff6ff;
  color: #1e40af;
  font-size: 0.85rem;
  font-weight: 600;
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
