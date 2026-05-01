<template>
  <div class="app-wrapper">
    <SideBar />

    <main class="content">
      <header class="content-header">
        <div>
          <h1>{{ currentRouteTitle }}</h1>
          <p>{{ currentRouteDescription }}</p>
        </div>

        <div class="header-actions">
          <span class="user-pill">{{ userDisplayName }}</span>
          <button @click="handleLogout" class="btn-secondary">Logout</button>
        </div>
      </header>

      <router-view />
    </main>
  </div>
</template>

<script setup>

import {computed, onMounted, ref} from "vue";
import {getAuthProfile, logout} from "./auth/keycloak";
import SideBar from "./components/SideBar.vue";
import { useRoute } from "vue-router";


const route = useRoute();
const userProfile = ref({ username: '', fullName: '' });

const userDisplayName = computed(
  () => userProfile.value.fullName || userProfile.value.username || 'Authenticated user'
);

const currentRouteTitle = computed(() => {
  return route.name || 'GenPM';
});

const currentRouteDescription = computed(() => {
  return route.meta.description || 'Manage your platform settings and data.';
});


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
