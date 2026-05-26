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

    <GlobalModelingNotifications />
  </div>
</template>

<script setup>

import {computed, onMounted, ref} from "vue";
import {getAuthProfile, logout} from "./auth/keycloak";
import SideBar from "./components/SideBar.vue";
import GlobalModelingNotifications from "./features/modeling/components/GlobalModelingNotifications.vue";
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
