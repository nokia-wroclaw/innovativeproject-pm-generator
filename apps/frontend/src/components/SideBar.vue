<template>
  <aside :class="['sidebar', { 'is-collapsed': isCollapsed }]">
    <div class="sidebar-header">
      <div v-if="!isCollapsed" class="logo">GenPM</div>

      <SidebarToggleButton
        :is-collapsed="isCollapsed"
        @toggle="toggleSidebar"
      />
    </div>

    <nav class="menu">
      <div class="menu-item">
        <LayoutDashboard class="icon" :size="20" />
        <span v-if="!isCollapsed">Dashboard</span>
      </div>
      <div class="menu-item">
        <Database class="icon" :size="20" />
        <span v-if="!isCollapsed">S3</span>
      </div>
      <div class="menu-item">
        <Brain class="icon" :size="20" />
        <span v-if="!isCollapsed">Train</span>
      </div>
      <div class="menu-item">
        <Cpu class="icon" :size="20" />
        <span v-if="!isCollapsed">Generate</span>
      </div>
      <div class="menu-item">
        <LineChart class="icon" :size="20" />
        <span v-if="!isCollapsed">Metrics</span>
      </div>
    </nav>
  </aside>
</template>

<script setup>
import { ref } from 'vue';
import { LayoutDashboard, Database, Brain, Cpu, LineChart } from 'lucide-vue-next';
import SidebarToggleButton from './ToggleButton.vue';

const isCollapsed = ref(false);
const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value;
};
</script>

<style scoped>
.sidebar {
  background-color: #ffffff;
  width: 260px;
  height: 100vh;
  position: sticky;
  top: 0;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  border-right: 1px solid #e5e7eb;
  transition: width 0.3s ease;
  overflow-x: hidden;
}

.sidebar.is-collapsed {
  width: 70px;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 70px;
  padding: 0 20px;
  border-bottom: 1px solid #e5e7eb;
  white-space: nowrap;
}

.sidebar.is-collapsed .sidebar-header {
  justify-content: center;
  padding: 0;
}

.logo {
  font-weight: 700;
  font-size: 2rem;
  color: #005AFF;
}

.menu {
  padding: 20px 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.menu-item {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  cursor: pointer;
  color: #4b5563;
  font-weight: 500;
  font-size: 0.95rem;
  transition: all 0.2s ease;
  white-space: nowrap;
  border-left: 3px solid transparent;
}

.menu-item:hover {
  background-color: #f4f8ff;
  color: #005AFF;
  border-left: 3px solid #005AFF;
}

.sidebar.is-collapsed .menu-item {
  padding: 12px 0;
  justify-content: center;
}

.icon {
  display: flex;
  flex-shrink: 0;
  justify-content: center;
  margin-right: 16px;
  transition: margin 0.3s ease;
}

.sidebar.is-collapsed .icon {
  margin-right: 0;
}
</style>