import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import S3 from '../views/S3.vue'

const routes = [
  {
    path: '/',
    name: 'Dashboard',
    component: Dashboard,
    meta: { description: 'Manage your projects and overview statistics.' }
  },
  {
    path: '/s3',
    name: 'S3 Storage',
    component: S3,
    meta: { description: 'Upload, register, and manage datasets stored in S3.' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router