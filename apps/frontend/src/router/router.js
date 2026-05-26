import { createRouter, createWebHistory } from 'vue-router'
import S3 from '../views/S3.vue'

const routes = [
  {
    path: '/',
    name: 'DAGs',
    component: () => import('../features/dags/views/DagListView.vue'),
    meta: { description: 'Manage and monitor your Airflow DAGs.' }
  },
  {
    path: '/dags/:dagId',
    name: 'DAG details',
    component: () => import('../features/dags/views/DagDetailView.vue'),
    meta: { description: 'Interactive DAG graph, task instances and logs.' }
  },
  {
    path: '/s3',
    name: 'S3 Storage',
    component: S3,
    meta: { description: 'Upload, register, and manage datasets stored in S3.' }
  },
  {
    path: '/modeling',
    name: 'Modelowanie',
    component: () => import('../features/modeling/views/Modeling.vue'),
    meta: {
      description: 'Configure preprocessing, feature engineering and model training.'
    }
  }
]

if (import.meta.env.DEV) {
  routes.push({
    path: '/_dev/dag-mocks',
    name: 'DAG mocks (dev)',
    component: () => import('../features/dags/views/DagMocksView.vue'),
    meta: { description: 'Visual sandbox for DAG management components (Faza 2b).' }
  })
}

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router