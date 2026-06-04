import { createRouter, createWebHistory } from 'vue-router'
import { hasAdminRole } from '../auth/keycloak'
import Dags from '../views/Dags.vue'
import DagDetail from '../views/DagDetail.vue'
import S3 from '../views/S3.vue'

const routes = [
  {
    path: '/',
    redirect: () => (hasAdminRole() ? '/dags' : '/generate'),
  },
  {
    path: '/dags',
    name: 'DAGs',
    component: Dags,
    meta: {
      description: 'Manage and monitor your Airflow DAGs.',
      requiresAdmin: true,
    },
  },
  {
    path: '/dags/:dagId',
    name: 'DAG details',
    component: DagDetail,
    props: true,
    meta: {
      description: 'Interactive DAG graph, task instances and logs.',
      requiresAdmin: true,
    },
  },
  {
    path: '/s3',
    name: 'Storage',
    component: S3,
    meta: {
      description: 'Browse raw, preprocessed and generated datasets stored in S3.',
    },
  },
  {
    path: '/modeling',
    name: 'Modeling',
    component: () => import('../features/modeling/views/Modeling.vue'),
    meta: {
      description: 'Configure preprocessing, feature engineering and model training.',
      requiresAdmin: true,
    },
  },
  {
    path: '/generate',
    name: 'Generate',
    component: () => import('../features/modeling/views/Generate.vue'),
    meta: {
      description: 'Generate synthetic data from trained models.',
    },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to) => {
  if (to.meta.requiresAdmin && !hasAdminRole()) {
    return { path: '/generate' }
  }
})

export default router
