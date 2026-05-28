import { createRouter, createWebHistory } from 'vue-router'
import Dags from '../views/Dags.vue'
import DagDetail from '../views/DagDetail.vue'
import S3 from '../views/S3.vue'

const routes = [
  {
    path: '/',
    redirect: '/dags',
  },
  {
    path: '/dags',
    name: 'DAGs',
    component: Dags,
    meta: { description: 'Manage and monitor your Airflow DAGs.' },
  },
  {
    path: '/dags/:dagId',
    name: 'DAG details',
    component: DagDetail,
    props: true,
    meta: { description: 'Interactive DAG graph, task instances and logs.' },
  },
  {
    path: '/s3',
    name: 'Storage',
    component: S3,
    meta: { description: 'Upload, register, and manage datasets stored in S3.' },
  },
  {
    path: '/modeling',
    name: 'Modeling',
    component: () => import('../features/modeling/views/Modeling.vue'),
    meta: { description: 'Configure preprocessing, feature engineering and model training.' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
