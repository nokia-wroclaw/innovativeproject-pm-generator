/**
 * Public barrel for the DAG management feature.
 *
 * External code (router, App.vue, top-level views) should only import from
 * this file — never reach into internal paths like `features/dags/components/...`.
 */

export * as DagsApi from './services/dagsApi.js';

export { default as DagStatusBadge } from './components/DagStatusBadge.vue';
export { default as TaskStatusBadge } from './components/TaskStatusBadge.vue';
export { default as TaskNode } from './components/TaskNode.vue';
export { default as TaskDetailsSheet } from './components/TaskDetailsSheet.vue';
export { default as LogViewer } from './components/LogViewer.vue';
export { default as TriggerDagDialog } from './components/TriggerDagDialog.vue';
export { default as DagListView } from './views/DagListView.vue';
export { default as DagDetailView } from './views/DagDetailView.vue';
