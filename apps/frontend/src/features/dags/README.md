# features/dags

Feature-sliced module for DAG management (dashboard, graph, task details).

## Structure

```text
features/dags/
├── components/        # presentational Vue components (status badge, task node, sheet...)
├── composables/       # reusable hooks: useDagList, useDagGraph, useLogStream, useDagLayout
├── services/          # API client (wraps services/api.js with DAG endpoints)
├── views/             # route-level pages: DagListView, DagDetailView
├── types.js           # JSDoc @typedef definitions mirroring backend Pydantic DTOs
└── index.js           # public barrel — only this file is imported from outside features/dags
```

## Conventions

- **Source of truth for DTOs**: `docs/architecture/dag-management.md` (root of repo).
  Any change to `types.js` here must be accompanied by the matching change in
  `apps/backend/app/models/dags.py`.
- **Imports**: external code imports from `@/features/dags` (barrel), never deep.
- **State**: server state via TanStack Vue Query; UI state via local refs or a
  feature-scoped Pinia store (`composables/useDagsUiStore.js`, TBD).
- **Status colors**: defined exclusively in `components/DagStatusBadge.vue` (cva variants);
  no other component reaches for `bg-emerald-500` directly.
