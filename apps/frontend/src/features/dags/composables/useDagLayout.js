/**
 * @file useDagLayout — computes deterministic node positions via dagre.
 *
 * Airflow's DAG payload tells us *what* depends on *what* but not *where*
 * to draw it. We feed the graph through dagre (top-to-bottom by default,
 * mirroring Airflow's own Graph view) and emit Vue Flow-compatible
 * `Node[]` / `Edge[]`.
 *
 * Layout is pure & memoised by graph identity so re-rendering doesn't
 * trigger a re-layout while statuses change.
 */

import { computed, shallowRef, watch } from 'vue';
import dagre from '@dagrejs/dagre';

const NODE_WIDTH = 240;
const NODE_HEIGHT = 60;
const RANK_SEP = 70;   // vertical distance between rows (top-bottom)
const NODE_SEP = 40;   // horizontal distance between siblings

/**
 * @param {() => import('../types.js').DagGraph | null | undefined} graphGetter
 * @param {() => Record<string, import('../types.js').TaskInstance | undefined>} statusByTaskGetter
 *        map of task_id -> TaskInstance (for overlaying live status on each node)
 * @param {{ direction?: 'TB' | 'LR' }} [opts]
 */
export function useDagLayout(graphGetter, statusByTaskGetter, opts = {}) {
  const direction = opts.direction ?? 'TB';

  /** @type {import('vue').ShallowRef<{ nodes: any[], edges: any[] }>} */
  const layout = shallowRef({ nodes: [], edges: [] });

  const layoutKey = computed(() => {
    const graph = graphGetter();
    if (!graph) return '';
    // Cheap fingerprint: only re-layout when node/edge identity changes.
    return [
      graph.nodes.map((n) => n.task_id).join('|'),
      graph.edges.map((e) => `${e.source}>${e.target}`).join('|'),
      direction,
    ].join('::');
  });

  watch(
    layoutKey,
    () => {
      const graph = graphGetter();
      if (!graph) {
        layout.value = { nodes: [], edges: [] };
        return;
      }
      layout.value = _layoutGraph(graph, direction);
    },
    { immediate: true },
  );

  const nodes = computed(() => {
    const statusMap = statusByTaskGetter() || {};
    return layout.value.nodes.map((node) => {
      const taskInstance = statusMap[node.id];
      return {
        ...node,
        data: {
          ...node.data,
          status: taskInstance?.status ?? 'none',
          try_number: taskInstance?.try_number,
          max_tries: taskInstance?.max_tries,
          duration_ms: taskInstance?.duration_ms,
        },
      };
    });
  });

  const edges = computed(() => layout.value.edges);

  return { nodes, edges };
}

/** @param {import('../types.js').DagGraph} graph */
function _layoutGraph(graph, direction) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: direction,
    nodesep: NODE_SEP,
    ranksep: RANK_SEP,
    marginx: 24,
    marginy: 24,
  });
  g.setDefaultEdgeLabel(() => ({}));

  for (const node of graph.nodes) {
    g.setNode(node.task_id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of graph.edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  const flowNodes = graph.nodes.map((node) => {
    const { x, y } = g.node(node.task_id);
    return {
      id: node.task_id,
      type: 'task',
      position: { x: x - NODE_WIDTH / 2, y: y - NODE_HEIGHT / 2 },
      data: {
        task_id: node.task_id,
        label: node.label,
        operator: node.operator,
        is_group: node.is_group,
        status: 'none',
      },
    };
  });

  const flowEdges = graph.edges.map((edge) => ({
    id: `${edge.source}->${edge.target}`,
    source: edge.source,
    target: edge.target,
  }));

  return { nodes: flowNodes, edges: flowEdges };
}
