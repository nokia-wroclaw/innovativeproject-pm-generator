/**
 * @file API client for the DAG management feature.
 *
 * One thin function per endpoint defined in
 * `docs/architecture/dag-management.md` §4. The functions return whatever
 * the backend returns (JSON), with no extra transformation: shape mapping
 * already happens server-side via Pydantic, so the wire format equals our
 * JSDoc DTOs in `../types.js`.
 *
 * Errors are surfaced as `ApiError` (see `@/services/api`). Callers — usually
 * Vue Query composables — should not catch them here.
 */

import { authorizedRequest, buildSseUrl, ApiError } from '@/services/api';
import { getAccessToken } from '@/auth/keycloak';
// eslint-disable-next-line no-unused-vars -- imported for JSDoc cross-references
import * as Types from '../types.js';

const BASE = '/api/v1/dags';

const enc = encodeURIComponent;

const qs = (params = {}) => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v != null) sp.set(k, String(v));
  }
  const out = sp.toString();
  return out ? `?${out}` : '';
};

// ─── Reads ───────────────────────────────────────────────────────────────────

/** @returns {Promise<Types.DagSummary[]>} */
export const listDags = () => authorizedRequest(BASE);

/** @returns {Promise<Types.DagDetails>} */
export const getDagDetails = (dagId) =>
  authorizedRequest(`${BASE}/${enc(dagId)}`);

/** @returns {Promise<Types.DagRunSummary[]>} */
export const listDagRuns = (dagId, { limit, offset } = {}) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs${qs({ limit, offset })}`);

/** @returns {Promise<Types.TaskInstance[]>} */
export const listTaskInstances = (dagId, runId) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs/${enc(runId)}/tasks`);

/** @returns {Promise<Types.TaskInstance>} */
export const getTaskInstance = (dagId, runId, taskId) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs/${enc(runId)}/tasks/${enc(taskId)}`);

/** @returns {Promise<Types.TaskTry[]>} */
export const listTaskTries = (dagId, runId, taskId) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs/${enc(runId)}/tasks/${enc(taskId)}/tries`);

/** @returns {Promise<Types.LogChunk>} */
export const getTaskLogs = (dagId, runId, taskId, { tryNumber, token } = {}) =>
  authorizedRequest(
    `${BASE}/${enc(dagId)}/runs/${enc(runId)}/tasks/${enc(taskId)}/logs${qs({
      try_number: tryNumber,
      token,
    })}`,
  );

// ─── Mutations ───────────────────────────────────────────────────────────────

/** @returns {Promise<Types.ActionResponse>} */
export const triggerDag = (dagId, body = {}) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs`, {
    method: 'POST',
    body: JSON.stringify(body),
  });

/** @returns {Promise<Types.ActionResponse>} */
export const stopDagRun = (dagId, runId) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs/${enc(runId)}/stop`, {
    method: 'POST',
  });

/** @returns {Promise<Types.ActionResponse>} */
export const clearDagRun = (dagId, runId) =>
  authorizedRequest(`${BASE}/${enc(dagId)}/runs/${enc(runId)}/clear`, {
    method: 'POST',
  });

/** @returns {Promise<Types.ActionResponse>} */
export const clearTaskInstance = (dagId, runId, taskId, { downstream } = {}) =>
  authorizedRequest(
    `${BASE}/${enc(dagId)}/runs/${enc(runId)}/tasks/${enc(taskId)}/clear${qs({
      downstream,
    })}`,
    { method: 'POST' },
  );

// ─── SSE: live logs ──────────────────────────────────────────────────────────

/**
 * Opens an authenticated SSE connection to the log stream.
 *
 * Browser's native EventSource cannot set the Authorization header, so we
 * implement the SSE protocol over `fetch` + ReadableStream. This keeps the
 * Keycloak token in a header rather than the URL (no server-log leakage).
 *
 * @param {string} dagId
 * @param {string} runId
 * @param {string} taskId
 * @param {object} opts
 * @param {number} opts.tryNumber
 * @param {AbortSignal} [opts.signal]   pass an AbortController.signal to close
 * @param {(event: import('../types.js').LogStreamEvent) => void} opts.onEvent
 * @returns {Promise<void>}             resolves when the stream ends
 */
export async function consumeLogStream(
  dagId,
  runId,
  taskId,
  { tryNumber, signal, onEvent },
) {
  const token = await getAccessToken();
  const url = buildSseUrl(
    `${BASE}/${enc(dagId)}/runs/${enc(runId)}/tasks/${enc(taskId)}/logs/stream`,
    { try_number: tryNumber },
  );

  const response = await fetch(url, {
    method: 'GET',
    signal,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'text/event-stream',
    },
  });

  if (!response.ok || !response.body) {
    throw new ApiError({
      status: response.status,
      message: `Log stream failed: ${response.status} ${response.statusText}`,
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by double newline.
      let separator = buffer.indexOf('\n\n');
      while (separator !== -1) {
        const raw = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);
        const parsed = _parseSseEvent(raw);
        if (parsed) onEvent(parsed);
        separator = buffer.indexOf('\n\n');
      }
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      /* swallow */
    }
  }
}

/** @returns {import('../types.js').LogStreamEvent | null} */
function _parseSseEvent(raw) {
  let eventType = 'message';
  const dataLines = [];
  for (const rawLine of raw.split('\n')) {
    const line = rawLine.replace(/\r$/, '');
    if (line.startsWith(':')) continue; // comment / heartbeat
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (!dataLines.length) return null;
  const dataText = dataLines.join('\n');
  let data;
  try {
    data = JSON.parse(dataText);
  } catch {
    data = { raw: dataText };
  }
  return /** @type {import('../types.js').LogStreamEvent} */ ({
    type: eventType,
    data,
  });
}
