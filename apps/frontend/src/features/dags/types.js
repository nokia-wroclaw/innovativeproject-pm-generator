/**
 * @file JSDoc type definitions for the DAG management feature.
 *
 * These typedefs mirror the Pydantic DTOs in
 * `apps/backend/app/models/dags.py` and the contract in
 * `docs/architecture/dag-management.md` §3.
 *
 * Keeping them in JSDoc form gives us editor IntelliSense without
 * pulling TypeScript into a JS-only codebase. When you change a DTO
 * on the backend, change it here too. There is no codegen.
 *
 * Convention:
 *   - Timestamps are ISO 8601 strings (the wire format).
 *     Convert to Date only at the rendering edge.
 *   - Durations are numbers in milliseconds.
 *   - Statuses use the project-level enum strings, never raw Airflow
 *     states (those are preserved in `rawState` / `raw_state`).
 */

// ─────────────────────────────────────────────────────────────────────────────
// Status enums
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {(
 *   "success"
 *   | "running"
 *   | "failed"
 *   | "up_for_retry"
 *   | "queued"
 *   | "skipped"
 *   | "none"
 * )} TaskStatus
 */

/**
 * @typedef {(
 *   "success"
 *   | "running"
 *   | "failed"
 *   | "queued"
 * )} DagRunStatus
 */

/**
 * @typedef {(
 *   "manual"
 *   | "scheduled"
 *   | "backfill"
 *   | "asset_triggered"
 * )} RunType
 */

/**
 * @typedef {(
 *   "DEBUG"
 *   | "INFO"
 *   | "WARNING"
 *   | "ERROR"
 *   | "CRITICAL"
 * )} LogLevel
 */

// ─────────────────────────────────────────────────────────────────────────────
// DAG list / dashboard
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {object} DagStats
 * @property {number} success
 * @property {number} failed
 * @property {number} running
 * @property {number} total
 */

/**
 * @typedef {object} DagRunSummary
 * @property {string} run_id
 * @property {string} logical_date           ISO 8601
 * @property {string | null} start_date      ISO 8601 or null
 * @property {string | null} end_date        ISO 8601 or null
 * @property {number | null} duration_ms
 * @property {DagRunStatus} status
 * @property {string} raw_state              raw Airflow state
 * @property {RunType} run_type
 * @property {string | null} triggered_by    Keycloak preferred_username, if known
 */

/**
 * @typedef {object} DagSummary
 * @property {string} dag_id
 * @property {string} display_name
 * @property {string | null} description
 * @property {string[]} owners
 * @property {string[]} tags
 * @property {boolean} is_paused
 * @property {boolean} is_active
 * @property {string | null} schedule        cron / preset / null
 * @property {string | null} next_run_at     ISO 8601 or null
 * @property {DagRunSummary | null} last_run
 * @property {DagStats} stats_24h
 */

// ─────────────────────────────────────────────────────────────────────────────
// DAG detail / graph
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {object} TaskNode
 * @property {string} task_id
 * @property {string} label
 * @property {string} operator
 * @property {boolean} is_group
 * @property {string} trigger_rule
 * @property {number} retries_max
 * @property {boolean} depends_on_past
 */

/**
 * @typedef {object} TaskEdge
 * @property {string} source                 task_id of source
 * @property {string} target                 task_id of target
 */

/**
 * @typedef {object} DagGraph
 * @property {TaskNode[]} nodes
 * @property {TaskEdge[]} edges
 */

/**
 * @typedef {object} DagDetails
 * @property {DagSummary} summary
 * @property {DagGraph} graph
 * @property {DagRunSummary[]} recent_runs
 */

// ─────────────────────────────────────────────────────────────────────────────
// Task instance / tries / logs
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {object} TaskInstance
 * @property {string} task_id
 * @property {string} run_id
 * @property {TaskStatus} status
 * @property {string} raw_state
 * @property {number} try_number
 * @property {number} max_tries
 * @property {string | null} start_date
 * @property {string | null} end_date
 * @property {number | null} duration_ms
 * @property {string} operator
 * @property {string} pool
 * @property {string} queue
 * @property {Record<string, unknown>} executor_config
 * @property {string | null} note
 */

/**
 * @typedef {object} TaskTry
 * @property {number} try_number
 * @property {TaskStatus} status
 * @property {string | null} start_date
 * @property {string | null} end_date
 * @property {number | null} duration_ms
 */

/**
 * @typedef {object} LogLine
 * @property {string | null} timestamp        ISO 8601 if parsed
 * @property {LogLevel | null} level
 * @property {string | null} source           e.g. "scheduler", "task", "trigger"
 * @property {string} message
 */

/**
 * @typedef {object} LogChunk
 * @property {number} try_number
 * @property {number} seq                     monotonically increasing within a try
 * @property {LogLine[]} lines
 * @property {boolean} has_more
 * @property {string | null} continuation     opaque cursor for next page
 */

// ─────────────────────────────────────────────────────────────────────────────
// Mutations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {object} TriggerRequest
 * @property {Record<string, unknown> | null} [conf]
 * @property {string | null} [logical_date]   ISO 8601; defaults to server now()
 * @property {string | null} [note]
 */

/**
 * @typedef {object} ActionResponse
 * @property {string | null} run_id           new run_id (for trigger), otherwise null
 * @property {string} message                 human-readable, for toast
 * @property {number} airflow_status          raw HTTP status propagated from Airflow
 */

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Stable error codes returned by /api/v1.
 * @typedef {(
 *   "UNAUTHENTICATED"
 *   | "FORBIDDEN"
 *   | "DAG_NOT_FOUND"
 *   | "RUN_NOT_FOUND"
 *   | "TASK_NOT_FOUND"
 *   | "VALIDATION_ERROR"
 *   | "AIRFLOW_UNAVAILABLE"
 *   | "AIRFLOW_AUTH_FAILED"
 *   | "AIRFLOW_CONFLICT"
 *   | "RATE_LIMITED"
 *   | "INTERNAL_ERROR"
 * )} ApiErrorCode
 */

/**
 * @typedef {object} ApiError
 * @property {ApiErrorCode} error
 * @property {string} message
 * @property {Record<string, unknown> | null} details
 * @property {string} request_id
 */

// ─────────────────────────────────────────────────────────────────────────────
// SSE event union (logs stream)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @typedef {object} LogStreamChunkEvent
 * @property {"chunk"} type
 * @property {LogChunk} data
 *
 * @typedef {object} LogStreamHeartbeatEvent
 * @property {"heartbeat"} type
 * @property {{ ts: string }} data
 *
 * @typedef {object} LogStreamEndEvent
 * @property {"end"} type
 * @property {{ reason: "task_finished" | "user_disconnect" | "max_duration" }} data
 *
 * @typedef {object} LogStreamErrorEvent
 * @property {"error"} type
 * @property {{ error: ApiErrorCode, message: string }} data
 *
 * @typedef {(
 *   LogStreamChunkEvent
 *   | LogStreamHeartbeatEvent
 *   | LogStreamEndEvent
 *   | LogStreamErrorEvent
 * )} LogStreamEvent
 */

export {};
