import pmSchemaColumns from '../../../pm_schema_columns.json';

export const PM_REQUIRED_COLUMNS = pmSchemaColumns.required_columns;

/**
 * @param {string[]} columns
 * @returns {{ ok: boolean, missing: string[], present: string[], message: string, summary: object }}
 */
export function validatePmColumns(columns) {
  const present = (columns || []).map((c) => String(c).trim()).filter(Boolean);
  const presentSet = new Set(present);
  const missing = PM_REQUIRED_COLUMNS.filter((col) => !presentSet.has(col));

  if (missing.length === 0) {
    return { ok: true, missing: [], present, message: '', summary: null };
  }

  const message =
    'This dataset is not compatible with PM visualizations. ' +
    `Missing required columns: ${missing.join(', ')}. ` +
    `Found columns: ${present.length ? present.join(', ') : '(none)'}.`;

  const summary = {
    status: 'unsupported_schema',
    missing_columns: missing,
    required_columns: [...PM_REQUIRED_COLUMNS],
    present_columns: present,
    message,
  };

  return { ok: false, missing, present, message, summary };
}
