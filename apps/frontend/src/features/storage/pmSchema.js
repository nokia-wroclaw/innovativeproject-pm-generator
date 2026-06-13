import pmSchemaColumns from '../../../pm_schema_columns.json';

export const PM_REQUIRED_COLUMNS = pmSchemaColumns.required_columns;
export const PM_COLUMN_ALIASES = pmSchemaColumns.column_aliases ?? {};
export const PM_DERIVED_COLUMNS = pmSchemaColumns.derived_columns ?? {};

function columnSatisfied(canonical, presentSet) {
  if (presentSet.has(canonical)) return true;
  for (const alias of PM_COLUMN_ALIASES[canonical] ?? []) {
    if (presentSet.has(alias)) return true;
  }
  const source = PM_DERIVED_COLUMNS[canonical];
  if (source && columnSatisfied(source, presentSet)) return true;
  return false;
}

/**
 * @param {string[]} columns
 * @returns {{ ok: boolean, missing: string[], present: string[], message: string, summary: object }}
 */
export function validatePmColumns(columns) {
  const present = (columns || []).map((c) => String(c).trim()).filter(Boolean);
  const presentSet = new Set(present);
  const missing = PM_REQUIRED_COLUMNS.filter((col) => !columnSatisfied(col, presentSet));

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
