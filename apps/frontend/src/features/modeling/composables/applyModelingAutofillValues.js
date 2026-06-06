export function applyModelingAutofillValues(form, values) {
  if (!values || typeof values !== 'object') return;

  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) continue;
    if (key in form) {
      form[key] = value;
    }
  }
}
