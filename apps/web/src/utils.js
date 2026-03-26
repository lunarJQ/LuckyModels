export function escapeValue(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

export function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
}

export function statusClass(status) {
  return String(status || "unknown").toLowerCase();
}

export function formatArray(value) {
  return Array.isArray(value) && value.length ? value.join(", ") : "-";
}
