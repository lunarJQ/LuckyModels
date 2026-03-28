export function escapeValue(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

export function safeJsonParse(value) {
  if (value && typeof value === "object") {
    return value;
  }
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

export function formatStatusLabel(status) {
  const normalized = String(status || "unknown").toLowerCase();
  const mapping = {
    running: "运行中",
    stopped: "已停止",
    starting: "启动中",
    stopping: "停止中",
    planned: "已规划",
    deleted: "已删除",
    failed: "失败",
    completed: "已完成",
    unknown: "未知",
    inference_ready: "可推理",
    package_ready: "仅制品就绪",
    pending: "待检测",
    empty: "空白",
  };
  return mapping[normalized] || String(status || "-");
}

export function formatBooleanLabel(value, yesLabel = "是", noLabel = "否") {
  return value ? yesLabel : noLabel;
}
