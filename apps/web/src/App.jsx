import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

import {
  createAccessUser,
  createBenchmark,
  createDeployment,
  deleteBenchmark,
  deleteDeployment,
  deleteModelPackage,
  deletePredictionArchive,
  exportModelPackage,
  fetchAuthSession,
  getDatasetSampleText,
  getDeploymentPolicy,
  getDeploymentPolicyAutomation,
  getDeploymentHealth,
  getDeploymentLogs,
  getModelProfiles,
  importModelArchive,
  loginWithPassword,
  loadDashboardData,
  logoutSession,
  predictWithDeployment,
  resetAccessUserPassword,
  restartDeployment,
  runDeploymentPolicyAutomationNow,
  runTaskRoutingStrategy,
  switchCurrentAccessUser,
  stopDeployment,
  toggleAccessUser,
  updatePredictionArchiveFeedback,
  updateDeploymentPolicyAutomation,
  updateAccessUser,
} from "./api";
import { EmptyState, MetricCard, SectionHeader, StatusPill } from "./components";
import { navGroups } from "./nav";
import { escapeValue, formatBooleanLabel, formatStatusLabel, safeJsonParse } from "./utils";

const initialData = {
  summary: null,
  models: [],
  modelDetails: {},
  capabilityMatrix: [],
  deployments: [],
  benchmarks: [],
  predictArchives: [],
  report: null,
  gpuStatus: null,
  accessOverview: null,
  importOptions: null,
  datasetCatalog: null,
  taskRoutingFeedbackOverview: null,
};

const DEFAULT_PREDICT_TEXT = "LuckyModels provides a stable fallback sample for relation extraction.";
const DEFAULT_PREDICT_DATASETS = [
  { dataset_key: "webnlg", label: "WebNLG", dataset_name: "WebNLG", available: false, sample_ready: false },
  { dataset_key: "nyt", label: "NYT", dataset_name: "NYT", available: false, sample_ready: false },
  { dataset_key: "scierc", label: "SciERC", dataset_name: "SciERC", available: false, sample_ready: false },
  { dataset_key: "duie", label: "DuIE2.0", dataset_name: "DuIE2.0", available: false, sample_ready: false },
  {
    dataset_key: "equipmentmix15k",
    label: "IndustrialEquipmentMix15k",
    dataset_name: "IndustrialEquipmentMix15k",
    available: false,
    sample_ready: false,
  },
];

function buildModelTransferTask(overrides = {}) {
  return {
    taskId: overrides.taskId || `transfer-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    direction: "",
    phase: "idle",
    percent: null,
    loadedBytes: 0,
    totalBytes: null,
    fileName: "",
    modelId: "",
    detail: "",
    ...overrides,
  };
}

const pageMeta = {
  "/": {
    title: "平台首页",
    note: "面向 LuckyModels 的真实控制台总览，展示模型、部署、压测与运行状态。",
  },
  "/dashboard": {
    title: "平台首页",
    note: "面向 LuckyModels 的真实控制台总览，展示模型、部署、压测与运行状态。",
  },
  "/experiments": {
    title: "在线推理",
    note: "选择运行中的部署实例做真实推理，查看三元组、健康状态和运行日志。",
  },
  "/access-control": {
    title: "用户权限",
    note: "统一查看当前用户、角色、权限范围以及系统内的用户构成。",
  },
  "/admin-console": {
    title: "管理员面板",
    note: "创建用户、调整角色、停用账号，并管理平台管理员相关配置。",
  },
  "/task-routing": {
    title: "任务调度",
    note: "根据目标偏好、文本难度、领域信息以及 benchmark 与样本回放证据，在单模型与多模型抽取之间自适应切换并做轻量自校准。",
  },
  "/deployment-policy": {
    title: "部署策略",
    note: "根据热温冷分层、冷启动风险与 GPU 压力，生成自适应部署建议，并支持定时自动优化。",
  },
  "/archive": {
    title: "结果归档",
    note: "集中查看最近的部署实例与基准记录，便于截图和整理论文表格。",
  },
  "/models": {
    title: "模型库",
    note: "查看模型资产并从平台直接发起新的部署实例。",
  },
  "/capability-matrix": {
    title: "能力矩阵",
    note: "固化展示模型接入状态、推理能力、基准支持情况与当前限制，直接服务论文表格整理。",
  },
  "/artifacts": {
    title: "制品包",
    note: "查看 manifest、ONNX 制品、tokenizer 和配置模板解析结果。",
  },
  "/deployments": {
    title: "部署记录",
    note: "新建、停止、重启、删除部署实例，并查看健康与日志。",
  },
  "/benchmarks": {
    title: "基准剖析",
    note: "为运行中的部署实例创建基准测试，并管理历史基准记录。",
  },
  "/resources": {
    title: "运行资源",
    note: "汇总 GPU 设备状态、部署绑定关系、执行后端信息与运行中推理进程。",
  },
  "/logs": {
    title: "日志中心",
    note: "聚焦 LuckyModels 当前选中部署实例的运行日志。",
  },
};

function buildInitialDeploymentForm(models, details) {
  const preferred = models.find((model) => model.predict_ready) || models[0];
  const modelId = preferred?.model_id || "";
  const profiles = getProfilesForModel(modelId, details);
  return {
    modelId,
    profileName: profiles.includes("default") ? "default" : profiles[0] || "",
    useGpu: true,
    gpuIdsText: "",
    notes: "",
    launchRuntime: true,
  };
}

function buildInitialModelImportForm(importOptions) {
  const defaultFamily = importOptions?.families?.[0];
  return {
    importMode: "auto_package",
    modelFamily: defaultFamily?.code || "casrel",
    language: defaultFamily?.default_language || "en",
    datasetName: "",
    version: "v1",
    modelName: "",
    backbone: "",
    alias: "",
    archive: null,
  };
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "-";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getProfilesForModel(modelId, details) {
  if (!modelId) {
    return [];
  }
  return Object.keys(details[modelId]?.resolved_profiles || {});
}

function resolveProfileName(modelId, details, preferredProfileName = "") {
  const profiles = getProfilesForModel(modelId, details);
  if (!profiles.length) {
    return "";
  }
  if (preferredProfileName && profiles.includes(preferredProfileName)) {
    return preferredProfileName;
  }
  if (profiles.includes("default")) {
    return "default";
  }
  return profiles[0];
}

function normalizeKeyword(value) {
  return String(value || "").toLowerCase().trim();
}

function matchesKeyword(values, keyword) {
  const term = normalizeKeyword(keyword);
  if (!term) {
    return true;
  }
  return values.some((value) => normalizeKeyword(value).includes(term));
}

function sortByCreatedDesc(items) {
  return [...items].sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")));
}

function normalizePredictionFeedbackStatus(value) {
  const normalized = String(value || "pending").trim().toLowerCase();
  if (normalized === "satisfied" || normalized === "unsatisfied") {
    return normalized;
  }
  return "pending";
}

function formatPredictionFeedbackLabel(value) {
  const normalized = normalizePredictionFeedbackStatus(value);
  if (normalized === "satisfied") {
    return "满意";
  }
  if (normalized === "unsatisfied") {
    return "不满意";
  }
  return "暂定";
}

function getDeploymentHealthPayload(deployment) {
  return safeJsonParse(deployment?.health?.payload || {});
}

function getDeploymentExecution(deployment) {
  const payload = getDeploymentHealthPayload(deployment);
  return payload.execution || deployment.execution || {};
}

function getDeploymentProviders(deployment) {
  const payload = getDeploymentHealthPayload(deployment);
  return Array.isArray(payload.providers) ? payload.providers : [];
}

function formatExecutionSummary(deployment) {
  const execution = getDeploymentExecution(deployment);
  const workers = execution.instance_count ?? 1;
  const batchSize = execution.batch_size ?? 1;
  const intraThreads = execution.intra_op_threads ?? 0;
  const interThreads = execution.inter_op_threads ?? 0;
  return `实例=${workers} · 批量=${batchSize} · 算子内线程=${intraThreads} · 算子间线程=${interThreads}`;
}

function formatProviderSummary(deployment) {
  const providers = getDeploymentProviders(deployment);
  return providers.length ? providers.join(", ") : "待检测";
}

function getDeploymentAssignedGpuIds(deployment) {
  const payload = getDeploymentHealthPayload(deployment);
  if (Array.isArray(payload.assigned_gpu_ids) && payload.assigned_gpu_ids.length) {
    return payload.assigned_gpu_ids;
  }
  return Array.isArray(deployment?.gpu_ids) ? deployment.gpu_ids : [];
}

function formatGpuIds(ids) {
  if (!Array.isArray(ids) || !ids.length) {
    return "自动分配";
  }
  return ids.join(", ");
}

function formatLanguageLabel(language) {
  const normalized = String(language || "").toLowerCase();
  if (["en", "eng", "english"].includes(normalized)) {
    return "英文";
  }
  if (["zh", "zh-cn", "cn", "chinese"].includes(normalized)) {
    return "中文";
  }
  return language || "-";
}

function parseGpuIdsText(value) {
  const text = String(value || "").replaceAll("，", ",").trim();
  if (!text) {
    return [];
  }
  const seen = new Set();
  return text
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((token) => {
      if (!/^\d+$/.test(token)) {
        throw new Error(`GPU 卡号格式无效：${token}`);
      }
      return Number.parseInt(token, 10);
    })
    .filter((gpuId) => {
      if (seen.has(gpuId)) {
        return false;
      }
      seen.add(gpuId);
      return true;
    });
}

function formatFixed(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function meanBy(items, selector) {
  const values = items
    .map((item) => selector(item))
    .filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value)))
    .map((value) => Number(value));
  if (!values.length) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function csvEscape(value) {
  const text = value === null || value === undefined ? "" : String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function buildDownloadFilename(filename) {
  const text = String(filename || "luckymodels-export").trim();
  const lastDot = text.lastIndexOf(".");
  const hasExtension = lastDot > 0 && lastDot < text.length - 1;
  const baseName = hasExtension ? text.slice(0, lastDot) : text;
  const extension = hasExtension ? text.slice(lastDot) : "";
  const safeBaseName = baseName
    .replace(/[\\/:*?"<>|]+/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "") || "luckymodels-export";
  const now = new Date();
  const timestamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, "0"),
    String(now.getDate()).padStart(2, "0"),
    "-",
    String(now.getHours()).padStart(2, "0"),
    String(now.getMinutes()).padStart(2, "0"),
    String(now.getSeconds()).padStart(2, "0"),
    "-",
    String(now.getMilliseconds()).padStart(3, "0"),
  ].join("");
  return `${safeBaseName}-${timestamp}${extension}`;
}

function downloadCsvFile(filename, rows) {
  if (!rows.length) {
    return;
  }
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.map(csvEscape).join(","),
    ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
  ];
  const blob = new Blob([`\uFEFF${lines.join("\n")}`], { type: "text/csv;charset=utf-8;" });
  downloadBlobFile(filename, blob);
}

function downloadJsonFile(filename, value) {
  const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: "application/json;charset=utf-8;" });
  downloadBlobFile(filename, blob);
}

function downloadBlobFile(filename, blob) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = buildDownloadFilename(filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

function buildInitialBenchmarkForm(deployments) {
  const running = deployments.find((item) => item.status === "running") || deployments[0];
  return {
    deploymentId: running?.deployment_id || "",
    text: "",
    label: "",
    iterations: 3,
    warmupRuns: 1,
  };
}

function resolvePredictDatasetKey(datasetName) {
  const raw = String(datasetName || "").trim().toLowerCase();
  if (!raw) {
    return "webnlg";
  }
  const aliasMap = {
    duie: "duie",
    "duie2.0": "duie",
    "duie2.0-main": "duie",
    nyt: "nyt",
    raw_webnlg: "webnlg",
    scierc: "scierc",
    webnlg: "webnlg",
    equipmentmix15k: "equipmentmix15k",
    industrial: "equipmentmix15k",
    industrialequipmentmix15k: "equipmentmix15k",
    industrialequipmentmix15k_mfgdominant: "equipmentmix15k",
  };
  return aliasMap[raw] || raw;
}

function buildInitialPredictForm(deployments, models) {
  const running = deployments.find((item) => item.status === "running") || deployments[0];
  const matchingModel = models.find((item) => item.model_id === running?.model_id);
  const datasetKey = resolvePredictDatasetKey(matchingModel?.dataset_name);
  return {
    deploymentId: running?.deployment_id || "",
    text: DEFAULT_PREDICT_TEXT,
    scene: datasetKey,
    optionsJson: "{\"max_length\": 100}",
  };
}

function buildInitialTaskRoutingForm(report) {
  const sampleText =
    report?.samples?.primary_case?.sentText ||
    report?.samples?.secondary_case?.sentText ||
    "";
  return {
    text: sampleText,
    objective: "domain",
    language: "auto",
    maxModels: 3,
    scene: "zh_industrial",
    optionsJson: "{}",
  };
}

function buildInitialDeploymentPolicyAutomationForm() {
  return {
    enabled: true,
    intervalSeconds: 900,
    protectActiveWindowSeconds: 120,
    gpuMemoryRestartEnabled: true,
    gpuMemoryRestartIntervalSeconds: 60,
    gpuMemoryRestartThresholdMb: 5120,
  };
}

function buildInitialAccessUserForm(roles) {
  const preferredRole = roles.find((item) => item.role_id === "user") || roles[0];
  return {
    username: "",
    displayName: "",
    email: "",
    password: "",
    roleId: preferredRole?.role_id || "",
    notes: "",
  };
}

function buildInitialLoginForm() {
  return {
    username: "",
    password: "",
  };
}

function formatRoleLabel(roleId) {
  const mapping = {
    admin: "平台管理员",
    user: "普通用户",
  };
  return mapping[roleId] || roleId || "-";
}

function formatPermissionLabel(permissionId) {
  const mapping = {
    "models.view": "查看模型",
    "deployments.view": "查看部署",
    "deployments.manage": "管理部署",
    "benchmarks.view": "查看基准",
    "benchmarks.manage": "运行基准",
    "predictions.view": "查看预测归档",
    "strategies.manage": "运行策略",
    "resources.view": "查看资源",
    "archive.export": "导出归档",
    "users.view": "查看权限",
    "users.manage": "管理用户",
  };
  return mapping[permissionId] || permissionId || "-";
}

const pageConfigMap = Object.fromEntries(
  navGroups.flatMap((group) =>
    group.items.map((item) => [item.path, item]),
  ),
);

function hasRequiredPermissions(currentPermissions, requiredPermissions = []) {
  if (!requiredPermissions.length) {
    return true;
  }
  const permissionSet = new Set(currentPermissions || []);
  return requiredPermissions.every((permissionId) => permissionSet.has(permissionId));
}

function getRequiredPermissionsForPath(pathname) {
  if (pathname === "/dashboard") {
    return pageConfigMap["/"]?.requiredPermissions || [];
  }
  return pageConfigMap[pathname]?.requiredPermissions || [];
}

function isAdminOnlyPath(pathname) {
  if (pathname === "/dashboard") {
    return Boolean(pageConfigMap["/"]?.adminOnly);
  }
  return Boolean(pageConfigMap[pathname]?.adminOnly);
}

function canAccessPath(pathname, currentPermissions, currentUser) {
  const hasPermissions = hasRequiredPermissions(currentPermissions, getRequiredPermissionsForPath(pathname));
  if (!hasPermissions) {
    return false;
  }
  if (isAdminOnlyPath(pathname) && !currentUser?.is_admin) {
    return false;
  }
  return true;
}

function getVisibleNavGroups(currentPermissions, currentUser) {
  return navGroups
    .map((group) => ({
      ...group,
      items: group.items.filter((item) => canAccessPath(item.path, currentPermissions, currentUser)),
    }))
    .filter((group) => group.items.length);
}

function getFallbackPath(currentPermissions, currentUser) {
  const visibleGroups = getVisibleNavGroups(currentPermissions, currentUser);
  return visibleGroups[0]?.items[0]?.path || "/";
}

export default function App() {
  const [data, setData] = useState(initialData);
  const [authSession, setAuthSession] = useState(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [loginForm, setLoginForm] = useState(buildInitialLoginForm());
  const [loadStatus, setLoadStatus] = useState("正在加载");
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState({ type: "ok", text: "" });
  const [actionBusy, setActionBusy] = useState({});
  const [modelImportForm, setModelImportForm] = useState(buildInitialModelImportForm(null));
  const [modelTransferItems, setModelTransferItems] = useState([]);
  const [deploymentForm, setDeploymentForm] = useState(buildInitialDeploymentForm([], {}));
  const [benchmarkForm, setBenchmarkForm] = useState(buildInitialBenchmarkForm([]));
  const [predictForm, setPredictForm] = useState(buildInitialPredictForm([], []));
  const [predictSampleInfo, setPredictSampleInfo] = useState(null);
  const [predictSampleCursor, setPredictSampleCursor] = useState({});
  const [predictResult, setPredictResult] = useState(null);
  const [taskRoutingForm, setTaskRoutingForm] = useState(buildInitialTaskRoutingForm(null));
  const [taskRoutingResult, setTaskRoutingResult] = useState(null);
  const [deploymentPolicy, setDeploymentPolicy] = useState(null);
  const [deploymentPolicyAutomation, setDeploymentPolicyAutomation] = useState(null);
  const [deploymentPolicyAutomationForm, setDeploymentPolicyAutomationForm] = useState(buildInitialDeploymentPolicyAutomationForm());
  const [selectedHealth, setSelectedHealth] = useState(null);
  const [selectedLogs, setSelectedLogs] = useState(null);

  function addModelTransferTask(task) {
    setModelTransferItems((current) => [task, ...current]);
  }

  function updateModelTransferTask(taskId, patch) {
    setModelTransferItems((current) =>
      current.map((item) => (item.taskId === taskId ? { ...item, ...patch } : item)),
    );
  }

  function removeModelTransferTask(taskId) {
    setModelTransferItems((current) => current.filter((item) => item.taskId !== taskId));
  }

  function scheduleModelTransferCleanup(taskId, delayMs = 3500) {
    window.setTimeout(() => {
      setModelTransferItems((current) => current.filter((item) => item.taskId !== taskId));
    }, delayMs);
  }

  useEffect(() => {
    void initializeSession();
  }, []);

  useEffect(() => {
    if (!data.models.length) return;
    setDeploymentForm((current) => {
      if (current.modelId && data.models.some((item) => item.model_id === current.modelId)) {
        const profiles = Object.keys(data.modelDetails[current.modelId]?.resolved_profiles || {});
        if (profiles.includes(current.profileName)) {
          return current;
        }
        return {
          ...current,
          profileName: profiles.includes("default") ? "default" : profiles[0] || "",
        };
      }
      return buildInitialDeploymentForm(data.models, data.modelDetails);
    });
  }, [data.models, data.modelDetails]);

  useEffect(() => {
    if (!data.importOptions?.families?.length) {
      return;
    }
    setModelImportForm((current) => {
      const families = data.importOptions.families;
      const matchedFamily = families.find((item) => item.code === current.modelFamily) || families[0];
      return {
        ...current,
        modelFamily: matchedFamily.code,
        language: current.language || matchedFamily.default_language || "en",
      };
    });
  }, [data.importOptions]);

  useEffect(() => {
    setBenchmarkForm((current) => {
      if (current.deploymentId && data.deployments.some((item) => item.deployment_id === current.deploymentId)) {
        return current;
      }
      return buildInitialBenchmarkForm(data.deployments);
    });
    setPredictForm((current) => {
      if (current.deploymentId && data.deployments.some((item) => item.deployment_id === current.deploymentId)) {
        return current;
      }
      return buildInitialPredictForm(data.deployments, data.models);
    });
  }, [data.deployments, data.models]);

  useEffect(() => {
    const sampleText =
      data.report?.samples?.primary_case?.sentText ||
      data.report?.samples?.secondary_case?.sentText ||
      "";

    if (!sampleText) {
      return;
    }

    setPredictForm((current) => {
      return current;
    });

    setBenchmarkForm((current) => {
      if (current.text) {
        return current;
      }
      return { ...current, text: sampleText };
    });

    setTaskRoutingForm((current) => {
      if (current.text) {
        return current;
      }
      return { ...current, text: sampleText };
    });
  }, [data.report]);

  async function hydratePredictSample(datasetKey, { advance = false } = {}) {
    const resolvedDatasetKey = resolvePredictDatasetKey(datasetKey);
    const currentIndex = predictSampleCursor[resolvedDatasetKey] ?? 0;
    const nextIndex = advance ? currentIndex + 1 : 0;
    setActionBusy((current) => ({ ...current, [`predict-sample-${resolvedDatasetKey}`]: true }));
    try {
      const sample = await getDatasetSampleText(resolvedDatasetKey, nextIndex);
      const text = String(sample?.text || data.datasetCatalog?.default_fallback_text || DEFAULT_PREDICT_TEXT);
      setPredictSampleInfo(sample);
      setPredictSampleCursor((current) => ({
        ...current,
        [resolvedDatasetKey]: Number(sample?.effective_index ?? 0),
      }));
      setPredictForm((current) => (
        current.scene === resolvedDatasetKey
          ? { ...current, text }
          : current
      ));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const fallbackText = data.datasetCatalog?.default_fallback_text || DEFAULT_PREDICT_TEXT;
      setPredictSampleInfo({
        dataset_key: resolvedDatasetKey,
        dataset_name: resolvedDatasetKey,
        dataset_source: "-",
        record_count: 0,
        effective_index: 0,
        case_id: null,
        text: fallbackText,
        triple_count: 0,
        is_fallback: true,
        detail: message,
      });
      setPredictSampleCursor((current) => ({ ...current, [resolvedDatasetKey]: 0 }));
      setPredictForm((current) => (
        current.scene === resolvedDatasetKey
          ? { ...current, text: fallbackText }
          : current
      ));
    } finally {
      setActionBusy((current) => ({ ...current, [`predict-sample-${resolvedDatasetKey}`]: false }));
    }
  }

  useEffect(() => {
    if (!predictForm.scene) {
      return;
    }
    void hydratePredictSample(predictForm.scene, { advance: false });
  }, [predictForm.scene]);

  async function refreshAll({ silent = false, sessionOverride = null } = {}) {
    if (!silent) {
      setLoadError("");
      setLoadStatus("正在加载");
    }
    try {
      const next = await loadDashboardData(sessionOverride || authSession);
      startTransition(() => {
        setData(next);
        setLoadStatus("已加载");
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes("401")) {
        setAuthSession({ authenticated: false, current_user: null, current_permissions: [] });
      }
      setLoadError(message);
      setLoadStatus("加载失败");
    }
  }

  async function initializeSession() {
    setAuthChecking(true);
    setLoadError("");
    try {
      const session = await fetchAuthSession();
      setAuthSession(session);
      if (session.authenticated) {
        await refreshAll({ sessionOverride: session });
      } else {
        setLoadStatus("等待登录");
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLoadError(message);
      setLoadStatus("认证检查失败");
    } finally {
      setAuthChecking(false);
    }
  }

  async function runAction(key, action, successText) {
    setActionBusy((current) => ({ ...current, [key]: true }));
    setNotice({ type: "ok", text: "" });
    setLoadError("");
    try {
      const result = await action();
      if (successText) {
        setNotice({ type: "ok", text: successText });
      }
      await refreshAll({ silent: true });
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
      throw error;
    } finally {
      setActionBusy((current) => ({ ...current, [key]: false }));
    }
  }

  async function handleLogin() {
    setActionBusy((current) => ({ ...current, login: true }));
    setNotice({ type: "ok", text: "" });
    setLoadError("");
    try {
      const session = await loginWithPassword(loginForm);
      setAuthSession(session);
      setNotice({ type: "ok", text: `登录成功：${session.current_user?.display_name || session.current_user?.username || ""}` });
      await refreshAll({ sessionOverride: session });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, login: false }));
    }
  }

  async function handleLogout() {
    setActionBusy((current) => ({ ...current, logout: true }));
    setNotice({ type: "ok", text: "" });
    try {
      const session = await logoutSession();
      setAuthSession(session);
      setData(initialData);
      setPredictResult(null);
      setPredictSampleInfo(null);
      setPredictSampleCursor({});
      setTaskRoutingResult(null);
      setDeploymentPolicy(null);
      setDeploymentPolicyAutomation(null);
      setDeploymentPolicyAutomationForm(buildInitialDeploymentPolicyAutomationForm());
      setSelectedHealth(null);
      setSelectedLogs(null);
      setLoadStatus("已退出");
      setNotice({ type: "ok", text: "已退出当前账号。" });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, logout: false }));
    }
  }

  async function handleCreateDeployment(payloadOverride) {
    let gpuIds;
    try {
      gpuIds =
        payloadOverride?.gpu_ids ??
        parseGpuIdsText(payloadOverride?.gpuIdsText ?? deploymentForm.gpuIdsText);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
      return;
    }

    const useGpu = payloadOverride?.use_gpu ?? deploymentForm.useGpu;
    if (!useGpu) {
      gpuIds = [];
    }

    const requestedModelId = payloadOverride?.model_id || deploymentForm.modelId;
    const payload = {
      ...payloadOverride,
      model_id: requestedModelId,
      profile_name: resolveProfileName(
        requestedModelId,
        data.modelDetails,
        payloadOverride?.profile_name || deploymentForm.profileName,
      ),
      launch_runtime: payloadOverride?.launch_runtime ?? deploymentForm.launchRuntime,
      use_gpu: useGpu,
      gpu_ids: gpuIds,
      notes: payloadOverride?.notes ?? deploymentForm.notes ?? undefined,
    };
    await runAction("create-deployment", () => createDeployment(payload), `部署已创建：${payload.model_id}`);
  }

  async function handleImportModel() {
    if (!modelImportForm.archive) {
      setNotice({ type: "error", text: "请先选择一个模型压缩包。" });
      return;
    }
    if (!modelImportForm.modelFamily) {
      setNotice({ type: "error", text: "请先选择模型种类。" });
      return;
    }
    if (modelImportForm.importMode === "auto_package" && !modelImportForm.datasetName.trim()) {
      setNotice({ type: "error", text: "模式A需要填写数据集名称，平台才能生成统一命名。" });
      return;
    }
    setNotice({ type: "ok", text: "" });
    const task = buildModelTransferTask({
      taskId: `upload-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      direction: "upload",
      phase: "running",
      percent: 0,
      loadedBytes: 0,
      totalBytes: modelImportForm.archive.size || null,
      fileName: modelImportForm.archive.name,
      modelId: "",
      detail: "正在上传模型压缩包到 LuckyModels。",
    });
    addModelTransferTask(task);
    try {
      const result = await importModelArchive({
        archive: modelImportForm.archive,
        importMode: modelImportForm.importMode,
        modelFamily: modelImportForm.modelFamily,
        language: modelImportForm.language,
        datasetName: modelImportForm.datasetName.trim(),
        version: modelImportForm.version.trim() || "v1",
        modelName: modelImportForm.modelName.trim(),
        backbone: modelImportForm.backbone.trim(),
        alias: modelImportForm.alias.trim(),
        onUploadProgress: (progress) => {
          updateModelTransferTask(task.taskId, {
            direction: "upload",
            phase: "running",
            percent: progress.percent,
            loadedBytes: progress.loaded,
            totalBytes: progress.total,
            detail: progress.percent === 100
              ? "上传完成，正在由平台生成标准模型包。"
              : "正在上传模型压缩包到 LuckyModels。",
          });
        },
      });
      await refreshAll({ silent: true });
      setModelImportForm(buildInitialModelImportForm(data.importOptions));
      updateModelTransferTask(task.taskId, {
        direction: "upload",
        phase: "completed",
        percent: 100,
        loadedBytes: modelImportForm.archive.size || 0,
        totalBytes: modelImportForm.archive.size || null,
        fileName: modelImportForm.archive.name,
        modelId: result.model_id,
        detail: `模型制品已导入：${result.model_id}`,
      });
      scheduleModelTransferCleanup(task.taskId);
      setNotice({
        type: "ok",
        text: `模型制品已导入：${result.model_id}。命名规则为 模型家族-语言-数据集[-别名]-版本。`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updateModelTransferTask(task.taskId, {
        direction: "upload",
        phase: "failed",
        detail: message,
      });
      scheduleModelTransferCleanup(task.taskId);
      setNotice({ type: "error", text: message });
    }
  }

  async function handleDeploymentAction(actionName, deploymentId) {
    if (actionName === "stop") {
      await runAction(`stop-${deploymentId}`, () => stopDeployment(deploymentId), `Deployment 已停止：${deploymentId}`);
      return;
    }
    if (actionName === "restart") {
      await runAction(
        `restart-${deploymentId}`,
        () => restartDeployment(deploymentId),
        `Deployment 已重启：${deploymentId}`,
      );
      return;
    }
    if (actionName === "delete") {
      if (!window.confirm(`确认删除 deployment ${deploymentId} 吗？`)) {
        return;
      }
      await runAction(
        `delete-${deploymentId}`,
        () => deleteDeployment(deploymentId),
        `Deployment 已删除：${deploymentId}`,
      );
      setSelectedHealth((current) => (current?.deployment_id === deploymentId ? null : current));
      setSelectedLogs((current) => (current?.deployment_id === deploymentId ? null : current));
    }
  }

  async function handleInspectDeployment(target, deploymentId) {
    if (target === "health") {
      const result = await runAction(
        `health-${deploymentId}`,
        () => getDeploymentHealth(deploymentId),
        `已刷新 deployment 健康状态：${deploymentId}`,
      );
      setSelectedHealth(result);
      return;
    }
    if (target === "logs") {
      const result = await runAction(
        `logs-${deploymentId}`,
        () => getDeploymentLogs(deploymentId),
        `已加载 deployment 日志：${deploymentId}`,
      );
      setSelectedLogs(result);
    }
  }

  async function handleCreateBenchmark() {
    await runAction(
      "create-benchmark",
      () =>
        createBenchmark({
          deployment_id: benchmarkForm.deploymentId,
          text: benchmarkForm.text,
          label: benchmarkForm.label || undefined,
          iterations: Number(benchmarkForm.iterations),
          warmup_runs: Number(benchmarkForm.warmupRuns),
        }),
      "基准测试已创建。",
    );
  }

  async function handleDeleteBenchmark(benchmarkId) {
    if (!window.confirm(`确认删除 benchmark ${benchmarkId} 吗？`)) {
      return;
    }
    await runAction(
      `delete-benchmark-${benchmarkId}`,
      () => deleteBenchmark(benchmarkId),
      `Benchmark 已删除：${benchmarkId}`,
    );
  }

  async function handleDeletePredictionArchive(archiveId) {
    if (!window.confirm(`确认删除预测归档 ${archiveId} 吗？`)) {
      return;
    }
    await runAction(
      `delete-prediction-${archiveId}`,
      () => deletePredictionArchive(archiveId),
      `预测归档已删除：${archiveId}`,
    );
    setPredictResult((current) => (current?.archive_id === archiveId ? null : current));
  }

  async function handleUpdatePredictionArchiveFeedback(archiveId, feedbackStatus) {
    const nextStatus = normalizePredictionFeedbackStatus(feedbackStatus);
    const statusLabel = formatPredictionFeedbackLabel(nextStatus);
    const result = await runAction(
      `prediction-feedback-${archiveId}`,
      () =>
        updatePredictionArchiveFeedback(archiveId, {
          feedback_status: nextStatus,
        }),
      `预测归档反馈已更新为“${statusLabel}”。`,
    );
    setPredictResult((current) => {
      if (!current || current.archive_id !== archiveId) {
        return current;
      }
      return {
        ...current,
        feedback_status: result.feedback_status,
        feedback_note: result.feedback_note,
        feedback_updated_at: result.feedback_updated_at,
      };
    });
  }

  async function handlePredict() {
    const options = predictForm.optionsJson.trim() ? safeJsonParse(predictForm.optionsJson) : {};
    const result = await runAction(
      `predict-${predictForm.deploymentId}`,
      () =>
        predictWithDeployment(predictForm.deploymentId, {
          text: predictForm.text,
          scene: predictForm.scene || undefined,
          options,
        }),
      "在线推理已完成。",
    );
    setPredictResult(result);
  }

  function handleSelectPredictDeployment(deploymentId) {
    const targetDeployment = data.deployments.find((item) => item.deployment_id === deploymentId);
    const targetModel = data.models.find((item) => item.model_id === targetDeployment?.model_id);
    const nextScene = resolvePredictDatasetKey(targetModel?.dataset_name);
    setPredictForm((current) => ({
      ...current,
      deploymentId,
      scene: nextScene,
    }));
  }

  async function handleRefreshPredictSample() {
    await hydratePredictSample(predictForm.scene, { advance: true });
  }

  async function handleRunTaskRouting() {
    setActionBusy((current) => ({ ...current, "task-routing": true }));
    setNotice({ type: "ok", text: "" });
    try {
      const options = taskRoutingForm.optionsJson.trim() ? safeJsonParse(taskRoutingForm.optionsJson) : {};
      const normalizedScene = String(taskRoutingForm.scene || "").trim();
      const requestLanguage =
        normalizedScene === "en_general"
          ? "en"
          : normalizedScene === "zh_general" || normalizedScene === "zh_industrial"
            ? "zh"
            : taskRoutingForm.language === "auto"
              ? undefined
              : taskRoutingForm.language;
      const result = await runTaskRoutingStrategy({
        text: taskRoutingForm.text,
        objective: taskRoutingForm.objective,
        language: requestLanguage,
        scene: normalizedScene || undefined,
        max_models: Math.max(1, Math.min(5, Number(taskRoutingForm.maxModels) || 1)),
        options,
      });
      setTaskRoutingResult(result);
      setNotice({ type: "ok", text: "任务调度策略已执行。" });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, "task-routing": false }));
    }
  }

  async function handleLoadDeploymentPolicy() {
    setActionBusy((current) => ({ ...current, "deployment-policy": true }));
    setNotice({ type: "ok", text: "" });
    try {
      const [policyResult, automationResult] = await Promise.all([
        getDeploymentPolicy(),
        getDeploymentPolicyAutomation(),
      ]);
      setDeploymentPolicy(policyResult);
      setDeploymentPolicyAutomation(automationResult);
      setDeploymentPolicyAutomationForm({
        enabled: Boolean(automationResult.enabled),
        intervalSeconds: automationResult.interval_seconds || 900,
        protectActiveWindowSeconds: automationResult.protect_active_window_seconds || 120,
        gpuMemoryRestartEnabled: automationResult.gpu_memory_restart_enabled ?? true,
        gpuMemoryRestartIntervalSeconds: automationResult.gpu_memory_restart_interval_seconds || 60,
        gpuMemoryRestartThresholdMb: automationResult.gpu_memory_restart_threshold_mb || 5120,
      });
      setNotice({ type: "ok", text: "部署策略建议已刷新。" });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, "deployment-policy": false }));
    }
  }

  async function handleUpdateDeploymentPolicyAutomation(payloadOverride) {
    setActionBusy((current) => ({ ...current, "deployment-policy-automation": true }));
    setNotice({ type: "ok", text: "" });
    try {
      const payload = {
        enabled: payloadOverride?.enabled ?? deploymentPolicyAutomationForm.enabled,
        interval_seconds: Number(payloadOverride?.interval_seconds ?? deploymentPolicyAutomationForm.intervalSeconds),
        protect_active_window_seconds: Number(
          payloadOverride?.protect_active_window_seconds ?? deploymentPolicyAutomationForm.protectActiveWindowSeconds,
        ),
        gpu_memory_restart_enabled:
          payloadOverride?.gpu_memory_restart_enabled ?? deploymentPolicyAutomationForm.gpuMemoryRestartEnabled,
        gpu_memory_restart_interval_seconds: Number(
          payloadOverride?.gpu_memory_restart_interval_seconds ?? deploymentPolicyAutomationForm.gpuMemoryRestartIntervalSeconds,
        ),
        gpu_memory_restart_threshold_mb: Number(
          payloadOverride?.gpu_memory_restart_threshold_mb ?? deploymentPolicyAutomationForm.gpuMemoryRestartThresholdMb,
        ),
      };
      const result = await updateDeploymentPolicyAutomation(payload);
      setDeploymentPolicyAutomation(result);
      setDeploymentPolicyAutomationForm({
        enabled: Boolean(result.enabled),
        intervalSeconds: result.interval_seconds || payload.interval_seconds,
        protectActiveWindowSeconds: result.protect_active_window_seconds || payload.protect_active_window_seconds,
        gpuMemoryRestartEnabled: result.gpu_memory_restart_enabled ?? payload.gpu_memory_restart_enabled,
        gpuMemoryRestartIntervalSeconds:
          result.gpu_memory_restart_interval_seconds || payload.gpu_memory_restart_interval_seconds,
        gpuMemoryRestartThresholdMb: result.gpu_memory_restart_threshold_mb || payload.gpu_memory_restart_threshold_mb,
      });
      setNotice({ type: "ok", text: result.enabled ? "自动优化已开启。" : "自动优化已关闭。" });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, "deployment-policy-automation": false }));
    }
  }

  async function handleRunDeploymentPolicyAutomationNow() {
    setActionBusy((current) => ({ ...current, "deployment-policy-run-now": true }));
    setNotice({ type: "ok", text: "" });
    try {
      const [automationResult, policyResult] = await Promise.all([
        runDeploymentPolicyAutomationNow(),
        getDeploymentPolicy(),
      ]);
      setDeploymentPolicyAutomation(automationResult);
      setDeploymentPolicy(policyResult);
      setNotice({ type: "ok", text: "自动优化已执行一轮。" });
      await refreshAll({ silent: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, "deployment-policy-run-now": false }));
    }
  }

  async function handleCreateAccessUser(payload) {
    await runAction(
      "create-access-user",
      () =>
        createAccessUser({
          username: payload.username,
          display_name: payload.displayName,
          email: payload.email,
          password: payload.password || undefined,
          role_ids: payload.roleId ? [payload.roleId] : [],
          notes: payload.notes || undefined,
        }),
      `用户已创建：${payload.displayName || payload.username}`,
    );
  }

  async function handleUpdateAccessUser(userId, payload) {
    await runAction(
      `update-access-user-${userId}`,
      () =>
        updateAccessUser(userId, {
          display_name: payload.displayName,
          email: payload.email,
          role_ids: payload.roleIds,
          notes: payload.notes,
        }),
      `用户权限已更新：${userId}`,
    );
  }

  async function handleToggleAccessUser(userId, isActive) {
    await runAction(
      `toggle-access-user-${userId}`,
      () => toggleAccessUser(userId, { is_active: isActive }),
      `${isActive ? "已启用" : "已停用"}用户：${userId}`,
    );
  }

  async function handleSwitchCurrentAccessUser(userId) {
    await runAction(
      "switch-access-user",
      () => switchCurrentAccessUser({ user_id: userId }),
      `当前身份已切换：${userId}`,
    );
  }

  async function handleResetAccessUserPassword(userId, password) {
    await runAction(
      `reset-access-user-password-${userId}`,
      () => resetAccessUserPassword(userId, { password }),
      `密码已重置：${userId}`,
    );
  }

  async function handleExportModelManifest(modelId) {
    const detail = data.modelDetails[modelId];
    if (!detail) {
      setNotice({ type: "error", text: `未找到模型详情：${modelId}` });
      return;
    }
    downloadJsonFile(`${modelId}-manifest.json`, detail.manifest);
    setNotice({ type: "ok", text: `已导出模型 manifest：${modelId}` });
  }

  async function handleExportModelProfiles(modelId) {
    try {
      const profiles = await getModelProfiles(modelId);
      downloadJsonFile(`${modelId}-profiles.json`, {
        model_id: modelId,
        exported_at: new Date().toISOString(),
        profiles,
      });
      setNotice({ type: "ok", text: `已导出配置模板：${modelId}` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    }
  }

  async function handleExportModelPackage(modelId) {
    setActionBusy((current) => ({ ...current, [`export-model-${modelId}`]: true }));
    const task = buildModelTransferTask({
      taskId: `download-${modelId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      direction: "download",
      phase: "running",
      percent: 0,
      loadedBytes: 0,
      totalBytes: null,
      fileName: `${modelId}.zip`,
      modelId,
      detail: `正在导出模型制品：${modelId}`,
    });
    addModelTransferTask(task);
    try {
      const { blob, filename } = await exportModelPackage(modelId, (progress) => {
        updateModelTransferTask(task.taskId, {
          direction: "download",
          phase: "running",
          percent: progress.percent,
          loadedBytes: progress.loaded,
          totalBytes: progress.total,
          fileName: task.fileName || `${modelId}.zip`,
          modelId,
          detail: "正在从平台下载模型制品。",
        });
      });
      downloadBlobFile(filename || `${modelId}.zip`, blob);
      updateModelTransferTask(task.taskId, {
        direction: "download",
        phase: "completed",
        percent: 100,
        loadedBytes: blob.size || task.loadedBytes,
        totalBytes: blob.size || null,
        fileName: filename || `${modelId}.zip`,
        modelId,
        detail: `模型制品已导出：${modelId}`,
      });
      scheduleModelTransferCleanup(task.taskId);
      setNotice({ type: "ok", text: `已导出模型制品包：${modelId}` });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updateModelTransferTask(task.taskId, {
        direction: "download",
        phase: "failed",
        modelId,
        detail: message,
      });
      scheduleModelTransferCleanup(task.taskId);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, [`export-model-${modelId}`]: false }));
    }
  }

  async function handleDeleteModelPackage(modelId) {
    const detail = data.modelDetails[modelId];
    const originLabel = detail?.summary?.package_origin_label || "本机自带";
    if (!detail?.summary?.deletable) {
      setNotice({ type: "error", text: `当前模型制品不可删除：${modelId}（来源：${originLabel}）` });
      return;
    }
    if (!window.confirm(`确认删除模型制品 ${modelId} 吗？关联部署、基准和预测归档会一并删除。`)) {
      return;
    }
    const result = await runAction(
      `delete-model-${modelId}`,
      () => deleteModelPackage(modelId),
      `模型制品已删除：${modelId}`,
    );
    setPredictResult((current) => (current?.model_id === modelId ? null : current));
    setSelectedHealth(null);
    setSelectedLogs(null);
    if (result) {
      setNotice({
        type: "ok",
        text: `模型制品已删除：${modelId}，同步清理部署 ${result.deleted_deployment_count} 条、基准 ${result.deleted_benchmark_count} 条、预测归档 ${result.deleted_prediction_archive_count} 条。`,
      });
    }
  }

  const derived = useMemo(() => {
    const summary = data.summary || { model_count: 0, deployment_count: 0, benchmark_count: 0, user_count: 0, admin_count: 0 };
    const models = data.models || [];
    const modelDetails = data.modelDetails || {};
    const capabilityMatrix = data.capabilityMatrix || [];
    const deployments = data.deployments || [];
    const benchmarks = data.benchmarks || [];
    const predictArchives = data.predictArchives || [];
    const report = data.report || {};
    const gpuStatus = data.gpuStatus || { available: false, gpu_count: 0, devices: [] };
    const accessOverview = data.accessOverview || { users: [], roles: [], permissions: [], current_permissions: [] };
    const datasetCatalog = data.datasetCatalog || { datasets: DEFAULT_PREDICT_DATASETS, default_fallback_text: DEFAULT_PREDICT_TEXT };
    const taskRoutingFeedbackOverview = data.taskRoutingFeedbackOverview || {
      total_decisions: 0,
      total_feedback: 0,
      acceptance_rate: null,
      pending_feedback_count: 0,
      last_feedback_at: null,
      archive_feedback_total: 0,
      archive_satisfied_count: 0,
      archive_unsatisfied_count: 0,
      archive_pending_count: 0,
      objective_summaries: [],
      execution_path_summaries: [],
      model_adoption_summaries: [],
      weight_profiles: [],
      recent_decisions: [],
      insights: [],
    };
    const runningDeployments = deployments.filter((item) => item.status === "running");
    const predictableModels = models.filter((item) => item.predict_ready);
    const latestDeployment = deployments.length ? [...deployments].reverse()[0] : null;
    const latestBenchmark = benchmarks.length ? [...benchmarks].reverse()[0] : null;
    const latestPredictArchive = predictArchives.length ? sortByCreatedDesc(predictArchives)[0] : null;
    const deploymentProfiles = getProfilesForModel(deploymentForm.modelId, modelDetails);
    return {
      summary,
      report,
      models,
      modelDetails,
      capabilityMatrix,
      deployments,
      benchmarks,
      predictArchives,
      gpuStatus,
      accessOverview,
      datasetCatalog,
      taskRoutingFeedbackOverview,
      accessUsers: accessOverview.users || [],
      accessRoles: accessOverview.roles || [],
      accessPermissions: accessOverview.permissions || [],
      currentAccessUser: accessOverview.current_user || authSession?.current_user || null,
      currentAccessPermissions:
        (accessOverview.current_permissions && accessOverview.current_permissions.length
          ? accessOverview.current_permissions
          : authSession?.current_permissions) || [],
      runningDeployments,
      predictableModels,
      latestDeployment,
      latestBenchmark,
      latestPredictArchive,
      deploymentProfiles,
    };
  }, [authSession, data, deploymentForm.modelId]);

  if (authChecking) {
    return <LoginPage actionBusy={actionBusy} authChecking={authChecking} loadError={loadError} loginForm={loginForm} notice={notice} onLogin={() => void handleLogin()} onLoginFormChange={setLoginForm} />;
  }

  if (!authSession?.authenticated) {
    return <LoginPage actionBusy={actionBusy} authChecking={false} loadError={loadError} loginForm={loginForm} notice={notice} onLogin={() => void handleLogin()} onLoginFormChange={setLoginForm} />;
  }

  return (
    <Shell
      actionBusy={actionBusy}
      authSession={authSession}
      benchmarkForm={benchmarkForm}
      data={data}
      deploymentForm={deploymentForm}
      derived={derived}
      loadError={loadError}
      loadStatus={loadStatus}
      notice={notice}
      onBenchmarkFormChange={setBenchmarkForm}
      onCreateBenchmark={handleCreateBenchmark}
      onCreateDeployment={handleCreateDeployment}
      onDeleteBenchmark={handleDeleteBenchmark}
      onDeletePredictionArchive={handleDeletePredictionArchive}
      onUpdatePredictionArchiveFeedback={handleUpdatePredictionArchiveFeedback}
      onDeploymentAction={handleDeploymentAction}
      onDeploymentFormChange={setDeploymentForm}
      onExportDeploymentRows={(rows, format) => {
        if (format === "json") {
          downloadJsonFile("luckymodels-deployments.json", rows);
          return;
        }
        downloadCsvFile("luckymodels-deployments.csv", rows);
      }}
      onExportModelManifest={handleExportModelManifest}
      onExportModelPackage={handleExportModelPackage}
      onExportModelProfiles={handleExportModelProfiles}
      onDeleteModelPackage={handleDeleteModelPackage}
      onExportPredictResult={(value) =>
        downloadJsonFile(
          `${value?.model_id || "predict"}-${value?.archive_id || "result"}.json`,
          value,
        )
      }
      onImportModel={handleImportModel}
      onCreateAccessUser={handleCreateAccessUser}
      onLoadDeploymentPolicy={handleLoadDeploymentPolicy}
      onLogout={handleLogout}
      onSwitchCurrentAccessUser={handleSwitchCurrentAccessUser}
      onToggleAccessUser={handleToggleAccessUser}
      onUpdateAccessUser={handleUpdateAccessUser}
      onInspectDeployment={handleInspectDeployment}
      onPredict={handlePredict}
      onRefreshPredictSample={handleRefreshPredictSample}
      onSelectPredictDeployment={handleSelectPredictDeployment}
      onPredictFormChange={setPredictForm}
      onRefresh={refreshAll}
      onResetAccessUserPassword={handleResetAccessUserPassword}
      onRunTaskRouting={handleRunTaskRouting}
      onTaskRoutingFormChange={setTaskRoutingForm}
      predictForm={predictForm}
      predictSampleInfo={predictSampleInfo}
      predictResult={predictResult}
      modelImportForm={modelImportForm}
      modelTransferItems={modelTransferItems}
      onDismissModelTransfer={removeModelTransferTask}
      onModelImportFormChange={setModelImportForm}
      deploymentPolicy={deploymentPolicy}
      deploymentPolicyAutomation={deploymentPolicyAutomation}
      deploymentPolicyAutomationForm={deploymentPolicyAutomationForm}
      onRunDeploymentPolicyAutomationNow={handleRunDeploymentPolicyAutomationNow}
      onUpdateDeploymentPolicyAutomation={handleUpdateDeploymentPolicyAutomation}
      onDeploymentPolicyAutomationFormChange={setDeploymentPolicyAutomationForm}
      selectedHealth={selectedHealth}
      selectedLogs={selectedLogs}
      taskRoutingForm={taskRoutingForm}
      taskRoutingResult={taskRoutingResult}
    />
  );
}

function LoginPage({ actionBusy, authChecking, loadError, loginForm, notice, onLogin, onLoginFormChange }) {
  return (
    <main className="login-shell">
      <section className="login-stage">
        <article className="login-hero-panel">
          <div className="hero-kicker">LuckyModels Platform</div>
          <h1>模型部署、在线推理与实验归档的一体化控制台</h1>
          <p className="hero-desc">
            这不是一个演示性质的静态页面，而是面向知识抽取模型资产、运行实例、基准实验和权限治理的真实平台入口。
          </p>
          <div className="login-badge-grid">
            <div className="login-metric">
              <strong>部署控制</strong>
              <span>统一管理模型上线、重启、停止与资源分配。</span>
            </div>
            <div className="login-metric">
              <strong>实验闭环</strong>
              <span>在线推理、基准结果、归档导出在同一平台完成。</span>
            </div>
            <div className="login-metric">
              <strong>权限治理</strong>
              <span>管理员与普通用户分层访问，菜单和页面按权限收口。</span>
            </div>
          </div>
          <div className="login-callout">
            <div className="login-callout-title">演示账号</div>
            <div className="account-preset-grid">
              <button
                className="account-preset"
                onClick={() => onLoginFormChange({ username: "admin", password: "admin" })}
                type="button"
              >
                <strong>管理员体验</strong>
                <span>admin / admin</span>
              </button>
              <button
                className="account-preset"
                onClick={() => onLoginFormChange({ username: "user", password: "user" })}
                type="button"
              >
                <strong>普通用户体验</strong>
                <span>user / user</span>
              </button>
            </div>
          </div>
        </article>

        <section className="login-panel">
          <div className="brand-line">
            <div className="brand-mark">LM</div>
            <div>
              <strong>LuckyModels</strong>
              <span>Platform Access Gateway</span>
            </div>
          </div>
          <div className="login-panel-head">
            <h2>平台登录</h2>
            <p>请输入账号密码进入控制台。页面菜单和管理员能力会按当前身份自动收口。</p>
          </div>
          {notice.text ? (
            <div className={`banner ${notice.type === "error" ? "banner-error" : "banner-ok"}`}>{notice.text}</div>
          ) : null}
          {loadError ? <div className="banner banner-error">{loadError}</div> : null}
          <form
            className="login-form"
            onSubmit={(event) => {
              event.preventDefault();
              onLogin();
            }}
          >
            <label className="stack-field">
              <span>用户名</span>
              <input
                autoComplete="username"
                onChange={(event) => onLoginFormChange((current) => ({ ...current, username: event.target.value }))}
                placeholder="请输入用户名"
                type="text"
                value={loginForm.username}
              />
            </label>
            <label className="stack-field">
              <span>密码</span>
              <input
                autoComplete="current-password"
                onChange={(event) => onLoginFormChange((current) => ({ ...current, password: event.target.value }))}
                placeholder="请输入密码"
                type="password"
                value={loginForm.password}
              />
            </label>
            <div className="login-actions">
              <button
                className="btn-primary"
                disabled={authChecking || actionBusy.login || !loginForm.username.trim() || !loginForm.password.trim()}
                type="submit"
              >
                {authChecking ? "检查会话中" : actionBusy.login ? "登录中" : "进入平台"}
              </button>
            </div>
          </form>
          <div className="platform-bullet-list">
            <span className="tag">访问控制已启用</span>
            <span className="tag">管理员支持重置密码</span>
            <span className="tag">普通用户禁止进入管理员面板</span>
          </div>
        </section>
      </section>
    </main>
  );
}

function GuardedPage({ allowed, children, fallbackPath, title, note }) {
  if (allowed) {
    return children;
  }
  return <AccessDeniedPage fallbackPath={fallbackPath} note={note} title={title} />;
}

function AccessDeniedPage({ fallbackPath, note, title }) {
  return (
    <section className="page-shell card section-shell">
      <div className="access-denied-shell">
        <div className="access-denied-icon">403</div>
        <h2>{title || "当前页面不可访问"}</h2>
        <p>{note || "当前账号没有访问该页面的权限，请联系管理员或切换为具备权限的账号。"}</p>
        <div className="actions">
          <NavLink className="btn-linklike btn-primary" to={fallbackPath || "/"}>
            返回可访问页面
          </NavLink>
        </div>
      </div>
    </section>
  );
}

function Shell(props) {
  const location = useLocation();
  const currentMeta = pageMeta[location.pathname] || pageMeta["/"];
  const {
    actionBusy,
    authSession,
    benchmarkForm,
    data,
    deploymentForm,
    derived,
    loadError,
    loadStatus,
    notice,
    onBenchmarkFormChange,
    onCreateBenchmark,
    onCreateDeployment,
    onDeleteBenchmark,
    onDeletePredictionArchive,
    onUpdatePredictionArchiveFeedback,
    onDeploymentAction,
    onDeploymentFormChange,
    onExportDeploymentRows,
    onExportModelManifest,
    onExportModelPackage,
    onExportModelProfiles,
    onDeleteModelPackage,
    onExportPredictResult,
    onImportModel,
    onCreateAccessUser,
    onLoadDeploymentPolicy,
    onRunDeploymentPolicyAutomationNow,
    onUpdateDeploymentPolicyAutomation,
    onDeploymentPolicyAutomationFormChange,
    onLogout,
    onSwitchCurrentAccessUser,
    onToggleAccessUser,
    onUpdateAccessUser,
    onInspectDeployment,
    onPredict,
    onRefreshPredictSample,
    onSelectPredictDeployment,
    onPredictFormChange,
    onRefresh,
    onResetAccessUserPassword,
    onRunTaskRouting,
    onTaskRoutingFormChange,
    predictForm,
    predictSampleInfo,
    predictResult,
    modelImportForm,
    modelTransferItems,
    onDismissModelTransfer,
    onModelImportFormChange,
    deploymentPolicy,
    deploymentPolicyAutomation,
    deploymentPolicyAutomationForm,
    selectedHealth,
    selectedLogs,
    taskRoutingForm,
    taskRoutingResult,
  } = props;
  const currentPermissions = derived.currentAccessPermissions || [];
  const currentUser = derived.currentAccessUser || authSession?.current_user || null;
  const visibleNavGroups = getVisibleNavGroups(currentPermissions, currentUser);
  const fallbackPath = getFallbackPath(currentPermissions, currentUser);
  const currentRoleNames = (currentUser?.role_ids || []).map((item) => formatRoleLabel(item)).join(" / ") || "未分配角色";

  function renderGuardedRoute(path, element, options = {}) {
    return (
      <Route
        element={
          <GuardedPage
            allowed={canAccessPath(path, currentPermissions, currentUser)}
            fallbackPath={fallbackPath}
            note={options.note}
            title={options.title}
          >
            {element}
          </GuardedPage>
        }
        key={path}
        path={path}
      />
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-card">
          <div className="brand">
            <div className="brand-mark">LM</div>
            <div>
              <strong>LuckyModels</strong>
              <span>知识抽取模型库控制台</span>
            </div>
          </div>
          <div className="brand-note">
            平台当前已具备部署、推理、实验归档和访问控制能力，并支持按权限展示页面级菜单。
          </div>
        </div>

        <div className="session-card">
          <div className="session-kicker">当前会话</div>
          <div className="session-user">{currentUser?.display_name || currentUser?.username || "未登录"}</div>
          <div className="session-role">{currentRoleNames}</div>
          <div className="session-grid">
            <div>
              <strong>{currentPermissions.length}</strong>
              <span>权限项</span>
            </div>
            <div>
              <strong>{derived.summary.user_count || 0}</strong>
              <span>平台用户</span>
            </div>
          </div>
        </div>

        {visibleNavGroups.map((group) => (
          <div className="nav-group" key={group.title}>
            <div className="nav-title">{group.title}</div>
            {group.items.map((item) => (
              <NavLink
                className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
                end={item.path === "/"}
                key={item.id}
                to={item.path}
              >
                <span className="nav-item-icon">{item.icon || "•"}</span>
                <span className="nav-item-copy">
                  <strong>{item.label}</strong>
                  <small>{item.description || "页面入口"}</small>
                </span>
              </NavLink>
            ))}
          </div>
        ))}

        <div className="nav-group">
          <div className="nav-title">当前状态</div>
          <div className="footer-note">模型 {derived.summary.model_count}</div>
          <div className="footer-note">部署 {derived.summary.deployment_count}</div>
          <div className="footer-note">基准 {derived.summary.benchmark_count}</div>
          <div className="footer-note">用户 {derived.summary.user_count || 0}</div>
          <div className="footer-note mono">127.0.0.1:38080</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-copy">
            <div className="topbar-kicker">LuckyModels Control Center</div>
            <h1>{currentMeta.title}</h1>
            <div className="topbar-note">{currentMeta.note}</div>
          </div>
          <div className="topbar-actions">
            <span className="hero-pill">当前身份：{currentUser?.username || "未登录"}</span>
            <span className="hero-pill">角色：{currentRoleNames}</span>
            <span className="hero-pill">状态：{loadStatus}</span>
            <button disabled={actionBusy.logout} onClick={() => void onLogout()} type="button">
              {actionBusy.logout ? "退出中" : "退出"}
            </button>
            <div className="top-icon">LM</div>
          </div>
        </header>

        <section className="content">
          {notice.text ? (
            <div className={`banner ${notice.type === "error" ? "banner-error" : "banner-ok"}`}>{notice.text}</div>
          ) : null}
          {loadError ? <div className="banner banner-error">{loadError}</div> : null}

          <Routes>
            {renderGuardedRoute("/", <OverviewPage derived={derived} />)}
            {renderGuardedRoute("/dashboard", <OverviewPage derived={derived} />)}
            {renderGuardedRoute(
              "/experiments",
              <ExperimentsPage
                actionBusy={actionBusy}
                derived={derived}
                onDeletePredictionArchive={onDeletePredictionArchive}
                onExportPredictResult={onExportPredictResult}
                onUpdatePredictionArchiveFeedback={onUpdatePredictionArchiveFeedback}
                onPredict={onPredict}
                onRefreshPredictSample={onRefreshPredictSample}
                onSelectPredictDeployment={onSelectPredictDeployment}
                onPredictFormChange={onPredictFormChange}
                onRefresh={onRefresh}
                predictForm={predictForm}
                predictSampleInfo={predictSampleInfo}
                predictResult={predictResult}
                selectedHealth={selectedHealth}
                selectedLogs={selectedLogs}
              />,
            )}
            {renderGuardedRoute(
              "/task-routing",
              <TaskRoutingPage
                actionBusy={actionBusy}
                derived={derived}
                onRunTaskRouting={onRunTaskRouting}
                onTaskRoutingFormChange={onTaskRoutingFormChange}
                taskRoutingForm={taskRoutingForm}
                taskRoutingResult={taskRoutingResult}
              />,
            )}
            {renderGuardedRoute(
              "/deployment-policy",
              <DeploymentPolicyPage
                actionBusy={actionBusy}
                derived={derived}
                deploymentPolicy={deploymentPolicy}
                deploymentPolicyAutomation={deploymentPolicyAutomation}
                deploymentPolicyAutomationForm={deploymentPolicyAutomationForm}
                onCreateDeployment={onCreateDeployment}
                onLoadDeploymentPolicy={onLoadDeploymentPolicy}
                onRunDeploymentPolicyAutomationNow={onRunDeploymentPolicyAutomationNow}
                onUpdateDeploymentPolicyAutomation={onUpdateDeploymentPolicyAutomation}
                onDeploymentPolicyAutomationFormChange={onDeploymentPolicyAutomationFormChange}
              />,
            )}
            {renderGuardedRoute(
              "/access-control",
              <AccessControlPage
                actionBusy={actionBusy}
                derived={derived}
                onSwitchCurrentAccessUser={onSwitchCurrentAccessUser}
              />,
              {
                title: "当前账号不可访问用户权限页",
                note: "该页面仅对具备用户权限查看能力的账号开放。",
              },
            )}
            {renderGuardedRoute(
              "/admin-console",
              <AdminConsolePage
                actionBusy={actionBusy}
                derived={derived}
                onCreateAccessUser={onCreateAccessUser}
                onResetAccessUserPassword={onResetAccessUserPassword}
                onToggleAccessUser={onToggleAccessUser}
                onUpdateAccessUser={onUpdateAccessUser}
              />,
              {
                title: "普通用户禁止进入管理员面板",
                note: "当前页面仅对具备 users.manage 权限的管理员开放。",
              },
            )}
            {renderGuardedRoute(
              "/archive",
              <ArchivePage
                derived={derived}
                onDeletePredictionArchive={onDeletePredictionArchive}
                onExportPredictionArchive={(archive) => onExportPredictResult(archive)}
                onUpdatePredictionArchiveFeedback={onUpdatePredictionArchiveFeedback}
                actionBusy={actionBusy}
              />,
            )}
            {renderGuardedRoute("/capability-matrix", <CapabilityMatrixPage derived={derived} />)}
            {renderGuardedRoute(
              "/models",
              <ModelsPage
                actionBusy={actionBusy}
                data={data}
                deploymentForm={deploymentForm}
                deploymentPolicy={deploymentPolicy}
                derived={derived}
                onCreateDeployment={onCreateDeployment}
                onDeploymentFormChange={onDeploymentFormChange}
                modelImportForm={modelImportForm}
                modelTransferItems={modelTransferItems}
                onDismissModelTransfer={onDismissModelTransfer}
                onDeleteModelPackage={onDeleteModelPackage}
                onExportModelPackage={onExportModelPackage}
                onImportModel={onImportModel}
                onModelImportFormChange={onModelImportFormChange}
              />,
            )}
            {renderGuardedRoute(
              "/artifacts",
              <ArtifactsPage
                data={data}
                actionBusy={actionBusy}
                onDeleteModelPackage={onDeleteModelPackage}
                onExportModelManifest={onExportModelManifest}
                onExportModelProfiles={onExportModelProfiles}
              />,
            )}
            {renderGuardedRoute(
              "/deployments",
              <DeploymentsPage
                actionBusy={actionBusy}
                deploymentForm={deploymentForm}
                derived={derived}
                onCreateDeployment={onCreateDeployment}
                onDeploymentAction={onDeploymentAction}
                onDeploymentFormChange={onDeploymentFormChange}
                onExportDeploymentRows={onExportDeploymentRows}
                onInspectDeployment={onInspectDeployment}
                selectedHealth={selectedHealth}
                selectedLogs={selectedLogs}
              />,
            )}
            {renderGuardedRoute(
              "/benchmarks",
              <BenchmarksPage
                actionBusy={actionBusy}
                benchmarkForm={benchmarkForm}
                derived={derived}
                onBenchmarkFormChange={onBenchmarkFormChange}
                onCreateBenchmark={onCreateBenchmark}
                onDeleteBenchmark={onDeleteBenchmark}
              />,
            )}
            {renderGuardedRoute("/resources", <ResourcesPage derived={derived} />)}
            {renderGuardedRoute("/logs", <LogsPage derived={derived} selectedLogs={selectedLogs} />)}
            <Route element={<Navigate replace to={fallbackPath} />} path="*" />
          </Routes>
        </section>
      </main>
    </div>
  );
}

function OverviewPage({ derived }) {
  return (
    <>
      <section className="summary-strip page-shell">
        <div className="summary-card">
          <div className="title">模型总数</div>
          <div className="big">{derived.summary.model_count}</div>
          <div className="small">已登记模型包</div>
        </div>
        <div className="summary-card">
          <div className="title">可推理模型</div>
          <div className="big">{derived.predictableModels.length}</div>
          <div className="small">已完成在线推理适配</div>
        </div>
        <div className="summary-card">
          <div className="title">运行中部署</div>
          <div className="big">{derived.runningDeployments.length}</div>
          <div className="small">当前活跃推理实例</div>
        </div>
        <div className="summary-card">
          <div className="title">基准总数</div>
          <div className="big">{derived.summary.benchmark_count}</div>
          <div className="small">历史基准测试记录</div>
        </div>
        <div className="summary-card">
          <div className="title">平台用户</div>
          <div className="big">{derived.summary.user_count || 0}</div>
          <div className="small">管理员 {derived.summary.admin_count || 0} 人</div>
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="近期动态" title="最近状态" />
          <div className="metrics">
            <MetricCard title="最新部署" value={derived.latestDeployment?.deployment_id || "-"} note={derived.latestDeployment?.model_id || "暂无"} />
            <MetricCard title="部署状态" value={formatStatusLabel(derived.latestDeployment?.status || "-")} note={derived.latestDeployment?.service_url || "-"} />
            <MetricCard title="最新基准" value={derived.latestBenchmark?.benchmark_id || "-"} note={derived.latestBenchmark?.model_id || "暂无"} />
            <MetricCard title="CasRel 报告" value={formatStatusLabel(derived.report.status || "empty")} note="/api/v1/demo/casrel/report" />
          </div>
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="模型状态" title="模型运行口径" />
          <div className="tag-list">
            {derived.models.map((model) => (
              <span className="tag" key={model.model_id}>
                {model.model_id} · {formatStatusLabel(model.runtime_status)}
              </span>
            ))}
          </div>
        </article>
      </section>
    </>
  );
}

function PredictionGraphCard({ predictResult }) {
  const [nodeLimit, setNodeLimit] = useState(20);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState(null);

  const graphData = useMemo(() => {
    const triples = Array.isArray(predictResult?.triples) ? predictResult.triples : [];
    if (!triples.length) {
      return { nodes: [], edges: [], totalNodeCount: 0 };
    }

    const nodes = [];
    const nodeIndex = new Map();
    const edges = [];

    for (const triple of triples) {
      const subject = String(triple?.subject || "").trim();
      const object = String(triple?.object || "").trim();
      const predicate = String(triple?.predicate || "").trim();
      if (!subject || !object || !predicate) {
        continue;
      }

      for (const label of [subject, object]) {
        if (!nodeIndex.has(label)) {
          nodeIndex.set(label, nodes.length);
          nodes.push({ id: label, label });
        }
      }

      edges.push({
        source: subject,
        target: object,
        predicate,
        confidence: triple?.confidence,
      });
    }

    const limitedNodes = nodes.slice(0, Math.max(1, nodeLimit));
    const allowed = new Set(limitedNodes.map((item) => item.id));
    const limitedEdges = edges.filter((item) => allowed.has(item.source) && allowed.has(item.target));

    const centerX = 340;
    const centerY = 220;
    const radius = Math.max(110, Math.min(190, 30 + limitedNodes.length * 7));
    const positionedNodes = limitedNodes.map((item, index) => {
      const angle = (Math.PI * 2 * index) / Math.max(1, limitedNodes.length);
      return {
        ...item,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });

    const positionMap = new Map(positionedNodes.map((item) => [item.id, item]));
    const positionedEdges = limitedEdges
      .map((item) => ({
        ...item,
        sourceNode: positionMap.get(item.source),
        targetNode: positionMap.get(item.target),
      }))
      .filter((item) => item.sourceNode && item.targetNode);

    return {
      nodes: positionedNodes,
      edges: positionedEdges,
      totalNodeCount: nodes.length,
    };
  }, [nodeLimit, predictResult]);

  function handleWheel(event) {
    event.preventDefault();
    setZoom((current) => {
      const next = current + (event.deltaY < 0 ? 0.1 : -0.1);
      return Math.max(0.5, Math.min(2.4, Number(next.toFixed(2))));
    });
  }

  function handleDragStart(event) {
    setDragStart({
      x: event.clientX - pan.x,
      y: event.clientY - pan.y,
    });
  }

  function handleDragMove(event) {
    if (!dragStart) {
      return;
    }
    setPan({
      x: event.clientX - dragStart.x,
      y: event.clientY - dragStart.y,
    });
  }

  function handleDragEnd() {
    setDragStart(null);
  }

  return (
    <article className="card detail-card">
      <SectionHeader
        kicker="关系图谱"
        title="图谱视图"
        note="根据本次预测结果自动生成小图谱，支持滚轮缩放、拖拽平移与节点数量调节。"
      />
      {graphData.nodes.length ? (
        <>
          <div className="graph-toolbar">
            <label className="graph-slider">
              <span>最多节点数：{nodeLimit}</span>
              <input
                max={Math.max(20, Math.min(80, graphData.totalNodeCount))}
                min="5"
                onChange={(event) => setNodeLimit(Number(event.target.value))}
                type="range"
                value={Math.min(nodeLimit, Math.max(20, Math.min(80, graphData.totalNodeCount)))}
              />
            </label>
            <div className="actions">
              <button onClick={() => setZoom((current) => Math.max(0.5, Number((current - 0.1).toFixed(2))))} type="button">缩小</button>
              <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} type="button">重置视图</button>
              <button onClick={() => setZoom((current) => Math.min(2.4, Number((current + 0.1).toFixed(2))))} type="button">放大</button>
            </div>
          </div>
          <div
            className="graph-shell"
            onMouseDown={handleDragStart}
            onMouseLeave={handleDragEnd}
            onMouseMove={handleDragMove}
            onMouseUp={handleDragEnd}
            onWheel={handleWheel}
            role="presentation"
          >
            <svg className="graph-svg" viewBox="0 0 680 440">
              <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
                {graphData.edges.map((edge, index) => {
                  const midX = (edge.sourceNode.x + edge.targetNode.x) / 2;
                  const midY = (edge.sourceNode.y + edge.targetNode.y) / 2;
                  return (
                    <g key={`${edge.source}-${edge.target}-${edge.predicate}-${index + 1}`}>
                      <line
                        className="graph-edge"
                        x1={edge.sourceNode.x}
                        x2={edge.targetNode.x}
                        y1={edge.sourceNode.y}
                        y2={edge.targetNode.y}
                      />
                      <text className="graph-edge-label" x={midX} y={midY - 6}>
                        {edge.predicate}
                      </text>
                    </g>
                  );
                })}
                {graphData.nodes.map((node) => (
                  <g key={node.id} transform={`translate(${node.x} ${node.y})`}>
                    <circle className="graph-node" r="24" />
                    <text className="graph-node-label" textAnchor="middle" y="4">
                      {node.label.length > 10 ? `${node.label.slice(0, 10)}…` : node.label}
                    </text>
                  </g>
                ))}
              </g>
            </svg>
          </div>
          <div className="detail-note">
            当前展示 {graphData.nodes.length} / {graphData.totalNodeCount} 个节点，滚轮可缩放，按住图谱区域可拖拽平移。
          </div>
        </>
      ) : (
        <EmptyState icon="G" title="还没有可展示的图谱" note="先完成一次在线推理，平台会根据三元组自动绘制关系图谱。" />
      )}
    </article>
  );
}

function PredictionArchiveFeedbackControls({ actionBusy, archive, onUpdate }) {
  const currentStatus = normalizePredictionFeedbackStatus(archive?.feedback_status);
  const options = [
    { value: "satisfied", label: "满意" },
    { value: "unsatisfied", label: "不满意" },
    { value: "pending", label: "暂定" },
  ];

  return (
    <div className="actions archive-feedback-actions">
      {options.map((item) => (
        <button
          key={item.value}
          className={currentStatus === item.value ? "button-active" : ""}
          disabled={actionBusy?.[`prediction-feedback-${archive.archive_id}`]}
          onClick={() => void onUpdate(archive.archive_id, item.value)}
          type="button"
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function ExperimentsPage({
  actionBusy,
  derived,
  onDeletePredictionArchive,
  onExportPredictResult,
  onUpdatePredictionArchiveFeedback,
  onPredict,
  onRefreshPredictSample,
  onSelectPredictDeployment,
  onPredictFormChange,
  predictForm,
  predictSampleInfo,
  predictResult,
  selectedHealth,
  selectedLogs,
}) {
  const datasetOptions = derived.datasetCatalog?.datasets?.length
    ? derived.datasetCatalog.datasets
    : DEFAULT_PREDICT_DATASETS;
  const activeScene = resolvePredictDatasetKey(predictForm.scene);
  const sampleBusy = actionBusy[`predict-sample-${activeScene}`];
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageArchives = derived.currentAccessUser?.is_admin;
  const canExportArchives = currentPermissions.includes("archive.export");
  const currentArchive = predictResult?.archive_id
    ? derived.predictArchives.find((item) => item.archive_id === predictResult.archive_id) || null
    : null;

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="在线预测"
          title="在线推理"
          note="从平台里选择运行中的部署实例，直接通过控制面代理接口完成预测。"
          right={
            <div className="actions">
              <button disabled={sampleBusy} onClick={() => void onRefreshPredictSample()} type="button">
                {sampleBusy ? "刷新中" : "刷新数据"}
              </button>
            </div>
          }
        />
        {derived.runningDeployments.length ? (
          <div className="form-grid">
            <label className="stack-field">
              <span>选择部署实例</span>
              <select
                onChange={(event) => onSelectPredictDeployment(event.target.value)}
                value={predictForm.deploymentId}
              >
                {derived.runningDeployments.map((item) => (
                  <option key={item.deployment_id} value={item.deployment_id}>
                    {item.deployment_id} · {item.model_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="stack-field">
              <span>数据集场景</span>
              <select
                onChange={(event) => onPredictFormChange((current) => ({ ...current, scene: event.target.value }))}
                value={activeScene}
              >
                {datasetOptions.map((item) => (
                  <option key={item.dataset_key} value={item.dataset_key}>
                    {item.dataset_name}
                    {item.available ? "" : "（未迁移）"}
                  </option>
                ))}
              </select>
            </label>
            <label className="stack-field stack-field-wide">
              <span>输入文本</span>
              <textarea
                onChange={(event) => onPredictFormChange((current) => ({ ...current, text: event.target.value }))}
                rows="4"
                value={predictForm.text}
              />
              <small className="detail-note">
                {predictSampleInfo?.is_fallback
                  ? `当前数据集样本不可用，已回退到固定默认文本。${predictSampleInfo?.detail ? `原因：${predictSampleInfo.detail}` : ""}`
                  : predictSampleInfo?.case_id
                    ? `当前样本：${predictSampleInfo.case_id} · 来源：${predictSampleInfo.dataset_source}`
                    : "选择数据集后，平台会自动填充对应样本文本。"}
              </small>
            </label>
            <label className="stack-field stack-field-wide">
              <span>推理参数 JSON</span>
              <textarea
                onChange={(event) => onPredictFormChange((current) => ({ ...current, optionsJson: event.target.value }))}
                rows="4"
                value={predictForm.optionsJson}
              />
            </label>
            <div className="actions">
              <button
                className="btn-primary"
                disabled={actionBusy[`predict-${predictForm.deploymentId}`]}
                onClick={() => void onPredict()}
                type="button"
              >
                运行预测
              </button>
            </div>
          </div>
        ) : (
          <EmptyState icon="▶" title="暂无运行中的部署实例" note="先去“部署记录”或“模型库”里创建并启动一个部署实例。" />
        )}
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader
            kicker="预测结果"
            title="预测结果"
            right={
              predictResult ? (
                <div className="actions">
                  <button onClick={() => onExportPredictResult(predictResult)} type="button">
                    导出本次 JSON
                  </button>
                </div>
              ) : null
            }
          />
          {predictResult ? (
            <>
              <div className="metrics">
                <MetricCard title="模型" value={predictResult.model_id} note={predictResult.model_name} />
                <MetricCard title="延迟" value={predictResult.latency_ms} note="毫秒" />
                <MetricCard title="三元组" value={predictResult.triple_count} note={formatLanguageLabel(predictResult.language)} />
                <MetricCard title="版本" value={predictResult.version} note="运行时输出" />
                <MetricCard title="归档编号" value={predictResult.archive_id || "-"} note="预测结果已自动归档" />
                <MetricCard
                  title="当前反馈"
                  value={formatPredictionFeedbackLabel(currentArchive?.feedback_status)}
                  note={currentArchive?.feedback_updated_at || "默认暂定，可在下方结果归档中修改"}
                />
              </div>
              {currentArchive ? (
                <div className="banner banner-muted">
                  当前预测归档反馈：{formatPredictionFeedbackLabel(currentArchive.feedback_status)}。如果这次结果符合预期，可直接标记“满意”；明显不符合预期则标记“不满意”；暂时不判断就保留“暂定”。
                </div>
              ) : null}
              <div className="relation-list">
                {(predictResult.triples || []).map((item, index) => (
                  <div className="relation-item" key={`${item.subject}-${item.predicate}-${item.object}-${index + 1}`}>
                    {item.subject}
                    {" -> "}
                    {item.predicate}
                    {" -> "}
                    {item.object}
                    <br />
                    置信度：{item.confidence}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <EmptyState icon="T" title="还没有预测结果" note="从上方选择一个运行中的部署实例并提交文本。" />
          )}
        </article>

        <PredictionGraphCard predictResult={predictResult} />
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="预测归档"
          title="单次预测归档"
          note="每次在线推理成功后，平台会自动写入归档，方便后续导出 JSON、截图、回填论文案例，并补充满意度反馈。"
        />
        {derived.predictArchives.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>归档编号</th>
                  <th>模型</th>
                  <th>配置模板</th>
                  <th>输入摘要</th>
                  <th>结果摘要</th>
                  <th>反馈</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {sortByCreatedDesc(derived.predictArchives)
                  .slice(0, 8)
                  .map((item) => (
                    <tr key={item.archive_id}>
                      <td>
                        <strong>{item.archive_id}</strong>
                        <div className="table-note">{item.created_at}</div>
                      </td>
                      <td>{item.model_id}</td>
                      <td>{item.profile_name}</td>
                      <td>
                        <div className="table-note">{item.scene || "-"}</div>
                        <div className="table-note">{item.request_text || item.request_texts?.[0] || "-"}</div>
                      </td>
                      <td>
                        <div className="table-note">延迟：{escapeValue(item.latency_ms)}</div>
                        <div className="table-note">三元组：{escapeValue(item.triple_count)}</div>
                        <div className="table-note">样本数：{escapeValue(item.item_count)}</div>
                      </td>
                      <td>
                        <div>{formatPredictionFeedbackLabel(item.feedback_status)}</div>
                        <div className="table-note">{item.feedback_updated_at || "未补充反馈"}</div>
                        <PredictionArchiveFeedbackControls
                          actionBusy={actionBusy}
                          archive={item}
                          onUpdate={onUpdatePredictionArchiveFeedback}
                        />
                      </td>
                      <td>
                        <div className="actions row-actions">
                          {canExportArchives ? (
                            <button onClick={() => onExportPredictResult(item.response_payload || item)} type="button">导出结果</button>
                          ) : null}
                          {canManageArchives ? (
                            <button
                              disabled={actionBusy[`delete-prediction-${item.archive_id}`]}
                              onClick={() => void onDeletePredictionArchive(item.archive_id)}
                              type="button"
                            >
                              删除归档
                            </button>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon="A" title="暂无预测归档" note="完成一次在线推理后，结果会自动归档到这里。" />
        )}
      </section>
    </>
  );
}

function AccessControlPage({ actionBusy, derived, onSwitchCurrentAccessUser }) {
  const currentUser = derived.currentAccessUser;
  const users = derived.accessUsers || [];
  const roles = derived.accessRoles || [];
  const permissions = derived.accessPermissions || [];
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageUsers = currentPermissions.includes("users.manage");
  const [selectedUserId, setSelectedUserId] = useState(currentUser?.user_id || "");

  useEffect(() => {
    setSelectedUserId(currentUser?.user_id || "");
  }, [currentUser?.user_id]);

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="访问控制"
          title="用户权限总览"
          note="这里集中展示当前登录身份、角色绑定和平台内置权限，用于论文或答辩中说明平台治理能力。"
          right={
            <div className="actions">
              <button
                className="btn-primary"
                disabled={!canManageUsers || !selectedUserId || actionBusy["switch-access-user"]}
                onClick={() => void onSwitchCurrentAccessUser(selectedUserId)}
                type="button"
              >
                {actionBusy["switch-access-user"] ? "切换中" : "切换当前身份"}
              </button>
            </div>
          }
        />
        <div className="form-grid">
          <label className="stack-field">
            <span>查看身份</span>
            <select onChange={(event) => setSelectedUserId(event.target.value)} value={selectedUserId}>
              {users.map((item) => (
                <option key={item.user_id} value={item.user_id}>
                  {item.display_name} · {item.username}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>当前角色</span>
            <div className="stack-field-readonly">
              {currentUser?.role_ids?.length ? currentUser.role_ids.map((item) => formatRoleLabel(item)).join(" / ") : "-"}
            </div>
          </label>
          <label className="stack-field">
            <span>当前邮箱</span>
            <div className="stack-field-readonly">{currentUser?.email || "-"}</div>
          </label>
          <label className="stack-field">
            <span>状态</span>
            <div className="stack-field-readonly">
              {currentUser ? `${currentUser.is_active ? "启用" : "停用"} / ${currentUser.is_admin ? "管理员" : "普通用户"}` : "-"}
            </div>
          </label>
        </div>
        <div className="tag-list">
          {currentPermissions.length ? (
            currentPermissions.map((item) => (
              <span className="tag" key={item}>
                {formatPermissionLabel(item)}
              </span>
            ))
          ) : (
            <span className="tag">当前身份没有有效权限</span>
          )}
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="权限构成" title="角色与权限映射" note="每个角色都绑定了一组固定权限，用于描述平台中的读写边界。" />
        <div className="list-grid">
          {roles.map((role) => (
            <article className="list-card" key={role.role_id}>
              <h3>{role.role_name}</h3>
              <div className="table-note">{role.description}</div>
              <div className="tag-list">
                {(role.permissions || []).map((permissionId) => (
                  <span className="tag" key={`${role.role_id}-${permissionId}`}>
                    {formatPermissionLabel(permissionId)}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="用户列表" title="用户与权限矩阵" note="管理员可在这里查看所有账号的角色归属和启停状态。" />
        <div className="metrics">
          <MetricCard title="用户总数" value={users.length} note="当前 access store 中的用户数" />
          <MetricCard title="管理员数" value={users.filter((item) => item.is_admin && item.is_active).length} note="启用中的管理员" />
          <MetricCard title="启用用户" value={users.filter((item) => item.is_active).length} note="当前可切换身份" />
          <MetricCard title="权限项" value={permissions.length} note="平台当前内置权限项数量" />
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>用户</th>
                <th>角色</th>
                <th>权限摘要</th>
                <th>状态</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {users.map((item) => {
                const effectiveRoleNames = (item.role_ids || []).map((roleId) => formatRoleLabel(roleId)).join(" / ");
                const effectivePermissionCount = roles
                  .filter((role) => (item.role_ids || []).includes(role.role_id))
                  .reduce((sum, role) => sum + (role.permissions || []).length, 0);
                return (
                  <tr key={item.user_id}>
                    <td>
                      <strong>{item.display_name}</strong>
                      <div className="mono">{item.username}</div>
                      <div className="table-note">{item.email}</div>
                    </td>
                    <td>{effectiveRoleNames || "-"}</td>
                    <td>{effectivePermissionCount} 项</td>
                    <td>
                      <StatusPill status={item.is_active ? "completed" : "stopped"} />
                      <div className="table-note">{item.is_admin ? "管理员" : "普通用户"}</div>
                    </td>
                    <td className="limitation-cell">{item.notes || "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function AdminConsolePage({
  actionBusy,
  derived,
  onCreateAccessUser,
  onResetAccessUserPassword,
  onToggleAccessUser,
  onUpdateAccessUser,
}) {
  const roles = derived.accessRoles || [];
  const users = derived.accessUsers || [];
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageUsers = currentPermissions.includes("users.manage");
  const [userForm, setUserForm] = useState(buildInitialAccessUserForm(roles));
  const [draftRoles, setDraftRoles] = useState({});
  const [passwordDrafts, setPasswordDrafts] = useState({});

  useEffect(() => {
    setUserForm((current) => {
      if (current.roleId && roles.some((item) => item.role_id === current.roleId)) {
        return current;
      }
      return buildInitialAccessUserForm(roles);
    });
  }, [roles]);

  useEffect(() => {
    setDraftRoles(
      Object.fromEntries(users.map((item) => [item.user_id, item.role_ids?.[0] || "user"])),
    );
  }, [users]);

  useEffect(() => {
    setPasswordDrafts((current) =>
      Object.fromEntries(users.map((item) => [item.user_id, current[item.user_id] || ""])),
    );
  }, [users]);

  async function handleCreate() {
    await onCreateAccessUser(userForm);
    setUserForm(buildInitialAccessUserForm(roles));
  }

  async function handleResetPassword(userId) {
    const password = (passwordDrafts[userId] || "").trim();
    if (!password) {
      return;
    }
    await onResetAccessUserPassword(userId, password);
    setPasswordDrafts((current) => ({ ...current, [userId]: "" }));
  }

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="管理员入口"
          title="管理员控制台"
          note="这里聚焦账号创建、角色调整、启停控制和密码重置，是管理员专用的治理视图。"
        />
        <div className="metrics">
          <MetricCard title="当前可管理" value={canManageUsers ? "是" : "否"} note="是否具有 users.manage 权限" />
          <MetricCard title="平台管理员" value={users.filter((item) => item.is_admin).length} note="包含停用用户" />
          <MetricCard title="可切换身份" value={users.filter((item) => item.is_active).length} note="启用中的账号数" />
          <MetricCard title="角色模板" value={roles.length} note="当前平台内置角色数" />
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="新建用户" title="创建用户账号" note="默认采用单角色分配，适合平台演示和课程答辩中的权限说明。" />
        <div className="form-grid">
          <label className="stack-field">
            <span>用户名</span>
            <input
              onChange={(event) => setUserForm((current) => ({ ...current, username: event.target.value }))}
              type="text"
              value={userForm.username}
            />
          </label>
          <label className="stack-field">
            <span>显示名称</span>
            <input
              onChange={(event) => setUserForm((current) => ({ ...current, displayName: event.target.value }))}
              type="text"
              value={userForm.displayName}
            />
          </label>
          <label className="stack-field">
            <span>邮箱</span>
            <input
              onChange={(event) => setUserForm((current) => ({ ...current, email: event.target.value }))}
              type="email"
              value={userForm.email}
            />
          </label>
          <label className="stack-field">
            <span>初始密码</span>
            <input
              onChange={(event) => setUserForm((current) => ({ ...current, password: event.target.value }))}
              type="password"
              value={userForm.password}
            />
          </label>
          <label className="stack-field">
            <span>角色</span>
            <select
              onChange={(event) => setUserForm((current) => ({ ...current, roleId: event.target.value }))}
              value={userForm.roleId}
            >
              {roles.map((item) => (
                <option key={item.role_id} value={item.role_id}>
                  {item.role_name}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field stack-field-wide">
            <span>备注</span>
            <textarea
              onChange={(event) => setUserForm((current) => ({ ...current, notes: event.target.value }))}
              rows="3"
              value={userForm.notes}
            />
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={
                !canManageUsers ||
                actionBusy["create-access-user"] ||
                !userForm.username.trim() ||
                !userForm.displayName.trim() ||
                !userForm.email.trim() ||
                !userForm.password.trim()
              }
              onClick={() => void handleCreate()}
              type="button"
            >
              {actionBusy["create-access-user"] ? "创建中" : "创建用户"}
            </button>
          </div>
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="账号治理" title="角色调整、账号启停与密码重置" note="管理员可直接调整单个用户的角色，启用或停用账号，并为用户重置登录密码。" />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>用户</th>
                <th>当前角色</th>
                <th>调整角色</th>
                <th>状态</th>
                <th>重置密码</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {users.map((item) => (
                <tr key={item.user_id}>
                  <td>
                    <strong>{item.display_name}</strong>
                    <div className="mono">{item.user_id}</div>
                    <div className="table-note">{item.email}</div>
                  </td>
                  <td>{(item.role_ids || []).map((roleId) => formatRoleLabel(roleId)).join(" / ") || "-"}</td>
                  <td>
                    <select
                      disabled={!canManageUsers}
                      onChange={(event) =>
                        setDraftRoles((current) => ({ ...current, [item.user_id]: event.target.value }))
                      }
                      value={draftRoles[item.user_id] || item.role_ids?.[0] || "user"}
                    >
                      {roles.map((role) => (
                        <option key={role.role_id} value={role.role_id}>
                          {role.role_name}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <StatusPill status={item.is_active ? "completed" : "stopped"} />
                    <div className="table-note">{item.is_admin ? "管理员" : "普通用户"}</div>
                  </td>
                  <td>
                    <div className="table-inline-editor">
                      <input
                        className="table-input"
                        disabled={!canManageUsers}
                        onChange={(event) =>
                          setPasswordDrafts((current) => ({ ...current, [item.user_id]: event.target.value }))
                        }
                        placeholder="输入新密码"
                        type="password"
                        value={passwordDrafts[item.user_id] || ""}
                      />
                      <div className="table-note">密码长度至少 4 位</div>
                    </div>
                  </td>
                  <td>
                    <div className="actions row-actions">
                      <button
                        disabled={!canManageUsers || actionBusy[`update-access-user-${item.user_id}`]}
                        onClick={() =>
                          void onUpdateAccessUser(item.user_id, {
                            displayName: item.display_name,
                            email: item.email,
                            roleIds: [draftRoles[item.user_id] || item.role_ids?.[0] || "user"],
                            notes: item.notes,
                          })
                        }
                        type="button"
                      >
                        保存角色
                      </button>
                      <button
                        className="btn-primary"
                        disabled={
                          !canManageUsers ||
                          actionBusy[`reset-access-user-password-${item.user_id}`] ||
                          !(passwordDrafts[item.user_id] || "").trim()
                        }
                        onClick={() => void handleResetPassword(item.user_id)}
                        type="button"
                      >
                        {actionBusy[`reset-access-user-password-${item.user_id}`] ? "重置中" : "重置密码"}
                      </button>
                      <button
                        disabled={!canManageUsers || actionBusy[`toggle-access-user-${item.user_id}`]}
                        onClick={() => void onToggleAccessUser(item.user_id, !item.is_active)}
                        type="button"
                      >
                        {item.is_active ? "停用" : "启用"}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function TaskRoutingPage({
  actionBusy,
  derived,
  onRunTaskRouting,
  onTaskRoutingFormChange,
  taskRoutingForm,
  taskRoutingResult,
}) {
  const candidateOptions = [
    { value: "accuracy", label: "高准确", note: "优先稳定性与一致性投票，适合论文展示与知识库入库前校验。" },
    { value: "coverage", label: "高覆盖", note: "优先召回更多三元组，适合线索发现与候选补全。" },
    { value: "latency", label: "低时延", note: "优先低 P95 部署，适合在线交互快速返回。" },
    { value: "balanced", label: "综合平衡", note: "在准确、覆盖、时延和成本之间做折中。" },
    { value: "resource", label: "低成本", note: "优先低资源占用路径，适合 GPU 紧张场景。" },
    { value: "domain", label: "领域优先", note: "优先选择与输入领域更匹配的部署并交叉校验。" },
  ];
  const feedbackOverview = derived.taskRoutingFeedbackOverview || {
    total_decisions: 0,
    total_feedback: 0,
    acceptance_rate: null,
    pending_feedback_count: 0,
    last_feedback_at: null,
    archive_feedback_total: 0,
    archive_satisfied_count: 0,
    archive_unsatisfied_count: 0,
    archive_pending_count: 0,
    objective_summaries: [],
    execution_path_summaries: [],
    model_adoption_summaries: [],
    weight_profiles: [],
    recent_decisions: [],
    insights: [],
  };
  const sceneOptions = [
    { value: "en_general", label: "英文通用场景", note: "只在英文 deployment 中筛选，优先英文模型。" },
    { value: "zh_general", label: "中文通用场景", note: "只在中文通用 deployment 中筛选，优先使用 DuIE 一类通用中文模型。" },
    { value: "zh_industrial", label: "中文工业设备场景", note: "优先使用带 equipmentmix15k 标识的工业专用模型；若通用中文模型历史表现更强，也允许参与调度。" },
  ];
  const completedBenchmarks = sortByCreatedDesc(derived.benchmarks).filter((item) => item.status === "completed");
  const benchmarkedModelCount = new Set(completedBenchmarks.map((item) => item.model_id)).size;
  const calibrationArchiveCount = derived.predictArchives.length;
  const recentBenchmarkRows = completedBenchmarks.slice(0, 6);
  const selectedObjective = candidateOptions.find((item) => item.value === taskRoutingForm.objective) || candidateOptions[0];
  const selectedScene = sceneOptions.find((item) => item.value === taskRoutingForm.scene) || sceneOptions[2];
  const selectedDeployments = taskRoutingResult?.selected_deployments || [];
  const fusedTriples = taskRoutingResult?.fused_result?.triples || [];
  const topWeightEntry = Object.entries(taskRoutingResult?.objective_weights || {}).sort((a, b) => b[1] - a[1])[0];
  const executionPathLabels = {
    none: "未执行",
    single_low_latency: "单模型低时延直出",
    parallel_multi_union: "多模型并行补全",
    single_low_cost: "单模型低成本直出",
    cost_aware_primary_plus_reviewer: "低成本主模型 + 复核模型",
    domain_priority_single: "领域优先单模型",
    domain_priority_crosscheck: "领域优先交叉校验",
    single_balanced: "单模型综合平衡",
    consensus_with_backfill: "共识优先并补全",
    single_high_confidence: "单模型高置信直出",
    multi_consensus_vote: "多模型共识投票",
  };
  const fusionPolicyLabels = {
    empty: "无",
    single_primary: "主模型直出",
    union_all: "并集融合",
    primary_with_fallback: "主模型 + 补充结果",
    domain_priority: "领域优先融合",
    consensus_backfill: "共识补全",
    majority_vote: "多数投票",
  };
  const strategyModeLabels = {
    none: "未执行",
    "multi-objective-adaptive": "多目标自适应",
  };
  function formatExecutionPathLabel(value) {
    return executionPathLabels[String(value || "").trim()] || value || "-";
  }
  function formatFusionPolicyLabel(value) {
    return fusionPolicyLabels[String(value || "").trim()] || value || "-";
  }
  function formatStrategyModeLabel(value) {
    return strategyModeLabels[String(value || "").trim()] || value || "-";
  }
  const quickStartSteps = [
    {
      title: "1. 选目标",
      desc: "先告诉平台这次更看重什么，例如准确、覆盖、时延或领域匹配。",
    },
    {
      title: "2. 填文本",
      desc: "先选场景，再贴入待抽取文本。场景会直接限制候选 deployment 的范围。",
    },
    {
      title: "3. 运行调度",
      desc: "平台会只在当前运行中的 deployment 里自动选 1 到多个模型，必要时做交叉验证或融合。",
    },
    {
      title: "4. 看结果",
      desc: "先看最终三元组，再看平台为什么这么选，以及背后的 benchmark 和历史归档证据。",
    },
  ];
  const pageReadingHints = [
    "先看“调度结果”：这里直接回答平台最终选了什么执行路径，融合出了哪些三元组。",
    "再看“候选部署与执行明细”：这里解释为什么选中这些 deployment，而不是别的 deployment。",
    "接着看“运行证据与基准概览”：这里展示最近 benchmark 如何支撑本次调度选择。",
    "最后看“策略校准总览”和“权重快照”：这里说明调度层如何根据 benchmark 与归档记录做轻量自校准。",
    "如果要截图或追溯历史，直接看“最近决策记录”：一行就是一次真实任务调度。",
  ];

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="创新点一"
          title="任务级模型调度策略"
          note="这一页负责“先选模型，再做推理”。平台会综合任务目标、文本难度、领域特征、历史 benchmark、在线归档与资源成本，在单模型、交叉验证和多模型融合路径之间动态切换。"
          right={
            <div className="actions">
              <button
                className="btn-primary"
                disabled={actionBusy["task-routing"]}
                onClick={() => void onRunTaskRouting()}
                type="button"
              >
                {actionBusy["task-routing"] ? "执行中" : "运行任务调度"}
              </button>
            </div>
          }
        />

        <div className="metrics">
          <MetricCard title="运行中实例" value={derived.runningDeployments.length} note="当前真正可参与调度的 deployment 数量" />
          <MetricCard title="已完成基准" value={completedBenchmarks.length} note={`覆盖模型 ${benchmarkedModelCount} 个`} />
          <MetricCard title="在线归档" value={calibrationArchiveCount} note="历史推理归档会参与调度层自校准" />
          <MetricCard title="当前目标" value={selectedObjective.label} note={selectedObjective.note} />
        </div>

        <div className="task-routing-guide-grid">
          <div className="task-routing-step-grid">
            {quickStartSteps.map((item) => (
              <div className="task-routing-step-card" key={item.title}>
                <strong>{item.title}</strong>
                <span>{item.desc}</span>
              </div>
            ))}
          </div>
          <div className="task-routing-note-card">
            <div className="section-kicker">页面怎么看</div>
            <div className="task-routing-note-list">
              {pageReadingHints.map((item, index) => (
                <div className="relation-item" key={`${item}-${index + 1}`}>
                  {item}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="banner banner-muted">
          快速理解：在线推理页是“指定某个 deployment 直接跑”；任务调度页是“让平台先决定该调用哪些 deployment，再统一融合输出结果”。
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="操作区" title="运行参数" note="这是你每次真正需要填写的区域。一般只改目标偏好、场景标识、最大模型数和输入文本即可。" />
          <div className="banner banner-muted">
            快速上手建议：工业设备文本优先用“领域优先 + 中文工业设备场景”；普通中文文本优先用“综合平衡 + 中文通用场景”；想尽快返回就选“低时延”。
          </div>
          <div className="form-grid">
            <label className="stack-field">
              <span>目标偏好</span>
              <select
                onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, objective: event.target.value }))}
                value={taskRoutingForm.objective}
              >
                {candidateOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
              <small className="detail-note">{selectedObjective.note}</small>
            </label>
            <label className="stack-field">
              <span>最大模型数</span>
              <input
                min="1"
                max="5"
                onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, maxModels: event.target.value }))}
                type="number"
                value={taskRoutingForm.maxModels}
              />
              <small className="detail-note">表示这次任务最多允许多少个模型参与。它是“上限”，不是一定会选满；低时延场景通常仍只会选 1 个。</small>
            </label>
            <label className="stack-field">
              <span>场景标识</span>
              <select
                onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, scene: event.target.value }))}
                value={taskRoutingForm.scene}
              >
                {sceneOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
              <small className="detail-note">{selectedScene.note} 这个字段不是备注，而是真会过滤候选 deployment。</small>
            </label>
            <label className="stack-field">
              <span>模型范围</span>
              <div className="stack-field-readonly">
                {selectedScene.value === "en_general" ? "英文 deployment" : selectedScene.value === "zh_general" ? "中文通用 deployment（DuIE 路线）" : "中文工业 deployment（IndustrialEquipmentMix 路线）"}
              </div>
              <small className="detail-note">这里会随场景自动切换，不再单独手选语言，避免英文/中文模型混用，也避免工业模型和通用模型乱混。</small>
            </label>
            <label className="stack-field stack-field-wide">
              <span>输入文本</span>
              <textarea
                onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, text: event.target.value }))}
                rows="5"
                value={taskRoutingForm.text}
              />
            </label>
            <label className="stack-field stack-field-wide">
              <span>附加参数 JSON</span>
              <textarea
                onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, optionsJson: event.target.value }))}
                rows="3"
                value={taskRoutingForm.optionsJson}
              />
              <small className="detail-note">通常可以留空。只有你想临时传入额外控制项时再填写；如果不确定，直接留空即可。</small>
            </label>
          </div>
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="解读区" title="关键概念说明" note="如果你第一次看这页，先读这一块。它会告诉你页面里几个最容易混淆的词分别是什么意思。" />
          <div className="relation-list">
            <div className="relation-item">
              <strong>目标偏好</strong>
              <br />
              决定调度更偏向准确、覆盖、时延还是成本。它不是模型参数，而是调度层策略。
            </div>
            <div className="relation-item">
              <strong>场景标识</strong>
              <br />
              现在固定为 3 类：英文通用、中文通用、中文工业设备。它会直接约束候选 deployment 的范围，而不只是提示词。
            </div>
            <div className="relation-item">
              <strong>执行路径</strong>
              <br />
              指这次任务最终采用了什么方式完成，例如单模型直出、多模型交叉验证、候选融合等。
            </div>
            <div className="relation-item">
              <strong>候选部署</strong>
              <br />
              指当前运行中的 deployment。调度只会在这些已启动实例里选择，不会凭空调用未运行模型。
            </div>
            <div className="relation-item">
              <strong>自动归档</strong>
              <br />
              指每次调度的输入、选中的模型、执行路径和结果都会被记录下来，供后续统计与论文展示使用。
            </div>
            <div className="relation-item">
              <strong>自校准</strong>
              <br />
              指平台会参考 benchmark 与在线归档，调整调度评分权重；不会改动底层 ONNX 模型参数。
            </div>
          </div>
        </article>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="结果区" title="调度结果" note="建议先看这一块。这里最直接回答三个问题：平台选了什么路径、为什么这么选、最终输出了什么三元组。" />
          {taskRoutingResult ? (
            <>
              <div className="metrics">
                <MetricCard title="目标" value={selectedObjective.label} note={formatStrategyModeLabel(taskRoutingResult.strategy_mode)} />
                <MetricCard title="语言" value={taskRoutingResult.inferred_language} note={`请求语言：${taskRoutingResult.requested_language || "自动"}`} />
                <MetricCard title="任务难度" value={taskRoutingResult.difficulty} note={`候选 ${taskRoutingResult.candidate_count} 个，选中 ${taskRoutingResult.selected_count} 个`} />
                <MetricCard title="执行路径" value={formatExecutionPathLabel(taskRoutingResult.execution_path)} note={formatFusionPolicyLabel(taskRoutingResult.fusion_policy) || "任务级调度输出"} />
              </div>
              <div className="banner banner-muted">{taskRoutingResult.explanation}</div>
              <div className="metrics">
                <MetricCard title="领域" value={taskRoutingResult.context_features?.domain || "-"} note={`实体线索 ${taskRoutingResult.context_features?.entity_hint_count || 0}`} />
                <MetricCard title="文本长度" value={taskRoutingResult.context_features?.text_length || "-"} note={`子句数 ${taskRoutingResult.context_features?.clause_count || 0}`} />
                <MetricCard title="融合模式" value={formatFusionPolicyLabel(taskRoutingResult.fused_result?.fusion_mode)} note={`三元组 ${taskRoutingResult.fused_result?.triple_count || 0}`} />
                <MetricCard title="权重主项" value={topWeightEntry?.[0] || "-"} note="当前目标下影响最大的评分项" />
              </div>
              <div className="tag-list">
                {Object.entries(taskRoutingResult.objective_weights || {}).map(([key, value]) => (
                  <span className="tag" key={key}>
                    {key}: {Number(value).toFixed(2)}
                  </span>
                ))}
              </div>
              {(taskRoutingResult.adaptive_adjustments && Object.keys(taskRoutingResult.adaptive_adjustments).length > 0) ? (
                <div className="tag-list">
                  {Object.entries(taskRoutingResult.adaptive_adjustments || {}).map(([key, value]) => (
                    <span className="tag tag-emphasis" key={key}>
                      调整 {key}: {Number(value).toFixed(2)}
                    </span>
                  ))}
                </div>
              ) : null}
              {(taskRoutingResult.adaptation_reasons || []).length ? (
                <div className="logs">{(taskRoutingResult.adaptation_reasons || []).join("\n")}</div>
              ) : null}
              <div className="banner banner-muted">最终三元组列表：这是任务调度层融合后的统一输出，不再区分来自哪个底层模型。</div>
              <div className="relation-list">
                {fusedTriples.map((item, index) => (
                  <div className="relation-item" key={`${item.subject}-${item.predicate}-${item.object}-${index + 1}`}>
                    {item.subject}
                    {" -> "}
                    {item.predicate}
                    {" -> "}
                    {item.object}
                    <br />
                    票数：{item.vote_count || 1}
                    {" · "}
                    置信度：{item.confidence ?? "-"}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <EmptyState icon="S" title="还没有策略结果" note="填写文本后运行一次任务调度，这里会显示最终融合后的三元组。" />
          )}
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="解释区" title="候选部署与执行明细" note="这一表专门回答“为什么选中这些 deployment”。论文里它可以支撑任务调度不是拍脑袋，而是有评分依据。" />
          {taskRoutingResult ? (
            <>
              <div className="banner banner-muted">
                这张表看什么：看平台为什么选中这些 deployment，没有选中谁。重点看 `score`、`得分拆解`、`领域匹配` 和最后一列 `说明`。
              </div>
              <div className="banner banner-muted">
                术语提示：`得分拆解` 代表各 deployment 在准确、覆盖、时延等维度的评分贡献；`领域匹配` 表示它与当前文本领域的贴合度；`资源级别` 表示其运行成本或资源档位。
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>部署</th>
                      <th>模型</th>
                      <th>得分拆解</th>
                      <th>领域匹配</th>
                      <th>资源级别</th>
                      <th>说明</th>
                    </tr>
                  </thead>
                  <tbody>
                    {taskRoutingResult.selected_deployments.map((item) => (
                      <tr key={item.deployment_id}>
                        <td>
                          <strong>{item.deployment_id}</strong>
                          <div className="table-note">{item.profile_name}</div>
                        </td>
                        <td>
                          <div>{item.model_id}</div>
                          <div className="table-note">score={item.selection_score ?? "-"}</div>
                        </td>
                        <td>
                          <div className="table-note">acc {item.score_breakdown?.accuracy ?? "-"}</div>
                          <div className="table-note">cov {item.score_breakdown?.coverage ?? "-"}</div>
                          <div className="table-note">lat {item.score_breakdown?.latency ?? "-"}</div>
                        </td>
                        <td>{item.domain_match || "-"}</td>
                        <td>{item.resource_cost_level || "-"}</td>
                        <td className="limitation-cell">{item.selection_reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {(taskRoutingResult.decision_trace || []).length ? <div className="logs">{(taskRoutingResult.decision_trace || []).join("\n")}</div> : null}
            </>
          ) : (
            <div className="logs">运行后这里会给出候选排序、实际选中的 deployment，以及各模型的执行结果摘要。</div>
          )}
        </article>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="证据区" title="本次调度会沉淀哪些证据" note="真实业务里外露 API 通常不会回传“最终采用了哪个模型”，因此平台默认依赖自动归档；如果管理员或使用者愿意，也可以在结果归档里补充满意度反馈。" />
          {taskRoutingResult?.decision_id ? (
            <>
              <div className="metrics">
                <MetricCard title="决策编号" value={taskRoutingResult.decision_id} note="本次调度会进入历史记录，后面可检索和截图" />
                <MetricCard title="执行模型数" value={(taskRoutingResult.execution_results || []).length} note="本次真正参与运行的模型实例数" />
                <MetricCard title="融合三元组" value={taskRoutingResult.fused_result?.triple_count || 0} note="会进入在线归档，参与后续统计" />
                <MetricCard title="归档方式" value="自动归档 + 可选反馈" note="默认自动留痕，必要时可在结果归档补满意度" />
              </div>
              <div className="relation-list">
                {[
                  "任务调度记录：保存这次任务的目标偏好、执行路径、候选部署与最终选中的模型。",
                  "预测归档记录：保存这次任务输出的三元组数量、运行情况，以及可选的满意 / 不满意 / 暂定反馈。",
                  "调度层自校准：平台后续会优先参考 benchmark、在线归档和已补充的满意度反馈，轻量修正不同目标的调度评分权重。",
                ].map((item, index) => (
                  <div className="relation-item" key={`${item}-${index + 1}`}>
                    {item}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <EmptyState icon="F" title="还没有归档样本" note="先运行一次任务调度，系统才会生成这次任务的决策编号和归档记录。" />
          )}
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="证据区" title="运行证据与基准概览" note="这张表主要看最近有哪些 benchmark 在支撑调度层决策，尤其适合论文里引用 P95、成功率和 profile 信息。" />
          <div className="metrics">
            <MetricCard title="已完成 Benchmark" value={completedBenchmarks.length} note={`覆盖模型 ${benchmarkedModelCount} 个`} />
            <MetricCard title="在线归档" value={calibrationArchiveCount} note="作为运行证据参与策略校准" />
            <MetricCard title="当前场景" value={selectedScene.label} note="证据筛选会优先参考这一场景下的记录" />
            <MetricCard title="最近证据更新时间" value={completedBenchmarks[0]?.created_at || feedbackOverview.last_feedback_at || "-"} note="优先展示最近一条可引用记录" />
          </div>
          <div className="banner banner-muted">这张表看什么：看平台最近拿什么 benchmark 结果当作调度证据。重点看 `P95`、`成功率` 和 `Profile`。</div>
          <div className="banner banner-muted">如果某个模型或 profile 最近有稳定 benchmark，它在低时延或高稳定场景中通常会更容易被调度层优先考虑。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Benchmark</th>
                  <th>模型</th>
                  <th>Profile</th>
                  <th>P95</th>
                  <th>成功率</th>
                  <th>时间</th>
                </tr>
              </thead>
              <tbody>
                {recentBenchmarkRows.map((item) => (
                  <tr key={item.benchmark_id}>
                    <td>
                      <strong>{item.label || item.benchmark_id}</strong>
                      <div className="table-note">{item.benchmark_id}</div>
                    </td>
                    <td>{item.model_id}</td>
                    <td>{item.profile_name}</td>
                    <td>{item.p95_latency_ms ? `${Number(item.p95_latency_ms).toFixed(3)} ms` : "-"}</td>
                    <td>{formatPercent(item.total_requests ? item.success_count / item.total_requests : null)}</td>
                    <td>{item.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!recentBenchmarkRows.length ? (
            <div className="logs">当前还没有可引用的 benchmark 记录。建议先针对 CasRel、PURE、SpERT 等运行中的部署补几条固定样本 benchmark，再回来截图。</div>
          ) : null}
        </article>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="分析区" title="策略校准总览" note="这块主要回答“平台过去更常以什么目标运行、已经形成了多少历史调度记录”，它是调度层证据链的总览表。" />
          <div className="metrics">
            <MetricCard title="决策总数" value={feedbackOverview.total_decisions || 0} note="累计归档的任务调度记录" />
            <MetricCard title="Benchmark 证据" value={completedBenchmarks.length} note="已完成的 benchmark 记录数" />
            <MetricCard title="在线归档证据" value={calibrationArchiveCount} note="实时推理沉淀的归档记录数" />
            <MetricCard title="满意反馈" value={feedbackOverview.archive_satisfied_count || 0} note="结果归档里被标记为满意的记录数" />
            <MetricCard title="不满意反馈" value={feedbackOverview.archive_unsatisfied_count || 0} note="结果归档里被标记为不满意的记录数" />
            <MetricCard title="最近决策时间" value={feedbackOverview.recent_decisions?.[0]?.created_at || "-"} note="最近一条任务调度历史" />
          </div>
          <div className="relation-list">
            {(feedbackOverview.insights || []).map((item, index) => (
              <div className="relation-item" key={`${item}-${index + 1}`}>
                {item}
              </div>
            ))}
          </div>
          <div className="banner banner-muted">这张表看什么：看不同“目标偏好”下，平台已经积累了多少历史调度记录。它更像调度行为统计，不是模型性能榜单。</div>
          <div className="banner banner-muted">你可以在这里看出平台在哪些任务目标下已经有较多历史样本，以及常见的主参考执行路径是什么。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>目标</th>
                  <th>决策数</th>
                  <th>补充记录</th>
                  <th>主参考路径</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {(feedbackOverview.objective_summaries || []).map((item) => (
                  <tr key={item.objective}>
                    <td>{item.objective}</td>
                    <td>{item.decision_count}</td>
                    <td>{item.feedback_count}</td>
                    <td>{formatExecutionPathLabel(item.best_execution_path)}</td>
                    <td className="limitation-cell">
                      {item.decision_count > 0
                        ? `该目标已有 ${item.decision_count} 条历史调度，可直接作为权重更新的参考窗口。`
                        : "当前还没有该目标的历史调度记录。"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="分析区" title="调度评分权重快照" note="这张表不是模型训练权重，而是调度层评分权重。它说明不同目标下，平台更看重准确、覆盖、时延、领域还是资源成本。" />
          <div className="banner banner-muted">这张表看什么：看调度层“现在按什么规则打分”。如果某一列权重大，说明当前目标下平台更重视这一项。</div>
          <div className="banner banner-muted">术语提示：`有效权重` 是当前真正参与打分的权重；`基准修正` 来自 benchmark；`校准修正` 来自历史在线归档。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>目标</th>
                  <th>有效权重</th>
                  <th>基准修正</th>
                  <th>校准修正</th>
                </tr>
              </thead>
              <tbody>
                {(feedbackOverview.weight_profiles || []).map((item) => (
                  <tr key={item.objective}>
                    <td>{item.objective}</td>
                    <td className="limitation-cell">
                      {Object.entries(item.effective_weights || {}).map(([key, value]) => `${key}=${Number(value).toFixed(2)}`).join(" / ")}
                      <div className="table-note">{(item.reasons || []).slice(0, 2).join("；") || "暂无额外说明"}</div>
                    </td>
                    <td className="limitation-cell">
                      {Object.keys(item.benchmark_adjustments || {}).length
                        ? Object.entries(item.benchmark_adjustments || {}).map(([key, value]) => `${key}=${Number(value).toFixed(2)}`).join(" / ")
                        : "无"}
                    </td>
                    <td className="limitation-cell">
                      {Object.keys(item.archive_adjustments || {}).length
                        ? Object.entries(item.archive_adjustments || {}).map(([key, value]) => `${key}=${Number(value).toFixed(2)}`).join(" / ")
                        : "无"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="分析区" title="执行路径分布" note="这张表看的是平台常用哪种调度方式完成任务，例如单模型直出还是多模型融合。" />
          <div className="banner banner-muted">这张表看什么：看平台平时更常用哪种执行方式，例如单模型直出、多模型共识还是多模型融合。</div>
          <div className="banner banner-muted">这里统计的是“执行方式”出现了多少次，而不是某个具体模型的性能排名。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>执行路径</th>
                  <th>中文说明</th>
                  <th>决策数</th>
                  <th>补充记录</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {(feedbackOverview.execution_path_summaries || []).map((item) => (
                  <tr key={item.execution_path}>
                    <td>{item.execution_path}</td>
                    <td>{formatExecutionPathLabel(item.execution_path)}</td>
                    <td>{item.decision_count}</td>
                    <td>{item.feedback_count}</td>
                    <td className="limitation-cell">
                      {item.decision_count > 0
                        ? `该路径已出现 ${item.decision_count} 次，可用于观察不同目标下的执行路径分布。`
                        : "暂无路径记录。"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="分析区" title="最近决策记录" note="这张表最适合做论文截图。它能直接展示平台不是一次性推理，而是会把每次任务调度结果完整留痕。" />
          <div className="banner banner-muted">这张表看什么：一行就是一次真实任务调度。你可以直接看到目标、领域、执行路径、参与模型和归档说明。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>决策</th>
                  <th>目标 / 领域</th>
                  <th>执行路径</th>
                  <th>模型</th>
                  <th>归档说明</th>
                </tr>
              </thead>
              <tbody>
                {(feedbackOverview.recent_decisions || []).map((item) => (
                  <tr key={item.decision_id}>
                    <td>
                      <strong>{item.decision_id}</strong>
                      <div className="table-note">{item.created_at}</div>
                    </td>
                    <td>
                      <div>{item.objective}</div>
                      <div className="table-note">{item.context_domain} / {item.inferred_language}</div>
                    </td>
                    <td>
                      <div>{formatExecutionPathLabel(item.execution_path)}</div>
                      <div className="table-note">triples={item.fused_triple_count || 0}</div>
                    </td>
                    <td className="limitation-cell">{(item.selected_model_ids || []).join("、") || "-"}</td>
                    <td>
                      <div>{item.feedback_status === "completed" ? "附带补充记录" : "自动决策归档"}</div>
                      <div className="table-note">{formatFusionPolicyLabel(item.fusion_policy)}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>
    </>
  );
}

function DeploymentPolicyPage({
  actionBusy,
  derived,
  deploymentPolicy,
  deploymentPolicyAutomation,
  deploymentPolicyAutomationForm,
  onCreateDeployment,
  onLoadDeploymentPolicy,
  onRunDeploymentPolicyAutomationNow,
  onUpdateDeploymentPolicyAutomation,
  onDeploymentPolicyAutomationFormChange,
}) {
  const recommendations = deploymentPolicy?.recommendations || [];
  const summary = deploymentPolicy?.summary || null;
  const deployments = derived?.deployments || [];
  const hotModels = recommendations.filter((item) => item.service_tier === "hot");
  const warmModels = recommendations.filter((item) => item.service_tier === "warm");
  const coldModels = recommendations.filter((item) => item.service_tier === "cold");
  const gpuPressureModels = recommendations.filter((item) => item.gpu_pressure_action === "migrate_to_cpu");
  const recommendationByModelId = new Map(recommendations.map((item) => [item.model_id, item]));
  const freestGpu = deploymentPolicy?.gpu_snapshot?.devices?.length
    ? [...deploymentPolicy.gpu_snapshot.devices].sort(
        (left, right) =>
          ((right.memory_total_mb || 0) - (right.memory_used_mb || 0)) -
          ((left.memory_total_mb || 0) - (left.memory_used_mb || 0)),
      )[0]
    : null;

  const tierLabel = {
    hot: "热模型",
    warm: "温模型",
    cold: "冷模型",
  };
  const residencyLabel = {
    always_on: "常驻",
    keep_warm: "保温",
    on_demand: "按需启动",
  };
  const pressureActionLabel = {
    retain_gpu: "保留 GPU",
    share_gpu: "优先换卡 / 共享 GPU",
    migrate_to_cpu: "全局高压迁移 CPU",
  };
  const riskLabel = {
    low: "低",
    medium: "中",
    high: "高",
    unknown: "待评估",
  };
  const deploymentPlanRows = deployments
    .filter((item) => item.status !== "deleted")
    .map((deployment) => {
      const recommendation = recommendationByModelId.get(deployment.model_id);
      if (!recommendation) return null;
      const currentGpuLabel = deployment.use_gpu
        ? `GPU ${deployment.gpu_ids?.length ? deployment.gpu_ids.join(",") : "未指定"}`
        : "CPU";
      const targetGpuLabel = recommendation.recommended_use_gpu
        ? (recommendation.recommended_gpu_ids?.length
            ? `GPU ${recommendation.recommended_gpu_ids.join(",")}`
            : recommendation.recommended_device_label || "GPU")
        : "CPU";
      const profileChanged = deployment.profile_name !== recommendation.recommended_profile_name;
      const gpuChanged =
        Boolean(deployment.use_gpu) !== Boolean(recommendation.recommended_use_gpu) ||
        JSON.stringify(deployment.gpu_ids || []) !== JSON.stringify(recommendation.recommended_gpu_ids || []);
      let optimizationAction = "保持现状";
      if (gpuChanged && recommendation.recommended_use_gpu) {
        optimizationAction = "迁移到建议 GPU";
      } else if (gpuChanged && !recommendation.recommended_use_gpu) {
        optimizationAction = "转为 CPU";
      } else if (profileChanged) {
        optimizationAction = "调整 Profile";
      }
      return {
        deployment,
        recommendation,
        currentGpuLabel,
        targetGpuLabel,
        optimizationAction,
        profileChanged,
        gpuChanged,
      };
    })
    .filter(Boolean);

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="创新点二"
          title="分层自适应部署策略"
          note="根据模型热度、冷启动风险、运行时延和 GPU 压力，采用“先换卡、再降级”的两阶段策略生成热/温/冷部署建议。"
          right={
            <div className="actions">
              <button
                className="btn-primary"
                disabled={actionBusy["deployment-policy"]}
                onClick={() => void onLoadDeploymentPolicy()}
                type="button"
              >
                {actionBusy["deployment-policy"] ? "分析中" : "刷新部署建议"}
              </button>
              <button
                disabled={actionBusy["deployment-policy-run-now"]}
                onClick={() => void onRunDeploymentPolicyAutomationNow()}
                type="button"
              >
                {actionBusy["deployment-policy-run-now"] ? "执行中" : "立即执行一轮"}
              </button>
            </div>
          }
        />
        <div className="banner banner-muted">
          {summary?.strategy_overview ||
            "这里展示的不是简单“热门上好卡”的规则，而是面向知识抽取模型库的两阶段部署策略：先在 GPU 池内换卡与重平衡；只有在全局 GPU 高压时，才把冷模型或低优先级模型迁往 CPU 释放显存。"}
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="显存守护" title="GPU 显存超阈值自动重启" note="这部分独立于部署优化。它负责按固定间隔巡检 GPU 上各 deployment 的显存占用，单个 deployment 超阈值时直接触发重启。" />
        <div className="form-grid">
          <label className="stack-field">
            <span>显存守护</span>
            <select
              onChange={(event) =>
                onDeploymentPolicyAutomationFormChange((current) => ({
                  ...current,
                  gpuMemoryRestartEnabled: event.target.value === "true",
                }))
              }
              value={String(deploymentPolicyAutomationForm.gpuMemoryRestartEnabled)}
            >
              <option value="true">开启</option>
              <option value="false">关闭</option>
            </select>
          </label>
          <label className="stack-field">
            <span>巡检间隔（秒）</span>
            <input
              min="60"
              onChange={(event) =>
                onDeploymentPolicyAutomationFormChange((current) => ({
                  ...current,
                  gpuMemoryRestartIntervalSeconds: event.target.value,
                }))
              }
              type="number"
              value={deploymentPolicyAutomationForm.gpuMemoryRestartIntervalSeconds}
            />
          </label>
          <label className="stack-field">
            <span>显存重启阈值（MB）</span>
            <input
              min="1024"
              onChange={(event) =>
                onDeploymentPolicyAutomationFormChange((current) => ({
                  ...current,
                  gpuMemoryRestartThresholdMb: event.target.value,
                }))
              }
              type="number"
              value={deploymentPolicyAutomationForm.gpuMemoryRestartThresholdMb}
            />
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={actionBusy["deployment-policy-automation"]}
              onClick={() => void onUpdateDeploymentPolicyAutomation()}
              type="button"
            >
              {actionBusy["deployment-policy-automation"] ? "保存中" : "保存守护与优化配置"}
            </button>
          </div>
        </div>
        <div className="metrics">
          <MetricCard
            title="守护状态"
            value={deploymentPolicyAutomation?.gpu_memory_restart_enabled ? "已开启" : "已关闭"}
            note={`最近巡检 ${deploymentPolicyAutomation?.last_gpu_memory_restart_at || "-"}`}
          />
          <MetricCard
            title="巡检间隔"
            value={`${deploymentPolicyAutomation?.gpu_memory_restart_interval_seconds || 60} s`}
            note="默认每 60 秒巡检一次"
          />
          <MetricCard
            title="重启阈值"
            value={`${deploymentPolicyAutomation?.gpu_memory_restart_threshold_mb || 5120} MB`}
            note="单个 GPU deployment 超过该阈值即触发自动重启"
          />
          <MetricCard
            title="巡检轮次"
            value={deploymentPolicyAutomation?.gpu_memory_restart_run_count || 0}
            note="显存守护累计执行次数"
          />
        </div>
        <div className="detail-note">
          显存守护不参与“部署优化建议”决策。它只做一件事：独立按固定间隔巡检 GPU 进程，若单个 deployment 显存占用超过阈值，就在保证推理保护窗口的前提下自动重启该 deployment。
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="自动优化" title="定时部署优化开关" note="这部分才是部署优化本体。默认开启后，平台会按设定间隔执行换卡、重平衡和必要时的 CPU 迁移；不会停掉正在执行推理任务的模型。" />
        <div className="form-grid">
          <label className="stack-field">
            <span>部署优化</span>
            <select
              onChange={(event) =>
                onDeploymentPolicyAutomationFormChange((current) => ({
                  ...current,
                  enabled: event.target.value === "true",
                }))
              }
              value={String(deploymentPolicyAutomationForm.enabled)}
            >
              <option value="true">开启</option>
              <option value="false">关闭</option>
            </select>
          </label>
          <label className="stack-field">
            <span>执行间隔（秒）</span>
            <input
              min="60"
              onChange={(event) =>
                onDeploymentPolicyAutomationFormChange((current) => ({
                  ...current,
                  intervalSeconds: event.target.value,
                }))
              }
              type="number"
              value={deploymentPolicyAutomationForm.intervalSeconds}
            />
          </label>
          <label className="stack-field">
            <span>忙碌保护窗口（秒）</span>
            <input
              min="30"
              onChange={(event) =>
                onDeploymentPolicyAutomationFormChange((current) => ({
                  ...current,
                  protectActiveWindowSeconds: event.target.value,
                }))
              }
              type="number"
              value={deploymentPolicyAutomationForm.protectActiveWindowSeconds}
            />
          </label>
        </div>
        <div className="metrics">
          <MetricCard title="优化状态" value={deploymentPolicyAutomation?.enabled ? "已开启" : "已关闭"} note={deploymentPolicyAutomation?.last_status || "idle"} />
          <MetricCard title="执行间隔" value={`${deploymentPolicyAutomation?.interval_seconds || 900} s`} note="部署优化扫描频率" />
          <MetricCard
            title="忙碌保护"
            value={`${deploymentPolicyAutomation?.protect_active_window_seconds || 120} s`}
            note="该窗口内有推理请求的模型不会被自动优化"
          />
          <MetricCard title="最近优化" value={deploymentPolicyAutomation?.last_run_at || "-"} note={`累计 ${deploymentPolicyAutomation?.run_count || 0} 轮`} />
        </div>
        <div className="detail-note">
          部署优化遵循两阶段规则：优先做 GPU 内部换卡与重平衡；仅当所有 GPU 都处于高压状态时，才把冷模型或低优先级 deployment 迁移到 CPU。
        </div>
        {deploymentPolicyAutomation?.recent_actions?.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>动作</th>
                  <th>模型</th>
                  <th>状态</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {deploymentPolicyAutomation.recent_actions.map((item, index) => (
                  <tr key={`${item.created_at}-${item.model_id}-${index}`}>
                    <td>{item.created_at}</td>
                    <td>{item.action_type}</td>
                    <td>
                      <strong>{item.model_id}</strong>
                      <div className="table-note">{item.deployment_id || "-"}</div>
                    </td>
                    <td>{item.status}</td>
                    <td className="limitation-cell">{item.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="table-note">开启自动优化后，最近执行动作会记录在这里，便于截图和论文说明。</div>
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="GPU 快照" title="当前资源视图" />
        {deploymentPolicy?.gpu_snapshot?.available ? (
          <div className="metrics">
            <MetricCard title="GPU 数量" value={deploymentPolicy.gpu_snapshot.gpu_count} note="当前服务器可见设备数" />
            <MetricCard title="活跃进程" value={deploymentPolicy.gpu_snapshot.process_count || 0} note="nvidia-smi 检测到的活跃进程" />
            <MetricCard title="最空闲 GPU" value={freestGpu ? `GPU ${freestGpu.index}` : "-"} note="按显存余量估算" />
            <MetricCard title="CPU 迁移候选" value={gpuPressureModels.length} note="全局 GPU 高压时建议迁移到 CPU 的模型数" />
          </div>
        ) : (
          <EmptyState icon="G" title="还没有部署建议快照" note="点击上方“刷新部署建议”后，这里会展示实时 GPU 资源视图。" />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="分层概览" title="热温冷模型分层" note="这一层把 20+ 制品按服务热度与冷启动风险进行归类，便于统一讲述部署策略。" />
        {recommendations.length ? (
          <div className="list-grid">
            <article className="list-card">
              <div className="list-card-header">
                <div>
                  <h4>热模型</h4>
                  <div className="table-note">优先常驻，保障时延</div>
                </div>
                <span className="tag">{hotModels.length}</span>
              </div>
              <div className="tag-list">
                {hotModels.length ? hotModels.map((item) => <span className="tag" key={item.model_id}>{item.model_id}</span>) : <span className="table-note">暂无</span>}
              </div>
            </article>
            <article className="list-card">
              <div className="list-card-header">
                <div>
                  <h4>温模型</h4>
                  <div className="table-note">优先换卡与共享，必要时再考虑降级</div>
                </div>
                <span className="tag">{warmModels.length}</span>
              </div>
              <div className="tag-list">
                {warmModels.length ? warmModels.map((item) => <span className="tag" key={item.model_id}>{item.model_id}</span>) : <span className="table-note">暂无</span>}
              </div>
            </article>
            <article className="list-card">
              <div className="list-card-header">
                <div>
                  <h4>冷模型</h4>
                  <div className="table-note">按需启动，全局 GPU 高压时优先迁移到 CPU</div>
                </div>
                <span className="tag">{coldModels.length}</span>
              </div>
              <div className="tag-list">
                {coldModels.length ? coldModels.map((item) => <span className="tag" key={item.model_id}>{item.model_id}</span>) : <span className="table-note">暂无</span>}
              </div>
            </article>
          </div>
        ) : (
          <EmptyState icon="L" title="还没有策略分层结果" note="点击“刷新部署建议”后，这里会展示热模型、温模型和冷模型的划分结果。" />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="策略建议" title="现有 Deployment 优化方案表" note="每一行对应平台里已经存在的 deployment，展示它当前状态、建议目标和下一步优化动作。" />
        {deploymentPlanRows.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Deployment</th>
                  <th>模型 / 分层</th>
                  <th>当前部署</th>
                  <th>建议目标</th>
                  <th>优化动作</th>
                  <th>说明</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {deploymentPlanRows.map((row) => (
                  <tr key={row.deployment.deployment_id}>
                    <td>
                      <strong>{row.deployment.deployment_id}</strong>
                      <div className="table-note">{row.deployment.status}</div>
                      <div className="table-note">{row.deployment.notes || "-"}</div>
                    </td>
                    <td>
                      <strong>{row.recommendation.model_name}</strong>
                      <div className="mono">{row.recommendation.model_id}</div>
                      <div className="tag-list">
                        <span className="tag">{tierLabel[row.recommendation.service_tier] || row.recommendation.service_tier}</span>
                        <span className="tag">{residencyLabel[row.recommendation.residency_mode] || row.recommendation.residency_mode}</span>
                      </div>
                      <div className="table-note">需求等级：{row.recommendation.demand_level} / 分值：{formatFixed(row.recommendation.demand_score)}</div>
                    </td>
                    <td>
                      <div>{row.currentGpuLabel}</div>
                      <div className="table-note">Profile：{row.deployment.profile_name}</div>
                      <div className="table-note">runtime：{row.deployment.profile_runtime || row.deployment.runtime || "-"}</div>
                      <div className="table-note">近 7 天请求：{row.recommendation.recent_request_count}</div>
                    </td>
                    <td>
                      <div>{row.targetGpuLabel}</div>
                      <div className="table-note">建议 Profile：{row.recommendation.recommended_profile_name}</div>
                      <div className="table-note">实例数：{row.recommendation.recommended_instance_count}</div>
                      <div className="table-note">冷启动风险：{riskLabel[row.recommendation.cold_start_risk] || row.recommendation.cold_start_risk}</div>
                    </td>
                    <td>
                      <div>{row.optimizationAction}</div>
                      <div className="table-note">
                        {row.gpuChanged ? `设备调整为 ${row.targetGpuLabel}` : "设备保持不变"}
                      </div>
                      <div className="table-note">
                        {row.profileChanged ? `Profile 调整为 ${row.recommendation.recommended_profile_name}` : "Profile 保持不变"}
                      </div>
                      <div className="tag-list">
                        {(row.recommendation.policy_tags || []).map((tag) => (
                          <span className="tag" key={`${row.deployment.deployment_id}-${tag}`}>{tag}</span>
                        ))}
                      </div>
                    </td>
                    <td className="limitation-cell">
                      {row.recommendation.rationale}
                      <div className="table-note">高压动作：{pressureActionLabel[row.recommendation.gpu_pressure_action] || row.recommendation.gpu_pressure_action}</div>
                    </td>
                    <td>
                      <div className="actions row-actions">
                        <button
                          className="btn-primary"
                          onClick={() =>
                            void onCreateDeployment({
                              model_id: row.recommendation.model_id,
                              profile_name: row.recommendation.recommended_profile_name,
                              launch_runtime: true,
                              use_gpu: row.recommendation.recommended_use_gpu,
                              gpu_ids: row.recommendation.recommended_gpu_ids,
                              notes: `deployment_policy_${row.recommendation.service_tier}`,
                            })
                          }
                          type="button"
                        >
                          按建议新建
                        </button>
                        {row.recommendation.gpu_pressure_action === "migrate_to_cpu" ? (
                          <button
                            onClick={() =>
                              void onCreateDeployment({
                                model_id: row.recommendation.model_id,
                                profile_name: row.recommendation.fallback_profile_name,
                                launch_runtime: true,
                                use_gpu: false,
                                gpu_ids: [],
                                notes: "manual_cpu_fallback",
                              })
                            }
                            type="button"
                          >
                            新建 CPU 迁移实例
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon="P" title="还没有生成 Deployment 优化方案" note="点击“刷新部署建议”后，这里会按现有 deployment 生成优化方案。" />
        )}
      </section>
    </>
  );
}

function ArchivePage({ actionBusy, derived, onDeletePredictionArchive, onExportPredictionArchive, onUpdatePredictionArchiveFeedback }) {
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageArchives = derived.currentAccessUser?.is_admin;
  const canExportArchives = currentPermissions.includes("archive.export");
  const archiveSatisfiedCount = derived.predictArchives.filter((item) => normalizePredictionFeedbackStatus(item.feedback_status) === "satisfied").length;
  const archiveUnsatisfiedCount = derived.predictArchives.filter((item) => normalizePredictionFeedbackStatus(item.feedback_status) === "unsatisfied").length;
  const archivePendingCount = Math.max(derived.predictArchives.length - archiveSatisfiedCount - archiveUnsatisfiedCount, 0);
  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader kicker="结果归档" title="结果归档" note="这里只保留预测留痕与反馈统计，用来回看在线推理结果、整理论文样例和补充调度证据。" />
        <div className="metrics">
          <MetricCard title="最新预测归档" value={derived.latestPredictArchive?.archive_id || "-"} note={derived.latestPredictArchive?.model_id || "-"} />
          <MetricCard title="预测归档总数" value={derived.predictArchives.length} note="在线推理结果自动留痕" />
          <MetricCard title="满意结果" value={archiveSatisfiedCount} note="用户确认当前结果可接受" />
          <MetricCard title="不满意结果" value={archiveUnsatisfiedCount} note="用户明确标记结果不符合预期" />
          <MetricCard title="暂定结果" value={archivePendingCount} note="默认状态，可理解为用户无反馈" />
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="归档列表" title="预测结果归档列表" note="这里保留最近的单次预测结果，适合导出案例结果、补论文截图，也支持直接补满意/不满意反馈。" />
        <div className="banner banner-muted">这张表看什么：一行就是一次真实预测归档。除了看输入、延迟和三元组数量，还可以直接补“满意 / 不满意 / 暂定”反馈，作为后续调度分析的辅助证据。</div>
        {derived.predictArchives.length ? (
          <div className="table-wrap">
            <table className="archive-table">
              <thead>
                <tr>
                  <th>归档信息</th>
                  <th>模型与场景</th>
                  <th>输入文本</th>
                  <th>结果摘要</th>
                  <th>反馈</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {sortByCreatedDesc(derived.predictArchives)
                  .slice(0, 10)
                  .map((item) => (
                    <tr key={item.archive_id}>
                      <td>
                        <div className="archive-cell-stack">
                          <strong className="archive-main">{item.archive_id}</strong>
                          <div className="table-note">创建时间：{item.created_at}</div>
                        </div>
                      </td>
                      <td>
                        <div className="archive-cell-stack">
                          <div className="archive-main">{item.model_id}</div>
                          <div className="table-note">Profile：{item.profile_name || "-"}</div>
                          <div className="table-note">场景：{item.scene || "-"}</div>
                        </div>
                      </td>
                      <td>
                        <div className="archive-input">{item.request_text || item.request_texts?.[0] || "-"}</div>
                      </td>
                      <td>
                        <div className="archive-cell-stack">
                          <div className="archive-result-grid">
                            <span className="archive-result-chip">延迟 {escapeValue(item.latency_ms)} ms</span>
                            <span className="archive-result-chip">三元组 {escapeValue(item.triple_count)}</span>
                          </div>
                          <div className="table-note">结果归档用于回放预测输出、导出 JSON 和补充反馈。</div>
                        </div>
                      </td>
                      <td>
                        <div className="archive-cell-stack">
                          <div className="archive-main">{formatPredictionFeedbackLabel(item.feedback_status)}</div>
                          <div className="table-note">{item.feedback_updated_at || "尚未更新反馈时间"}</div>
                        </div>
                        <PredictionArchiveFeedbackControls
                          actionBusy={actionBusy}
                          archive={item}
                          onUpdate={onUpdatePredictionArchiveFeedback}
                        />
                      </td>
                      <td>
                        <div className="actions row-actions">
                          {canExportArchives ? (
                            <button onClick={() => onExportPredictionArchive(item)} type="button">导出</button>
                          ) : null}
                          {canManageArchives ? (
                            <button onClick={() => void onDeletePredictionArchive(item.archive_id)} type="button">删除</button>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState icon="A" title="暂无预测归档" note="先执行一次在线推理，这里就会出现归档结果。" />
        )}
      </section>
    </>
  );
}

function ModelsPage({
  actionBusy,
  data,
  deploymentForm,
  deploymentPolicy,
  derived,
  modelImportForm,
  modelTransferItems,
  onDismissModelTransfer,
  onDeleteModelPackage,
  onCreateDeployment,
  onDeploymentFormChange,
  onExportModelPackage,
  onImportModel,
  onModelImportFormChange,
}) {
  const [search, setSearch] = useState("");
  const [runtimeFilter, setRuntimeFilter] = useState("all");
  const [originFilter, setOriginFilter] = useState("all");
  const deferredSearch = useDeferredValue(search);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageDeployments = currentPermissions.includes("deployments.manage");
  const canExportPackages = currentPermissions.includes("archive.export");
  const canManageModels = derived.currentAccessUser?.is_admin;
  const familyOptions = data.importOptions?.families || [];
  const selectedFamilyOption = familyOptions.find((item) => item.code === modelImportForm.modelFamily) || familyOptions[0];
  const recommendationByModelId = useMemo(
    () => new Map((deploymentPolicy?.recommendations || []).map((item) => [item.model_id, item])),
    [deploymentPolicy],
  );

  const filteredModels = useMemo(() => {
    return [...derived.models]
      .filter((model) =>
        matchesKeyword(
          [
            model.model_id,
            model.model_name,
            model.dataset_name,
            model.runtime_status,
            model.runtime_adapter,
            model.backbone,
          ],
          deferredSearch,
        ),
      )
      .filter((model) => runtimeFilter === "all" || model.runtime_status === runtimeFilter)
      .filter((model) => originFilter === "all" || model.package_origin === originFilter)
      .sort(
        (left, right) =>
          Number(right.predict_ready) - Number(left.predict_ready) || left.model_name.localeCompare(right.model_name),
      );
  }, [deferredSearch, derived.models, originFilter, runtimeFilter]);

  return (
    <>
      {canManageModels ? (
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="模型导入"
          title="上传模型制品"
          note="模式A上传原始 ONNX 压缩包后由平台自动补齐标准目录；模式B直接接入已经标准化的 LuckyModels 模型包。"
        />
        <div className="detail-note upload-note">
          导入命名规则：模型家族-语言-数据集[-别名]-版本，例如 <span className="mono">casrel-zh-duie-myexp-v1</span>。
        </div>
        {modelTransferItems.length ? (
          <div className="transfer-list">
            {modelTransferItems.map((item) => {
              const transferRunning = item.phase === "running";
              return (
                <div className={`transfer-card transfer-${item.phase}`} key={item.taskId}>
                  <div className="transfer-head">
                    <strong>{item.direction === "download" ? "模型制品下载进度" : "模型制品上传进度"}</strong>
                    <div className="transfer-head-actions">
                      <span>
                        {item.percent !== null
                          ? `${item.percent}%`
                          : item.phase === "running"
                            ? "处理中"
                            : ""}
                      </span>
                      {!transferRunning ? (
                        <button className="btn-quiet" onClick={() => onDismissModelTransfer(item.taskId)} type="button">
                          收起
                        </button>
                      ) : null}
                    </div>
                  </div>
                  <div className="transfer-track">
                    <div
                      className={`transfer-fill ${item.percent === null ? "transfer-fill-indeterminate" : ""}`}
                      style={{ width: `${item.percent ?? 35}%` }}
                    />
                  </div>
                  <div className="transfer-meta">
                    <span>{item.fileName || item.modelId || "-"}</span>
                    <span>
                      {formatBytes(item.loadedBytes)}
                      {item.totalBytes ? ` / ${formatBytes(item.totalBytes)}` : ""}
                    </span>
                  </div>
                  <div className="detail-note">{item.detail}</div>
                </div>
              );
            })}
          </div>
        ) : null}
        <div className="form-grid">
          <label className="stack-field">
            <span>导入模式</span>
            <select
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, importMode: event.target.value }))}
              value={modelImportForm.importMode}
            >
              {(data.importOptions?.modes || []).map((mode) => (
                <option key={mode.code} value={mode.code}>
                  {mode.label}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>模型种类</span>
            <select
              onChange={(event) => {
                const nextFamily = familyOptions.find((item) => item.code === event.target.value);
                onModelImportFormChange((current) => ({
                  ...current,
                  modelFamily: event.target.value,
                  language: nextFamily?.default_language || current.language,
                }));
              }}
              value={modelImportForm.modelFamily}
            >
              {familyOptions.map((family) => (
                <option key={family.code} value={family.code}>
                  {family.label}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>语言</span>
            <select
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, language: event.target.value }))}
              value={modelImportForm.language}
            >
              <option value="en">英文</option>
              <option value="zh">中文</option>
            </select>
          </label>
          <label className="stack-field">
            <span>数据集</span>
            <input
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, datasetName: event.target.value }))}
              placeholder={modelImportForm.importMode === "auto_package" ? "例如 webnlg / duie / scierc" : "可选，留空沿用标准包"}
              type="text"
              value={modelImportForm.datasetName}
            />
          </label>
          <label className="stack-field">
            <span>版本</span>
            <input
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, version: event.target.value }))}
              placeholder="v1"
              type="text"
              value={modelImportForm.version}
            />
          </label>
          <label className="stack-field">
            <span>展示名称</span>
            <input
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, modelName: event.target.value }))}
              placeholder={selectedFamilyOption?.label || "模型名称"}
              type="text"
              value={modelImportForm.modelName}
            />
          </label>
          <label className="stack-field">
            <span>主干模型</span>
            <input
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, backbone: event.target.value }))}
              placeholder={modelImportForm.language === "zh" ? "bert-base-chinese" : "bert-base-cased"}
              type="text"
              value={modelImportForm.backbone}
            />
          </label>
          <label className="stack-field">
            <span>别名</span>
            <input
              onChange={(event) => onModelImportFormChange((current) => ({ ...current, alias: event.target.value }))}
              placeholder="可选，用于区分同家族不同实验包"
              type="text"
              value={modelImportForm.alias}
            />
          </label>
          <label className="stack-field stack-field-wide">
            <span>模型压缩包</span>
            <input
              accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2,.tar.xz"
              onChange={(event) =>
                onModelImportFormChange((current) => ({
                  ...current,
                  archive: event.target.files?.[0] || null,
                }))
              }
              type="file"
            />
            <small className="detail-note">
              当前选择：{modelImportForm.archive?.name || "未选择文件"}
            </small>
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={!modelImportForm.archive || !modelImportForm.modelFamily}
              onClick={() => void onImportModel()}
              type="button"
            >
              导入模型
            </button>
          </div>
        </div>
      </section>
      ) : null}

      {canManageDeployments ? (
      <section className="page-shell card section-shell">
        <SectionHeader kicker="新建部署" title="从模型库创建部署实例" note="选择模型、配置模板和备注，直接通过控制面创建部署实例。" />
        <div className="form-grid">
          <label className="stack-field">
            <span>模型</span>
            <select
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, modelId: event.target.value }))}
              value={deploymentForm.modelId}
            >
              {derived.models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>配置模板</span>
            <select
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, profileName: event.target.value }))}
              value={deploymentForm.profileName}
            >
              {derived.deploymentProfiles.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>GPU 卡号</span>
            <input
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, gpuIdsText: event.target.value }))}
              disabled={!deploymentForm.useGpu}
              placeholder="留空=自动，示例：0 或 1,2"
              type="text"
              value={deploymentForm.gpuIdsText}
            />
          </label>
          <label className="stack-field">
            <span>备注</span>
            <input
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, notes: event.target.value }))}
              type="text"
              value={deploymentForm.notes}
            />
          </label>
          <label className="stack-check">
            <input
              checked={deploymentForm.useGpu}
              onChange={(event) =>
                onDeploymentFormChange((current) => ({
                  ...current,
                  useGpu: event.target.checked,
                  gpuIdsText: event.target.checked ? current.gpuIdsText : "",
                }))
              }
              type="checkbox"
            />
            使用 GPU
          </label>
          <label className="stack-check">
            <input
              checked={deploymentForm.launchRuntime}
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, launchRuntime: event.target.checked }))}
              type="checkbox"
            />
            部署后立即启动
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={actionBusy["create-deployment"] || !deploymentForm.modelId || !deploymentForm.profileName}
              onClick={() => void onCreateDeployment()}
              type="button"
            >
              新建部署
            </button>
          </div>
        </div>
      </section>
      ) : null}

      <section className="page-shell card section-shell">
        <SectionHeader kicker="模型库" title="模型资产" note="模型卡片支持快速查看运行状态，并为可推理模型直接创建部署实例。" />
        <div className="filter-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>搜索模型</span>
              <input
                onChange={(event) => setSearch(event.target.value)}
                placeholder="模型名 / 模型编号 / 数据集 / 适配器"
                type="text"
                value={search}
              />
            </label>
            <label className="stack-field">
              <span>运行状态</span>
              <select onChange={(event) => setRuntimeFilter(event.target.value)} value={runtimeFilter}>
                <option value="all">全部状态</option>
                <option value="inference_ready">可推理</option>
                <option value="package_ready">仅制品就绪</option>
              </select>
            </label>
            <label className="stack-field">
              <span>来源</span>
              <select onChange={(event) => setOriginFilter(event.target.value)} value={originFilter}>
                <option value="all">全部来源</option>
                <option value="builtin">本机自带</option>
                <option value="user_upload">用户上传</option>
              </select>
            </label>
          </div>
          <div className="filter-foot">
            <div className="filter-meta">
              当前显示 {filteredModels.length} / {derived.models.length} 个模型
            </div>
            <button
              onClick={() => {
                setSearch("");
                setRuntimeFilter("all");
                setOriginFilter("all");
              }}
              type="button"
            >
              清空筛选
            </button>
          </div>
        </div>
        <div className="asset-grid">
          {filteredModels.map((model) => {
            const detail = data.modelDetails[model.model_id] || {};
            const manifest = detail.manifest || {};
            const recommendation = recommendationByModelId.get(model.model_id);
            const quickDeployProfile = resolveProfileName(
              model.model_id,
              data.modelDetails,
              recommendation?.recommended_profile_name || "default",
            );
            const quickDeployUseGpu = recommendation?.recommended_use_gpu ?? true;
            const quickDeployGpuIds = quickDeployUseGpu ? (recommendation?.recommended_gpu_ids || []) : [];
            return (
              <article className="asset-card" key={model.model_id}>
                <h4>{model.model_name}</h4>
                <div className="asset-meta">
                  编号：<span className="mono">{model.model_id}</span>
                  <br />
                  数据集：{model.dataset_name || "-"}
                  <br />
                  主干模型：{model.backbone}
                  <br />
                  适配器：{model.runtime_adapter}
                  <br />
                  配置模板：{Object.keys(detail.resolved_profiles || {}).join(", ") || "-"}
                </div>
                <div className="tag-list">
                  <span className="tag">{formatStatusLabel(model.runtime_status)}</span>
                  <span className="tag">{model.package_origin_label || "本机自带"}</span>
                  <span className="tag">{manifest.framework || "-"}</span>
                  <span className="tag">{manifest.stage_count || "-" } 个阶段</span>
                </div>
                <div className="actions inline-actions">
                  {canManageDeployments ? (
                    <button
                      className="btn-primary"
                      disabled={!model.predict_ready || actionBusy["create-deployment"]}
                      onClick={() =>
                        void onCreateDeployment({
                          model_id: model.model_id,
                          profile_name: quickDeployProfile,
                          launch_runtime: true,
                          use_gpu: quickDeployUseGpu,
                          gpu_ids: quickDeployGpuIds,
                          notes: recommendation
                            ? `从模型页快速创建（按部署策略：${recommendation.recommended_device_label || (quickDeployUseGpu ? "GPU" : "CPU")}）`
                            : "从模型页快速创建",
                        })
                      }
                      type="button"
                    >
                      快速部署
                    </button>
                  ) : null}
                  {canExportPackages ? (
                    <button
                      disabled={actionBusy[`export-model-${model.model_id}`]}
                      onClick={() => void onExportModelPackage(model.model_id)}
                      type="button"
                    >
                      导出制品
                    </button>
                  ) : null}
                  {canManageModels && model.deletable ? (
                    <button
                      className="btn-danger"
                      disabled={actionBusy[`delete-model-${model.model_id}`]}
                      onClick={() => void onDeleteModelPackage(model.model_id)}
                      type="button"
                    >
                      {actionBusy[`delete-model-${model.model_id}`] ? "删除中" : "删除制品"}
                    </button>
                  ) : !model.deletable ? (
                    <button disabled title="本机自带模型不可删除" type="button">
                      本机自带
                    </button>
                  ) : canManageModels ? null : (
                    <button disabled title="仅管理员可删除用户上传模型" type="button">
                      管理员可删除
                    </button>
                  )}
                </div>
              </article>
            );
          })}
          {!filteredModels.length ? (
            <EmptyState
              icon="M"
              note="当前筛选条件下没有匹配模型，试试放宽状态或关键词。"
              title="没有匹配的模型"
              wide
            />
          ) : null}
        </div>
      </section>
    </>
  );
}

function CapabilityMatrixPage({ derived }) {
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search);
  const [statusFilter, setStatusFilter] = useState("all");

  const filteredRows = useMemo(() => {
    return derived.capabilityMatrix.filter((item) => {
      const matchesStatus =
        statusFilter === "all" ||
        (statusFilter === "ready" && item.inference_ready) ||
        (statusFilter === "limited" && !item.inference_ready);
      if (!matchesStatus) {
        return false;
      }
      return matchesKeyword(
        [
          item.model_id,
          item.model_name,
          item.model_family,
          item.language,
          item.dataset_name,
          item.runtime_status,
          item.known_limitations,
        ],
        deferredSearch,
      );
    });
  }, [deferredSearch, derived.capabilityMatrix, statusFilter]);

  function handleExportCapabilityMatrix() {
    if (!filteredRows.length) {
      return;
    }
    downloadCsvFile(
      "luckymodels-capability-matrix.csv",
      filteredRows.map((item) => ({
        model_id: item.model_id,
        model_name: item.model_name,
        dataset_name: item.dataset_name,
        runtime_status: item.runtime_status,
        inference_ready: item.inference_ready ? "yes" : "no",
        supports_online_predict: item.supports_online_predict ? "yes" : "no",
        supports_benchmark: item.supports_benchmark ? "yes" : "no",
        supports_batch_predict: item.supports_batch_predict ? "yes" : "no",
        profile_count: item.profile_count,
        deployment_count: item.deployment_count,
        benchmark_count: item.benchmark_count,
        latest_deployment_id: item.latest_deployment_id || "-",
        latest_benchmark_id: item.latest_benchmark_id || "-",
        known_limitations: item.known_limitations,
      })),
    );
  }

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="能力矩阵"
          title="模型能力矩阵"
          note="统一汇总模型库当前的接入状态、在线推理能力、基准测试支持和已知限制，适合作为论文中的模型构成表与平台接入结果表。"
          right={
            <div className="actions">
              <button disabled={!filteredRows.length} onClick={handleExportCapabilityMatrix} type="button">
                导出矩阵 CSV
              </button>
            </div>
          }
        />

        <div className="metrics">
          <MetricCard title="模型总数" value={derived.capabilityMatrix.length} note="当前矩阵覆盖模型数" />
          <MetricCard title="可推理模型" value={derived.capabilityMatrix.filter((item) => item.inference_ready).length} note="运行状态为“可推理”" />
          <MetricCard title="已有基准证据" value={derived.capabilityMatrix.filter((item) => item.benchmark_count > 0).length} note="至少一条基准测试记录" />
          <MetricCard title="支持批量推理" value={derived.capabilityMatrix.filter((item) => item.supports_batch_predict).length} note="平台支持 texts 批量入口" />
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="筛选条件"
          title="矩阵筛选"
          note={`当前显示 ${filteredRows.length} / ${derived.capabilityMatrix.length} 个模型条目。`}
        />
        <div className="filter-panel">
          <label className="field">
            <span>搜索模型</span>
            <input
              onChange={(event) => setSearch(event.target.value)}
              placeholder="模型编号 / 模型名 / 数据集 / 已知限制"
              value={search}
            />
          </label>
          <label className="field">
            <span>运行状态</span>
            <select onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
              <option value="all">全部</option>
              <option value="ready">仅可推理模型</option>
              <option value="limited">仅受限模型</option>
            </select>
          </label>
        </div>
      </section>

      {filteredRows.length ? (
        <section className="page-shell card section-shell">
          <SectionHeader
            kicker="能力总表"
            title="固定能力总表"
            note="这里的列已经对齐论文里最常用的能力描述口径，可直接截图或导出后整理。"
          />
          <div className="table-wrap compare-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>模型</th>
                  <th>数据集</th>
                  <th>可推理</th>
                  <th>在线推理</th>
                  <th>基准测试</th>
                  <th>批量推理</th>
                  <th>证据</th>
                  <th>当前已知限制</th>
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((item) => (
                  <tr key={item.model_id}>
                    <td>
                      <strong>{item.model_name}</strong>
                      <div className="mono">{item.model_id}</div>
                    </td>
                    <td>{item.dataset_name || "-"}</td>
                    <td><StatusPill status={item.inference_ready ? "running" : "stopped"} /></td>
                    <td>{item.supports_online_predict ? "是" : "否"}</td>
                    <td>{item.supports_benchmark ? "是" : "否"}</td>
                    <td>{item.supports_batch_predict ? "是" : "否"}</td>
                    <td>
                      <div>部署：{item.deployment_count}</div>
                      <div>基准：{item.benchmark_count}</div>
                      <div className="mono">{item.latest_benchmark_id || item.latest_deployment_id || "-"}</div>
                    </td>
                    <td className="limitation-cell">{item.known_limitations || "暂无显式限制。"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : (
        <section className="page-shell">
          <EmptyState icon="M" title="暂无匹配的能力矩阵条目" note="调整筛选条件后，这里会展示固定的模型能力总表。" wide />
        </section>
      )}
    </>
  );
}

function ArtifactsPage({ actionBusy, data, onDeleteModelPackage, onExportModelManifest, onExportModelProfiles }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader kicker="制品清单" title="制品包清单" note="展示 manifest 与解析后的制品路径，便于检查平台识别结果。" />
      <div className="list-grid">
        {Object.values(data.modelDetails).map((detail) => (
          <article className="list-card" key={detail.summary?.model_id}>
            <div className="list-card-header">
              <div>
                <h4>{detail.summary?.model_name}</h4>
                <div className="mono">{detail.summary?.model_id}</div>
              </div>
              <StatusPill status={detail.summary?.runtime_status || "unknown"} />
            </div>
            <div className="list-card-body artifact-meta-body">
              <div className="artifact-meta-list">
                <div className="artifact-meta-row">
                  <span className="artifact-meta-label">package_root</span>
                  <span className="artifact-meta-value mono">{detail.summary?.package_root || "-"}</span>
                </div>
                <div className="artifact-meta-row">
                  <span className="artifact-meta-label">来源</span>
                  <span className="artifact-meta-value">{detail.summary?.package_origin_label || "本机自带"}</span>
                </div>
                <div className="artifact-meta-row">
                  <span className="artifact-meta-label">tokenizer</span>
                  <span className="artifact-meta-value">{detail.manifest?.tokenizer?.type || "-"}</span>
                </div>
                <div className="artifact-meta-row">
                  <span className="artifact-meta-label">relation_map</span>
                  <span className="artifact-meta-value">{detail.manifest?.labels?.relation_map || "-"}</span>
                </div>
              </div>
            </div>
            <div className="tag-list">
              <span className="tag">{detail.summary?.package_origin_label || "本机自带"}</span>
              {Object.entries(detail.resolved_artifacts || {}).map(([name, value]) => (
                <span className="tag" key={`${detail.summary?.model_id}-${name}`}>
                  {name}: {value}
                </span>
              ))}
            </div>
            <div className="actions inline-actions">
              <button onClick={() => void onExportModelManifest(detail.summary?.model_id)} type="button">导出 manifest</button>
              <button onClick={() => void onExportModelProfiles(detail.summary?.model_id)} type="button">导出配置模板</button>
              {detail.summary?.deletable ? (
                <button
                  className="btn-danger"
                  disabled={actionBusy[`delete-model-${detail.summary?.model_id}`]}
                  onClick={() => void onDeleteModelPackage(detail.summary?.model_id)}
                  type="button"
                >
                  {actionBusy[`delete-model-${detail.summary?.model_id}`] ? "删除中" : "删除制品"}
                </button>
              ) : (
                <button disabled title="本机自带模型不可删除" type="button">本机自带</button>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function DeploymentsPage({
  actionBusy,
  deploymentForm,
  derived,
  onCreateDeployment,
  onDeploymentAction,
  onDeploymentFormChange,
  onExportDeploymentRows,
  onInspectDeployment,
  selectedHealth,
  selectedLogs,
}) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [modelFilter, setModelFilter] = useState("all");
  const deferredSearch = useDeferredValue(search);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageDeployments = currentPermissions.includes("deployments.manage");
  const canViewResources = currentPermissions.includes("resources.view");

  const filteredDeployments = useMemo(() => {
    let next = sortByCreatedDesc(derived.deployments).filter((item) =>
      matchesKeyword(
        [item.deployment_id, item.model_id, item.profile_name, item.notes, item.service_url, item.status],
        deferredSearch,
      ),
    );

    if (statusFilter !== "all") {
      next = next.filter((item) => item.status === statusFilter);
    }

    if (modelFilter !== "all") {
      next = next.filter((item) => item.model_id === modelFilter);
    }

    return next;
  }, [deferredSearch, derived.deployments, modelFilter, statusFilter]);

  function exportableRows() {
    return filteredDeployments.map((item) => ({
      deployment_id: item.deployment_id,
      model_id: item.model_id,
      profile_name: item.profile_name,
      status: item.status,
      device: item.device || "",
      runtime: item.profile_runtime || item.runtime || "",
      use_gpu: item.use_gpu ? "yes" : "no",
      gpu_ids: formatGpuIds(getDeploymentAssignedGpuIds(item)),
      instance_count: getDeploymentExecution(item).instance_count ?? 1,
      batch_size: getDeploymentExecution(item).batch_size ?? 1,
      intra_op_threads: getDeploymentExecution(item).intra_op_threads ?? 0,
      inter_op_threads: getDeploymentExecution(item).inter_op_threads ?? 0,
      providers: formatProviderSummary(item),
      service_url: item.service_url || "",
      pid: item.pid || "",
      notes: item.notes || "",
      created_at: item.created_at,
      updated_at: item.updated_at,
    }));
  }

  return (
    <>
      {canManageDeployments ? (
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="部署控制"
          title="部署控制"
          note="可以在这里创建、停止、重启和删除部署实例。线程数、实例数和批量大小都已经真实接入运行时。"
        />
        <div className="banner banner-muted">
          如果某个部署实例是在 2026-03-28 这次配置模板更新前创建且一直没有重启，它仍可能运行旧参数。重启一次后，新的线程数、实例数和批量配置才会真正生效。
        </div>
        <div className="form-grid">
          <label className="stack-field">
            <span>模型</span>
            <select
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, modelId: event.target.value }))}
              value={deploymentForm.modelId}
            >
              {derived.models.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_id}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>配置模板</span>
            <select
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, profileName: event.target.value }))}
              value={deploymentForm.profileName}
            >
              {derived.deploymentProfiles.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>GPU 卡号</span>
            <input
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, gpuIdsText: event.target.value }))}
              disabled={!deploymentForm.useGpu}
              placeholder="留空=自动，示例：0 或 1,2"
              type="text"
              value={deploymentForm.gpuIdsText}
            />
          </label>
          <label className="stack-field">
            <span>备注</span>
            <input
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, notes: event.target.value }))}
              type="text"
              value={deploymentForm.notes}
            />
          </label>
          <label className="stack-check">
            <input
              checked={deploymentForm.useGpu}
              onChange={(event) =>
                onDeploymentFormChange((current) => ({
                  ...current,
                  useGpu: event.target.checked,
                  gpuIdsText: event.target.checked ? current.gpuIdsText : "",
                }))
              }
              type="checkbox"
            />
            使用 GPU
          </label>
          <label className="stack-check">
            <input
              checked={deploymentForm.launchRuntime}
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, launchRuntime: event.target.checked }))}
              type="checkbox"
            />
            部署后立即启动
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={actionBusy["create-deployment"] || !deploymentForm.modelId || !deploymentForm.profileName}
              onClick={() => void onCreateDeployment()}
              type="button"
            >
              创建部署
            </button>
          </div>
        </div>
      </section>
      ) : null}

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="部署列表"
          title="部署列表"
          right={
            filteredDeployments.length ? (
              <div className="actions">
                <button onClick={() => onExportDeploymentRows(exportableRows(), "csv")} type="button">导出 CSV</button>
                <button onClick={() => onExportDeploymentRows(exportableRows(), "json")} type="button">导出 JSON</button>
              </div>
            ) : null
          }
        />
        <div className="filter-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>搜索部署</span>
              <input
                onChange={(event) => setSearch(event.target.value)}
                placeholder="部署编号 / 模型编号 / 备注 / 服务地址"
                type="text"
                value={search}
              />
            </label>
            <label className="stack-field">
              <span>状态</span>
              <select onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
                <option value="all">全部状态</option>
                <option value="running">运行中</option>
                <option value="starting">启动中</option>
                <option value="planned">已规划</option>
                <option value="stopping">停止中</option>
                <option value="stopped">已停止</option>
                <option value="deleted">已删除</option>
              </select>
            </label>
            <label className="stack-field">
              <span>按模型查看</span>
              <select onChange={(event) => setModelFilter(event.target.value)} value={modelFilter}>
                <option value="all">全部模型</option>
                {derived.models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_id}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="filter-foot">
            <div className="filter-meta">
              当前显示 {filteredDeployments.length} / {derived.deployments.length} 条 deployment 记录
            </div>
            <button
              onClick={() => {
                setSearch("");
                setStatusFilter("all");
                setModelFilter("all");
              }}
              type="button"
            >
              恢复默认视图
            </button>
          </div>
        </div>
        <div className="table-wrap">
          {filteredDeployments.length ? (
            <table>
              <thead>
                <tr>
                  <th>部署编号</th>
                  <th>模型</th>
                  <th>配置模板</th>
                  <th>执行参数</th>
                  <th>状态</th>
                  <th>服务地址</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredDeployments.map((item) => (
                  <tr key={item.deployment_id}>
                    <td>
                      <strong>{item.deployment_id}</strong>
                      <div className="mono">{item.notes || "-"}</div>
                    </td>
                    <td>{item.model_id}</td>
                    <td>
                      <strong>{item.profile_name}</strong>
                    </td>
                    <td>
                      <div className="table-note">{item.device || "-"} / {item.profile_runtime || item.runtime || "-"}</div>
                      <div className="table-note">GPU 加速：{formatBooleanLabel(item.use_gpu)}</div>
                      <div className="table-note">GPU 卡号：{formatGpuIds(getDeploymentAssignedGpuIds(item))}</div>
                      <div className="table-note">{formatExecutionSummary(item)}</div>
                      <div className="table-note">执行后端：{formatProviderSummary(item)}</div>
                    </td>
                    <td><StatusPill status={item.status} /></td>
                    <td>
                      <div className="mono">{item.service_url || "-"}</div>
                      <div className="table-note">pid: {item.pid || "-"}</div>
                    </td>
                    <td>
                      <div className="actions row-actions">
                        <button onClick={() => void onInspectDeployment("health", item.deployment_id)} type="button">健康</button>
                        {canViewResources ? (
                          <button onClick={() => void onInspectDeployment("logs", item.deployment_id)} type="button">日志</button>
                        ) : null}
                        {canManageDeployments ? (
                          <button disabled={actionBusy[`stop-${item.deployment_id}`]} onClick={() => void onDeploymentAction("stop", item.deployment_id)} type="button">停止</button>
                        ) : null}
                        {canManageDeployments ? (
                          <button disabled={actionBusy[`restart-${item.deployment_id}`]} onClick={() => void onDeploymentAction("restart", item.deployment_id)} type="button">重启</button>
                        ) : null}
                        {canManageDeployments ? (
                          <button disabled={actionBusy[`delete-${item.deployment_id}`]} onClick={() => void onDeploymentAction("delete", item.deployment_id)} type="button">删除</button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState
              icon="D"
              note={derived.deployments.length ? "当前筛选条件下没有匹配部署记录。" : "平台里还没有创建部署实例。"}
              title="暂无部署记录"
            />
          )}
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="已选健康" title="健康状态" />
          <div className="logs">{selectedHealth ? JSON.stringify(selectedHealth, null, 2) : "点击部署实例的“健康”按钮后在此显示。"} </div>
        </article>
        {canViewResources ? (
          <article className="card detail-card">
            <SectionHeader kicker="已选日志" title="运行日志" />
            <div className="logs">{selectedLogs?.content || "点击部署实例的“日志”按钮后在此显示。"}</div>
          </article>
        ) : null}
      </section>
    </>
  );
}

function BenchmarksPage({
  actionBusy,
  benchmarkForm,
  derived,
  onBenchmarkFormChange,
  onCreateBenchmark,
  onDeleteBenchmark,
}) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [scopeFilter, setScopeFilter] = useState("recent");
  const [compareModelId, setCompareModelId] = useState("all");
  const [paperModelDraftId, setPaperModelDraftId] = useState("all");
  const [paperModelId, setPaperModelId] = useState("all");
  const deferredSearch = useDeferredValue(search);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageBenchmarks = currentPermissions.includes("benchmarks.manage");

  const completedBenchmarks = useMemo(
    () => sortByCreatedDesc(derived.benchmarks).filter((item) => item.status === "completed"),
    [derived.benchmarks],
  );

  const compareModelOptions = useMemo(() => {
    return Array.from(new Set(completedBenchmarks.map((item) => item.model_id))).sort((left, right) =>
      left.localeCompare(right),
    );
  }, [completedBenchmarks]);

  useEffect(() => {
    if (!compareModelOptions.length) {
      setCompareModelId("all");
      setPaperModelDraftId("all");
      setPaperModelId("all");
      return;
    }
    if (compareModelId !== "all" && !compareModelOptions.includes(compareModelId)) {
      setCompareModelId(compareModelOptions[0]);
    }
    if (paperModelDraftId !== "all" && !compareModelOptions.includes(paperModelDraftId)) {
      setPaperModelDraftId(compareModelOptions[0]);
    }
    if (paperModelId !== "all" && !compareModelOptions.includes(paperModelId)) {
      setPaperModelId(compareModelOptions[0]);
    }
  }, [compareModelId, compareModelOptions, paperModelDraftId, paperModelId]);

  const filteredBenchmarks = useMemo(() => {
    let next = sortByCreatedDesc(derived.benchmarks).filter((item) =>
      matchesKeyword(
        [item.benchmark_id, item.label, item.model_id, item.deployment_id, item.profile_name, item.status],
        deferredSearch,
      ),
    );

    if (statusFilter !== "all") {
      next = next.filter((item) => item.status === statusFilter);
    }

    if (scopeFilter === "completed") {
      next = next.filter((item) => item.status === "completed");
    } else if (scopeFilter === "problematic") {
      next = next.filter((item) => item.failure_count > 0 || item.status === "failed");
    } else if (scopeFilter === "recent") {
      next = next.slice(0, 12);
    }

    return next;
  }, [deferredSearch, derived.benchmarks, scopeFilter, statusFilter]);

  const compareRows = useMemo(() => {
    const deploymentMap = new Map(derived.deployments.map((item) => [item.deployment_id, item]));
    const scoped = completedBenchmarks.filter((item) => compareModelId === "all" || item.model_id === compareModelId);
    const grouped = new Map();
    scoped.forEach((item) => {
      const key = `${item.model_id}::${item.profile_name}`;
      const current = grouped.get(key) || {
        model_id: item.model_id,
        profile_name: item.profile_name,
        items: [],
      };
      current.items.push(item);
      grouped.set(key, current);
    });

    return Array.from(grouped.values())
      .map(({ model_id, profile_name, items }) => {
        const totalRequests = items.reduce((sum, item) => sum + Number(item.total_requests || 0), 0);
        const successCount = items.reduce((sum, item) => sum + Number(item.success_count || 0), 0);
        const latest = sortByCreatedDesc(items)[0];
        const latestDeployment = deploymentMap.get(latest?.deployment_id);
        const execution = latestDeployment ? getDeploymentExecution(latestDeployment) : {};
        const providers = latestDeployment ? getDeploymentProviders(latestDeployment) : [];
        return {
          model_id,
          profile_name,
          run_count: items.length,
          avg_latency_ms: meanBy(items, (item) => item.avg_latency_ms),
          p95_latency_ms: meanBy(items, (item) => item.p95_latency_ms),
          throughput_qps: meanBy(items, (item) => item.throughput_qps),
          remote_avg_latency_ms: meanBy(items, (item) => item.remote_avg_latency_ms),
          success_rate: totalRequests ? successCount / totalRequests : null,
          avg_batch_size: meanBy(items, (item) => item.sample_response?.batch_size ?? 1),
          instance_count: execution.instance_count ?? null,
          intra_op_threads: execution.intra_op_threads ?? null,
          inter_op_threads: execution.inter_op_threads ?? null,
          max_length: execution.max_length ?? null,
          device: latestDeployment?.device || null,
          providers: providers.join(" / "),
          latest_benchmark_id: latest?.benchmark_id || "-",
          latest_created_at: latest?.created_at || "-",
        };
      })
      .sort((left, right) => {
        if (left.model_id !== right.model_id) {
          return left.model_id.localeCompare(right.model_id);
        }
        return left.profile_name.localeCompare(right.profile_name);
      });
  }, [compareModelId, completedBenchmarks, derived.deployments]);

  const bestLatencyRow = useMemo(() => {
    return [...compareRows]
      .filter((item) => item.avg_latency_ms !== null)
      .sort((left, right) => Number(left.avg_latency_ms) - Number(right.avg_latency_ms))[0] || null;
  }, [compareRows]);

  const bestThroughputRow = useMemo(() => {
    return [...compareRows]
      .filter((item) => item.throughput_qps !== null)
      .sort((left, right) => Number(right.throughput_qps) - Number(left.throughput_qps))[0] || null;
  }, [compareRows]);

  const thesisRows = useMemo(() => {
    const scoped = compareRows.filter((item) => paperModelId === "all" || item.model_id === paperModelId);
    return scoped.sort((left, right) => {
      if (left.model_id !== right.model_id) {
        return left.model_id.localeCompare(right.model_id);
      }
      return left.profile_name.localeCompare(right.profile_name);
    });
  }, [compareRows, paperModelId]);

  const selectedPaperModelRows = useMemo(() => {
    if (paperModelId === "all") {
      return [];
    }
    return thesisRows.filter((item) => item.model_id === paperModelId);
  }, [paperModelId, thesisRows]);

  const selectedPaperModelSummary = useMemo(() => {
    if (!selectedPaperModelRows.length) {
      return { bestLatency: null, bestThroughput: null, bestSuccess: null };
    }
    return {
      bestLatency:
        [...selectedPaperModelRows]
          .filter((item) => item.avg_latency_ms !== null)
          .sort((left, right) => Number(left.avg_latency_ms) - Number(right.avg_latency_ms))[0] || null,
      bestThroughput:
        [...selectedPaperModelRows]
          .filter((item) => item.throughput_qps !== null)
          .sort((left, right) => Number(right.throughput_qps) - Number(left.throughput_qps))[0] || null,
      bestSuccess:
        [...selectedPaperModelRows]
          .filter((item) => item.success_rate !== null)
          .sort((left, right) => Number(right.success_rate) - Number(left.success_rate))[0] || null,
    };
  }, [selectedPaperModelRows]);

  const chartMaxima = useMemo(() => {
    return {
      latency: Math.max(...selectedPaperModelRows.map((item) => Number(item.avg_latency_ms || 0)), 1),
      throughput: Math.max(...selectedPaperModelRows.map((item) => Number(item.throughput_qps || 0)), 1),
      success: Math.max(...selectedPaperModelRows.map((item) => Number(item.success_rate || 0)), 1),
    };
  }, [selectedPaperModelRows]);

  function handleExportFilteredBenchmarks() {
    if (!filteredBenchmarks.length) {
      return;
    }
    downloadCsvFile(
      "luckymodels-benchmarks.csv",
      filteredBenchmarks.map((item) => ({
        benchmark_id: item.benchmark_id,
        label: item.label || "",
        model_id: item.model_id,
        profile_name: item.profile_name,
        deployment_id: item.deployment_id,
        status: item.status,
        iterations: item.iterations,
        warmup_runs: item.warmup_runs,
        total_requests: item.total_requests,
        success_count: item.success_count,
        failure_count: item.failure_count,
        avg_latency_ms: item.avg_latency_ms ?? "",
        p95_latency_ms: item.p95_latency_ms ?? "",
        throughput_qps: item.throughput_qps ?? "",
        remote_avg_latency_ms: item.remote_avg_latency_ms ?? "",
        batch_size: item.sample_response?.batch_size ?? "",
        item_count: item.sample_response?.item_count ?? "",
        total_triple_count: item.sample_response?.total_triple_count ?? "",
        created_at: item.created_at,
        service_url: item.service_url,
      })),
    );
  }

  function handleExportFilteredBenchmarksJson() {
    if (!filteredBenchmarks.length) {
      return;
    }
    downloadJsonFile("luckymodels-benchmarks.json", filteredBenchmarks);
  }

  function handleExportCompareRows() {
    if (!compareRows.length) {
      return;
    }
    downloadCsvFile(
      "luckymodels-benchmark-profile-compare.csv",
      compareRows.map((item) => ({
        model_id: item.model_id,
        profile_name: item.profile_name,
        run_count: item.run_count,
        avg_latency_ms: item.avg_latency_ms?.toFixed(3) ?? "",
        p95_latency_ms: item.p95_latency_ms?.toFixed(3) ?? "",
        throughput_qps: item.throughput_qps?.toFixed(3) ?? "",
        remote_avg_latency_ms: item.remote_avg_latency_ms?.toFixed(3) ?? "",
        success_rate: item.success_rate ?? "",
        avg_batch_size: item.avg_batch_size?.toFixed(2) ?? "",
        latest_benchmark_id: item.latest_benchmark_id,
        latest_created_at: item.latest_created_at,
      })),
    );
  }

  function handleExportPaperRows() {
    if (!thesisRows.length) {
      return;
    }
    downloadCsvFile(
      "luckymodels-paper-table.csv",
      thesisRows.map((item) => ({
        model_id: item.model_id,
        profile_name: item.profile_name,
        device: item.device || "",
        providers: item.providers || "",
        batch_size: item.avg_batch_size?.toFixed(2) ?? "",
        instance_count: item.instance_count ?? "",
        intra_op_threads: item.intra_op_threads ?? "",
        inter_op_threads: item.inter_op_threads ?? "",
        max_length: item.max_length ?? "",
        avg_latency_ms: item.avg_latency_ms?.toFixed(3) ?? "",
        p95_latency_ms: item.p95_latency_ms?.toFixed(3) ?? "",
        throughput_qps: item.throughput_qps?.toFixed(3) ?? "",
        remote_avg_latency_ms: item.remote_avg_latency_ms?.toFixed(3) ?? "",
        success_rate: item.success_rate?.toFixed(4) ?? "",
        run_count: item.run_count,
        latest_benchmark_id: item.latest_benchmark_id,
        latest_created_at: item.latest_created_at,
      })),
    );
  }

  function handleGeneratePaperView() {
    const nextModelId = paperModelDraftId || "all";
    setPaperModelId(nextModelId);
    if (nextModelId !== "all") {
      setCompareModelId(nextModelId);
    }
  }

  return (
    <>
      {canManageBenchmarks ? (
      <section className="page-shell card section-shell">
        <SectionHeader kicker="新建基准" title="新建基准测试" note="为运行中的部署实例发起基准测试，并将结果写回平台。" />
        {derived.runningDeployments.length ? (
          <div className="form-grid">
            <label className="stack-field">
              <span>部署实例</span>
              <select
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, deploymentId: event.target.value }))}
                value={benchmarkForm.deploymentId}
              >
                {derived.runningDeployments.map((item) => (
                  <option key={item.deployment_id} value={item.deployment_id}>
                    {item.deployment_id} · {item.model_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="stack-field">
              <span>标签</span>
              <input
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, label: event.target.value }))}
                type="text"
                value={benchmarkForm.label}
              />
            </label>
            <label className="stack-field">
              <span>重复次数</span>
              <input
                min="1"
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, iterations: event.target.value }))}
                type="number"
                value={benchmarkForm.iterations}
              />
            </label>
            <label className="stack-field">
              <span>预热轮数</span>
              <input
                min="0"
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, warmupRuns: event.target.value }))}
                type="number"
                value={benchmarkForm.warmupRuns}
              />
            </label>
            <label className="stack-field stack-field-wide">
              <span>文本</span>
              <textarea
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, text: event.target.value }))}
                rows="4"
                value={benchmarkForm.text}
              />
            </label>
            <div className="actions">
              <button className="btn-primary" disabled={actionBusy["create-benchmark"]} onClick={() => void onCreateBenchmark()} type="button">
                新建基准
              </button>
            </div>
          </div>
        ) : (
          <EmptyState icon="B" title="暂无运行中的部署实例" note="先创建并启动部署实例，才能在这里发起基准测试。" />
        )}
      </section>
      ) : null}

      <section className="page-shell card table-shell">
        <SectionHeader kicker="基准记录" title="基准记录" />
        <div className="filter-panel filter-panel-table">
          <div className="filter-grid">
            <label className="stack-field">
              <span>搜索基准</span>
              <input
                onChange={(event) => setSearch(event.target.value)}
                placeholder="基准编号 / 标签 / 模型编号 / 部署编号"
                type="text"
                value={search}
              />
            </label>
            <label className="stack-field">
              <span>状态</span>
              <select onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
                <option value="all">全部状态</option>
                <option value="completed">已完成</option>
                <option value="planned">已规划</option>
                <option value="failed">失败</option>
              </select>
            </label>
            <label className="stack-field">
              <span>显示范围</span>
              <select onChange={(event) => setScopeFilter(event.target.value)} value={scopeFilter}>
                <option value="recent">最近 12 条</option>
                <option value="completed">仅已完成</option>
                <option value="problematic">仅异常记录</option>
                <option value="all">全部记录</option>
              </select>
            </label>
          </div>
          <div className="filter-foot">
            <div className="filter-meta">
              当前显示 {filteredBenchmarks.length} / {derived.benchmarks.length} 条 benchmark 记录
            </div>
            <div className="actions">
              <button disabled={!filteredBenchmarks.length} onClick={handleExportFilteredBenchmarks} type="button">
                导出当前 CSV
              </button>
              <button disabled={!filteredBenchmarks.length} onClick={handleExportFilteredBenchmarksJson} type="button">
                导出当前 JSON
              </button>
              <button
                onClick={() => {
                  setSearch("");
                  setStatusFilter("all");
                  setScopeFilter("recent");
                }}
                type="button"
              >
                恢复默认视图
              </button>
            </div>
          </div>
        </div>
        <div className="table-wrap">
          {filteredBenchmarks.length ? (
            <table>
              <thead>
                <tr>
                  <th>基准编号</th>
                  <th>模型</th>
                  <th>状态</th>
                  <th>吞吐量</th>
                  <th>平均延迟</th>
                  <th>P95</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {filteredBenchmarks.map((item) => (
                  <tr key={item.benchmark_id}>
                    <td>
                      <strong>{item.label || item.benchmark_id}</strong>
                      <div className="mono">{item.benchmark_id}</div>
                    </td>
                    <td>{item.model_id}</td>
                    <td><StatusPill status={item.status} /></td>
                    <td>{escapeValue(item.throughput_qps)}</td>
                    <td>{escapeValue(item.avg_latency_ms)}</td>
                    <td>{escapeValue(item.p95_latency_ms)}</td>
                    <td>
                      {canManageBenchmarks ? (
                        <button
                          disabled={actionBusy[`delete-benchmark-${item.benchmark_id}`]}
                          onClick={() => void onDeleteBenchmark(item.benchmark_id)}
                          type="button"
                        >
                          删除
                        </button>
                      ) : (
                        <span className="table-note">只读</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState
              icon="☰"
              note={derived.benchmarks.length ? "当前筛选条件下没有匹配基准记录。" : "创建基准测试后会在这里出现。"}
              title="暂无基准记录"
            />
          )}
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="论文截图专区"
          title="论文截图视图"
          note="选择单个模型后，一键生成更适合论文截图的配置模板对比条形图和结论卡片。导出文件仍会下载到你本地浏览器默认下载目录。"
          right={
            <div className="actions">
              <button className="btn-primary" disabled={!compareModelOptions.length} onClick={handleGeneratePaperView} type="button">
                生成论文视图
              </button>
            </div>
          }
        />
        <div className="filter-panel compare-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>论文模型</span>
              <select onChange={(event) => setPaperModelDraftId(event.target.value)} value={paperModelDraftId}>
                <option value="all">全部模型</option>
                {compareModelOptions.map((modelId) => (
                  <option key={modelId} value={modelId}>
                    {modelId}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="filter-foot">
            <div className="filter-meta">
              当前展示对象：{paperModelId === "all" ? "全部模型聚合结果" : paperModelId}
            </div>
          </div>
        </div>

        {paperModelId !== "all" && selectedPaperModelRows.length ? (
          <>
            <div className="paper-summary-grid">
              <article className="paper-hero-card">
                <div className="paper-hero-kicker">可用于论文</div>
                <h3>{paperModelId}</h3>
                <p>
                  当前页面已经按配置模板聚合了该模型的已完成基准记录，可直接用于论文截图、性能趋势说明和实验对比表整理。
                </p>
                <div className="tag-list">
                  <span className="tag">模板数：{selectedPaperModelRows.length}</span>
                  <span className="tag">运行次数：{selectedPaperModelRows.reduce((sum, item) => sum + item.run_count, 0)}</span>
                  <span className="tag">执行后端：{selectedPaperModelRows[0]?.providers || "待检测"}</span>
                </div>
              </article>
              <MetricCard
                title="最低平均延迟"
                value={selectedPaperModelSummary.bestLatency ? formatFixed(selectedPaperModelSummary.bestLatency.avg_latency_ms) : "-"}
                note={selectedPaperModelSummary.bestLatency ? selectedPaperModelSummary.bestLatency.profile_name : "暂无"}
              />
              <MetricCard
                title="最高吞吐量"
                value={selectedPaperModelSummary.bestThroughput ? formatFixed(selectedPaperModelSummary.bestThroughput.throughput_qps) : "-"}
                note={selectedPaperModelSummary.bestThroughput ? selectedPaperModelSummary.bestThroughput.profile_name : "暂无"}
              />
              <MetricCard
                title="最高成功率"
                value={selectedPaperModelSummary.bestSuccess ? formatPercent(selectedPaperModelSummary.bestSuccess.success_rate) : "-"}
                note={selectedPaperModelSummary.bestSuccess ? selectedPaperModelSummary.bestSuccess.profile_name : "暂无"}
              />
            </div>

            <div className="paper-chart-grid">
              <article className="paper-chart-card">
                <div className="paper-chart-title">平均延迟对比</div>
                <div className="paper-chart-note">单位毫秒，越低越好</div>
                <div className="paper-bar-list">
                  {selectedPaperModelRows.map((item) => (
                    <div className="paper-bar-row" key={`${item.model_id}-${item.profile_name}-latency`}>
                      <div className="paper-bar-label">
                        <strong>{item.profile_name}</strong>
                        <span>批量 {formatFixed(item.avg_batch_size, 2)} / 实例 {item.instance_count ?? "-"}</span>
                      </div>
                      <div className="paper-bar-track">
                        <div
                          className="paper-bar-fill paper-bar-latency"
                          style={{ width: `${Math.max((Number(item.avg_latency_ms || 0) / chartMaxima.latency) * 100, 6)}%` }}
                        />
                        <span className="paper-bar-value">{formatFixed(item.avg_latency_ms)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </article>

              <article className="paper-chart-card">
                <div className="paper-chart-title">吞吐量对比</div>
                <div className="paper-chart-note">单位 QPS，越高越好</div>
                <div className="paper-bar-list">
                  {selectedPaperModelRows.map((item) => (
                    <div className="paper-bar-row" key={`${item.model_id}-${item.profile_name}-throughput`}>
                      <div className="paper-bar-label">
                        <strong>{item.profile_name}</strong>
                        <span>算子内 {item.intra_op_threads ?? "-"} / 算子间 {item.inter_op_threads ?? "-"}</span>
                      </div>
                      <div className="paper-bar-track">
                        <div
                          className="paper-bar-fill paper-bar-throughput"
                          style={{ width: `${Math.max((Number(item.throughput_qps || 0) / chartMaxima.throughput) * 100, 6)}%` }}
                        />
                        <span className="paper-bar-value">{formatFixed(item.throughput_qps)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </article>

              <article className="paper-chart-card">
                <div className="paper-chart-title">成功率对比</div>
                <div className="paper-chart-note">按总请求数统计</div>
                <div className="paper-bar-list">
                  {selectedPaperModelRows.map((item) => (
                    <div className="paper-bar-row" key={`${item.model_id}-${item.profile_name}-success`}>
                      <div className="paper-bar-label">
                        <strong>{item.profile_name}</strong>
                        <span>最大长度 {item.max_length ?? "-"} / 设备 {item.device || "-"}</span>
                      </div>
                      <div className="paper-bar-track">
                        <div
                          className="paper-bar-fill paper-bar-success"
                          style={{ width: `${Math.max((Number(item.success_rate || 0) / chartMaxima.success) * 100, 6)}%` }}
                        />
                        <span className="paper-bar-value">{formatPercent(item.success_rate)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </>
        ) : (
          <EmptyState
            icon="S"
            title="先生成一个单模型论文视图"
            note="选择某个模型后点击“生成论文视图”，这里就会出现更适合截图的配置模板图表。"
          />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="模板对比"
          title="配置模板对比视图"
          note="按模型聚合已完成基准测试，直接比较不同配置模板的平均延迟、P95、吞吐量和成功率。CSV 导出会由浏览器下载到你本地电脑。"
          right={
            <div className="actions">
              <button disabled={!compareRows.length} onClick={handleExportCompareRows} type="button">
                导出对比 CSV
              </button>
            </div>
          }
        />
        <div className="filter-panel compare-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>比较模型</span>
              <select onChange={(event) => setCompareModelId(event.target.value)} value={compareModelId}>
                <option value="all">全部模型</option>
                {compareModelOptions.map((modelId) => (
                  <option key={modelId} value={modelId}>
                    {modelId}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="filter-foot">
            <div className="filter-meta">
              当前聚合 {compareRows.length} 个模型/模板组合，基于 {completedBenchmarks.length} 条已完成基准记录。
            </div>
          </div>
        </div>

        {compareRows.length ? (
          <>
            <div className="metrics">
              <MetricCard
                title="最低平均延迟"
                value={bestLatencyRow ? Number(bestLatencyRow.avg_latency_ms).toFixed(3) : "-"}
                note={bestLatencyRow ? `${bestLatencyRow.model_id} · ${bestLatencyRow.profile_name}` : "暂无"}
              />
              <MetricCard
                title="最高吞吐量"
                value={bestThroughputRow ? Number(bestThroughputRow.throughput_qps).toFixed(3) : "-"}
                note={bestThroughputRow ? `${bestThroughputRow.model_id} · ${bestThroughputRow.profile_name}` : "暂无"}
              />
              <MetricCard
                title="对比组合数"
                value={compareRows.length}
                note={compareModelId === "all" ? "全部模型" : compareModelId}
              />
              <MetricCard
                title="已完成记录"
                value={completedBenchmarks.length}
                note="仅“已完成”基准记录纳入聚合"
              />
            </div>
            <div className="table-wrap compare-table-wrap">
              <table>
                <thead>
                <tr>
                  <th>模型</th>
                  <th>配置模板</th>
                  <th>样本数</th>
                  <th>平均延迟</th>
                  <th>P95</th>
                  <th>吞吐量</th>
                  <th>成功率</th>
                  <th>平均批量</th>
                  <th>最新记录</th>
                </tr>
                </thead>
                <tbody>
                  {compareRows.map((item) => (
                    <tr key={`${item.model_id}-${item.profile_name}`}>
                      <td>{item.model_id}</td>
                      <td>{item.profile_name}</td>
                      <td>{item.run_count}</td>
                      <td>{item.avg_latency_ms !== null ? item.avg_latency_ms.toFixed(3) : "-"}</td>
                      <td>{item.p95_latency_ms !== null ? item.p95_latency_ms.toFixed(3) : "-"}</td>
                      <td>{item.throughput_qps !== null ? item.throughput_qps.toFixed(3) : "-"}</td>
                      <td>{formatPercent(item.success_rate)}</td>
                      <td>{item.avg_batch_size !== null ? item.avg_batch_size.toFixed(2) : "-"}</td>
                      <td>
                        <strong>{item.latest_benchmark_id}</strong>
                        <div className="table-note">{item.latest_created_at}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <EmptyState
            icon="P"
            title="暂无可对比的基准记录"
            note="先运行至少一条已完成的基准记录，配置模板对比视图才会出现聚合结果。"
          />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="论文表格"
          title="论文表格视图"
          note="这一块按模型聚合生成更规整的实验表，适合直接截图或抄到论文正文。"
          right={
            <div className="actions">
              <button disabled={!thesisRows.length} onClick={handleExportPaperRows} type="button">
                导出论文表 CSV
              </button>
            </div>
          }
        />
        {thesisRows.length ? (
          <div className="paper-table-shell">
            <table className="paper-table">
              <thead>
                <tr>
                  <th>模型</th>
                  <th>配置模板</th>
                  <th>批量</th>
                  <th>实例数</th>
                  <th>算子内线程</th>
                  <th>算子间线程</th>
                  <th>最大长度</th>
                  <th>平均延迟</th>
                  <th>P95</th>
                  <th>吞吐量</th>
                  <th>成功率</th>
                  <th>运行次数</th>
                </tr>
              </thead>
              <tbody>
                {thesisRows.map((item) => (
                  <tr key={`${item.model_id}-${item.profile_name}-paper`}>
                    <td>
                      <strong>{item.model_id}</strong>
                      <div className="table-note">{item.providers || "-"}</div>
                    </td>
                    <td>{item.profile_name}</td>
                    <td>{formatFixed(item.avg_batch_size, 2)}</td>
                    <td>{item.instance_count ?? "-"}</td>
                    <td>{item.intra_op_threads ?? "-"}</td>
                    <td>{item.inter_op_threads ?? "-"}</td>
                    <td>{item.max_length ?? "-"}</td>
                    <td>{formatFixed(item.avg_latency_ms)}</td>
                    <td>{formatFixed(item.p95_latency_ms)}</td>
                    <td>{formatFixed(item.throughput_qps)}</td>
                    <td>{formatPercent(item.success_rate)}</td>
                    <td>{item.run_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="paper-caption">
              建议截图方式：选定单模型后先截“论文截图视图”，再截当前“论文表格视图”，两张图就足够支撑一小节 profile 对比实验。
            </div>
          </div>
        ) : (
          <EmptyState
            icon="T"
            title="暂无论文表格数据"
            note="先运行已完成的基准记录，或者在上方生成某个模型的论文视图。"
          />
        )}
      </section>
    </>
  );
}

function ResourcesPage({ derived }) {
  const providerSet = new Set();
  const runtimeSnapshots = sortByCreatedDesc(derived.runningDeployments).map((item) => {
    const payload = getDeploymentHealthPayload(item);
    const providers = getDeploymentProviders(item);
    providers.forEach((provider) => providerSet.add(provider));
    return {
      item,
      payload,
      providers,
      execution: getDeploymentExecution(item),
    };
  });
  const gpuStatus = derived.gpuStatus || { available: false, gpu_count: 0, devices: [] };
  const activeGpuProcesses = (gpuStatus.devices || []).reduce(
    (sum, device) => sum + Number(device.active_processes?.length || 0),
    0,
  );
  const explicitGpuBindings = derived.deployments.filter((item) => Array.isArray(item.gpu_ids) && item.gpu_ids.length);

  return (
    <>
      <section className="page-shell">
        <div className="resource-grid">
          <article className="resource-card">
            <h4>控制面</h4>
            <div className="resource-meta">
              可推理模型：{derived.predictableModels.length}
              <br />
              运行中 deployment：{derived.runningDeployments.length}
              <br />
              当前地址：<span className="mono">127.0.0.1:38080</span>
            </div>
          </article>
          <article className="resource-card">
            <h4>GPU 资源</h4>
            <div className="resource-meta">
              GPU 数量：{gpuStatus.gpu_count || 0}
              <br />
              活跃 GPU 进程：{activeGpuProcesses}
              <br />
              显式绑定 deployment：{explicitGpuBindings.length}
            </div>
          </article>
          <article className="resource-card">
            <h4>执行后端</h4>
            <div className="resource-meta">
              {Array.from(providerSet).join(" / ") || "待检测"}
            </div>
          </article>
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="GPU 状态"
          title="GPU 状态总览"
          note="直接读取服务器 nvidia-smi，展示每张卡的显存占用、活跃进程，以及 LuckyModels 部署实例的绑定关系。"
        />
        {gpuStatus.available ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>GPU</th>
                  <th>显存</th>
                  <th>利用率</th>
                  <th>温度 / 功耗</th>
                  <th>绑定 Deployment</th>
                  <th>活跃进程</th>
                </tr>
              </thead>
              <tbody>
                {(gpuStatus.devices || []).map((device) => (
                  <tr key={device.uuid || device.index}>
                    <td>
                      <strong>GPU {device.index ?? "-"}</strong>
                      <div>{device.name || "-"}</div>
                      <div className="mono">{device.uuid || "-"}</div>
                    </td>
                    <td>
                      {formatFixed(device.memory_used_mb, 0)} / {formatFixed(device.memory_total_mb, 0)} MB
                    </td>
                    <td>{formatFixed(device.utilization_gpu_pct, 0)}%</td>
                    <td>
                      {formatFixed(device.temperature_c, 0)} C
                      <br />
                      {formatFixed(device.power_draw_w, 0)} W
                    </td>
                    <td>
                      {device.assigned_deployments?.length ? (
                        device.assigned_deployments.map((deployment) => (
                          <div key={deployment.deployment_id}>
                            <strong>{deployment.deployment_id}</strong>
                            <div className="table-note">{deployment.model_id} · {deployment.status}</div>
                          </div>
                        ))
                      ) : (
                        <span className="table-note">无显式绑定</span>
                      )}
                    </td>
                    <td>
                      {device.active_processes?.length ? (
                        device.active_processes.map((process) => (
                          <div key={`${process.pid}-${process.gpu_uuid}`}>
                            <strong>pid {process.pid}</strong>
                            <div className="table-note">
                              {process.used_gpu_memory_mb ?? "-"} MB · {process.deployment_id || process.process_name}
                            </div>
                          </div>
                        ))
                      ) : (
                        <span className="table-note">暂无活跃进程</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon="G"
            title="暂时无法读取 GPU 状态"
            note={gpuStatus.detail || "当前 API 没有拿到 nvidia-smi 返回结果。"}
          />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="运行快照"
          title="运行实例快照"
          note="这里展示运行中部署实例的真实 GPU 绑定、执行后端、线程配置、批量参数和健康状态。"
        />
        {runtimeSnapshots.length ? (
          <div className="list-grid runtime-grid">
            {runtimeSnapshots.map(({ item, payload, providers, execution }) => (
              <article className="list-card" key={item.deployment_id}>
                <div className="list-card-header">
                  <div>
                    <h4>{item.deployment_id}</h4>
                    <div className="mono">{item.model_id}</div>
                  </div>
                  <StatusPill status={item.status} />
                </div>
                <div className="runtime-meta">
                  配置模板：<strong>{item.profile_name}</strong>
                  <br />
                  设备 / 运行时：{item.device || "-"} / {item.profile_runtime || item.runtime || "-"}
                  <br />
                  是否使用 GPU：{formatBooleanLabel(item.use_gpu)}
                  <br />
                  分配 GPU：{formatGpuIds(getDeploymentAssignedGpuIds(item))}
                  <br />
                  可见 GPU：{payload.cuda_visible_devices || "全部"}
                  <br />
                  实例数：{execution.instance_count ?? 1}
                  <br />
                  批量大小：{execution.batch_size ?? 1}
                  <br />
                  算子内 / 算子间线程：{execution.intra_op_threads ?? 0} / {execution.inter_op_threads ?? 0}
                  <br />
                  执行后端：{providers.join(", ") || "待检测"}
                  <br />
                  进程存活：{formatBooleanLabel(item.health?.pid_alive ?? false)}
                  <br />
                  会话已加载：{formatBooleanLabel(payload.sessions_loaded ?? false)}
                  <br />
                  服务地址：<span className="mono">{item.service_url || "-"}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            icon="R"
            title="暂无运行快照"
            note="先启动一个部署实例，平台才会自动采集 healthz 并展示真实执行参数。"
          />
        )}
      </section>
    </>
  );
}

function LogsPage({ derived, selectedLogs }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader kicker="日志中心" title="日志中心" note="默认展示当前选中的部署实例日志，没有选择时回退到最近 CasRel 报告日志。" />
      <div className="logs">{selectedLogs?.content || derived.report.logs?.content || "先到部署页加载某个部署实例的日志。"}</div>
    </section>
  );
}
