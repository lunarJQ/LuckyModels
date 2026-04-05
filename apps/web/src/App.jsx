import { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";

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
  optimizeModelPackage,
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
  updateDeploymentPolicyGpuPool,
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
  optimizationOptions: null,
  runtimeCapabilities: null,
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

const BENCHMARK_SCENE_OPTIONS = [
  { value: "en_general", label: "英文通用场景" },
  { value: "zh_general", label: "中文通用场景" },
  { value: "zh_industrial", label: "中文工业设备场景" },
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
  "/task-routing-history": {
    title: "调度历史",
    note: "集中查看任务调度的历史累计统计、各目标长期权重、执行路径分布与最近决策记录。",
  },
  "/deployment-performance": {
    title: "表现变化",
    note: "集中查看所有 Deployment 在任务调度与推理实验两条链路中的表现层变化，即使从未被使用过的 Deployment 也会保留在总表里。",
  },
  "/deployment-policy": {
    title: "部署策略",
    note: "根据热温冷分层、冷启动风险与 GPU 压力，生成自适应部署建议，并支持定时自动优化。",
  },
  "/archive": {
    title: "结果反馈",
    note: "集中查看最近的推理与调度结果，并以用户反馈为主补充平台在线学习证据。",
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
  const optimization = getOptimizationSummaryForModel(modelId, details);
  const useGpu = !isCpuPinnedOptimization(optimization.optimizationType);
  return {
    modelId,
    profileName: profiles.includes("default") ? "default" : profiles[0] || "",
    useGpu,
    gpuIdsText: "",
    providerPreference: "cpu",
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

function buildInitialModelOptimizationForm(models, optimizationOptions) {
  const preferred = models.find((model) => model.predict_ready) || models[0];
  const optionList = optimizationOptions?.options || [];
  return {
    modelId: preferred?.model_id || "",
    optimizationType: optionList[0]?.code || "ort_graph",
    alias: "",
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

function getOptimizationSummaryForModel(modelId, details) {
  const metadata = details?.[modelId]?.package_metadata || {};
  const rawOptimizationType = String(metadata.optimization_type || "").trim();
  const hintText = [
    rawOptimizationType,
    metadata.optimization_label,
    details?.[modelId]?.summary?.model_name,
    details?.[modelId]?.model_name,
    modelId,
  ]
    .map((item) => String(item || "").trim())
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  let inferredOptimizationType = rawOptimizationType;
  if (!inferredOptimizationType) {
    if (hintText.includes("int8")) {
      inferredOptimizationType = "int8_dynamic";
    } else if (hintText.includes("fp16")) {
      inferredOptimizationType = "fp16";
    } else if (hintText.includes("ort-format") || hintText.includes("[ort-format]")) {
      inferredOptimizationType = "ort_format";
    } else if (hintText.includes("optgraph") || hintText.includes("ort-optimized")) {
      inferredOptimizationType = "ort_graph";
    }
  }
  return {
    optimizationType: inferredOptimizationType,
    optimizationLabel: String(metadata.optimization_label || "").trim(),
    sourceModelId: String(metadata.source_model_id || "").trim(),
    isGenerated: String(metadata.origin || "").trim() === "generated",
  };
}

function isCpuPinnedOptimization(optimizationType = "") {
  return String(optimizationType || "").trim().toLowerCase().includes("int8");
}

function getCreatedAtTimestamp(value) {
  const timestamp = Date.parse(String(value || "").trim());
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function compareCreatedAtDesc(left, right) {
  const timestampDiff = getCreatedAtTimestamp(right?.created_at) - getCreatedAtTimestamp(left?.created_at);
  if (timestampDiff !== 0) {
    return timestampDiff;
  }
  return String(right?.benchmark_id || "").localeCompare(String(left?.benchmark_id || ""));
}

function selectLatestPreferredBenchmark(items) {
  const completed = (items || []).filter((item) => item?.status === "completed");
  if (!completed.length) {
    return null;
  }
  const effective = completed.filter((item) => item?.is_effective);
  const preferredPool = effective.length ? effective : completed;
  return [...preferredPool].sort(compareCreatedAtDesc)[0] || null;
}

function buildLatestPreferredBenchmarkCompareRows(benchmarks, deployments, modelDetails, compareModelId = "all") {
  const deploymentMap = new Map((deployments || []).map((item) => [item.deployment_id, item]));
  const scoped = (benchmarks || []).filter(
    (item) => item?.status === "completed" && (compareModelId === "all" || item?.model_id === compareModelId),
  );
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
      const selected = selectLatestPreferredBenchmark(items);
      if (!selected) {
        return null;
      }
      const latestDeployment = deploymentMap.get(selected.deployment_id);
      const execution = latestDeployment ? getDeploymentExecution(latestDeployment) : {};
      const providers = latestDeployment ? getDeploymentProviders(latestDeployment) : [];
      const optimizationSummary = getOptimizationSummaryForModel(model_id, modelDetails);
      const totalRequests = Number(selected.total_requests || 0);
      const successCount = Number(selected.success_count || 0);
      return {
        model_id,
        profile_name,
        run_count: items.length,
        avg_latency_ms: selected.avg_latency_ms ?? null,
        p95_latency_ms: selected.p95_latency_ms ?? null,
        throughput_qps: selected.throughput_qps ?? null,
        remote_avg_latency_ms: selected.remote_avg_latency_ms ?? null,
        success_rate: totalRequests ? successCount / totalRequests : null,
        avg_batch_size: selected.sample_response?.batch_size ?? 1,
        instance_count: execution.instance_count ?? null,
        intra_op_threads: execution.intra_op_threads ?? null,
        inter_op_threads: execution.inter_op_threads ?? null,
        max_length: execution.max_length ?? null,
        device: latestDeployment?.device || null,
        providers: providers.join(" / "),
        latest_benchmark_id: selected.benchmark_id || "-",
        latest_created_at: selected.created_at || "-",
        latest_scene: selected.scene || "",
        selection_mode: selected.is_effective ? "latest_effective" : "latest_completed",
        selection_label: selected.is_effective ? "最新有效记录" : "最新完成记录",
        selected_benchmark_id: selected.benchmark_id || "-",
        selected_benchmark_created_at: selected.created_at || "-",
        selected_benchmark_is_effective: Boolean(selected.is_effective),
        optimization_type: optimizationSummary.optimizationType,
        optimization_label: optimizationSummary.optimizationLabel,
        source_model_id: optimizationSummary.sourceModelId,
        comparison_root_model_id: optimizationSummary.sourceModelId || model_id,
        is_generated_variant: optimizationSummary.isGenerated,
      };
    })
    .filter(Boolean)
    .sort((left, right) => {
      if (left.model_id !== right.model_id) {
        return left.model_id.localeCompare(right.model_id);
      }
      return left.profile_name.localeCompare(right.profile_name);
    });
}

function buildOptimizationBenchmarkCompareRows(compareRows, optimizationSourceModelId = "all") {
  function selectRepresentativeRow(rows) {
    if (!rows.length) {
      return null;
    }
    const minLatency = Math.min(...rows.map((item) => Number(item.avg_latency_ms || Number.POSITIVE_INFINITY)));
    const maxThroughput = Math.max(...rows.map((item) => Number(item.throughput_qps || 0)));
    const maxSuccess = Math.max(...rows.map((item) => Number(item.success_rate || 0)));

    return [...rows].sort((left, right) => {
      const leftLatency = Number(left.avg_latency_ms || Number.POSITIVE_INFINITY);
      const rightLatency = Number(right.avg_latency_ms || Number.POSITIVE_INFINITY);
      const leftThroughput = Number(left.throughput_qps || 0);
      const rightThroughput = Number(right.throughput_qps || 0);
      const leftSuccess = Number(left.success_rate || 0);
      const rightSuccess = Number(right.success_rate || 0);
      const leftScore =
        (Number.isFinite(leftLatency) && leftLatency > 0 && Number.isFinite(minLatency) ? minLatency / leftLatency : 0)
        + (maxThroughput > 0 ? leftThroughput / maxThroughput : 0)
        + (maxSuccess > 0 ? leftSuccess / maxSuccess : 0);
      const rightScore =
        (Number.isFinite(rightLatency) && rightLatency > 0 && Number.isFinite(minLatency) ? minLatency / rightLatency : 0)
        + (maxThroughput > 0 ? rightThroughput / maxThroughput : 0)
        + (maxSuccess > 0 ? rightSuccess / maxSuccess : 0);

      if (rightScore !== leftScore) {
        return rightScore - leftScore;
      }
      if (leftLatency !== rightLatency) {
        return leftLatency - rightLatency;
      }
      if (rightThroughput !== leftThroughput) {
        return rightThroughput - leftThroughput;
      }
      return String(left.profile_name || "").localeCompare(String(right.profile_name || ""));
    })[0] || null;
  }

  const grouped = new Map();
  (compareRows || []).forEach((row) => {
    const rootModelId = row.comparison_root_model_id || row.model_id;
    const current = grouped.get(rootModelId) || [];
    current.push(row);
    grouped.set(rootModelId, current);
  });

  const rows = [];
  grouped.forEach((items, rootModelId) => {
    const hasGenerated = items.some((item) => item.is_generated_variant);
    if (!hasGenerated || items.length <= 1) {
      return;
    }
    if (optimizationSourceModelId !== "all" && rootModelId !== optimizationSourceModelId) {
      return;
    }
    const baselineRows = items.filter((item) => !item.is_generated_variant && item.model_id === rootModelId);
    const representativeBaselineRow = selectRepresentativeRow(baselineRows);
    const bestBaselineLatency = representativeBaselineRow?.avg_latency_ms ?? null;
    const bestBaselineThroughput = representativeBaselineRow?.throughput_qps ?? null;
    const generatedRows = items.filter((item) => item.is_generated_variant);
    const representativeGeneratedRows = Array.from(
      generatedRows.reduce((grouped, item) => {
        const optimizationKey = String(item.optimization_type || "generated").trim().toLowerCase() || "generated";
        const current = grouped.get(optimizationKey) || [];
        current.push(item);
        grouped.set(optimizationKey, current);
        return grouped;
      }, new Map()).values(),
    )
      .map((group) => selectRepresentativeRow(group))
      .filter(Boolean);
    const visibleItems = representativeBaselineRow
      ? [representativeBaselineRow, ...representativeGeneratedRows]
      : representativeGeneratedRows;

    visibleItems.forEach((item) => {
      const latencyValue = Number(item.avg_latency_ms);
      const throughputValue = Number(item.throughput_qps);
      const latencyGainPct =
        item.is_generated_variant && Number.isFinite(latencyValue) && Number.isFinite(bestBaselineLatency) && bestBaselineLatency > 0
          ? ((bestBaselineLatency - latencyValue) / bestBaselineLatency) * 100
          : null;
      const throughputGainPct =
        item.is_generated_variant && Number.isFinite(throughputValue) && Number.isFinite(bestBaselineThroughput) && bestBaselineThroughput > 0
          ? ((throughputValue - bestBaselineThroughput) / bestBaselineThroughput) * 100
          : null;
      rows.push({
        ...item,
        baseline_best_latency_ms: bestBaselineLatency,
        baseline_best_throughput_qps: bestBaselineThroughput,
        baseline_profile_name: representativeBaselineRow?.profile_name || "",
        baseline_device: representativeBaselineRow?.device || "",
        baseline_benchmark_id: representativeBaselineRow?.latest_benchmark_id || "-",
        latency_gain_pct: latencyGainPct,
        throughput_gain_pct: throughputGainPct,
        variant_label: item.is_generated_variant ? (item.optimization_label || "平台生成版本") : "原始最优版本",
        source_root_model_id: rootModelId,
      });
    });
  });

  return rows.sort((left, right) => {
    if (left.source_root_model_id !== right.source_root_model_id) {
      return left.source_root_model_id.localeCompare(right.source_root_model_id);
    }
    if (left.model_id !== right.model_id) {
      return left.model_id.localeCompare(right.model_id);
    }
    return left.profile_name.localeCompare(right.profile_name);
  });
}

function getRuntimeProviderCapability(runtimeCapabilities, providerName) {
  return (runtimeCapabilities?.providers || []).find((item) => item.provider === providerName) || null;
}

function isRuntimeProviderUsable(runtimeCapabilities, providerName) {
  return Boolean(getRuntimeProviderCapability(runtimeCapabilities, providerName)?.runtime_usable);
}

function getDeploymentBackendOptions(runtimeCapabilities, useGpu, optimizationType = "") {
  if (isCpuPinnedOptimization(optimizationType)) {
    return [{
      value: "cpu",
      label: "CPUExecutionProvider",
      note: "INT8 动态量化版本默认固定为 CPU 部署，避免 GPU 路径出现明显性能劣化。",
    }];
  }
  if (!useGpu) {
    return [{ value: "cpu", label: "CPUExecutionProvider", note: "当前部署按 CPU 路径运行。" }];
  }
  const options = [];
  if (isRuntimeProviderUsable(runtimeCapabilities, "CUDAExecutionProvider")) {
    options.push({
      value: "cuda",
      label: "CUDA Execution Provider",
      note: "使用 ONNX Runtime 的 CUDA EP 运行 GPU 推理。",
    });
  }
  return options;
}

function hasGpuDeploymentBackend(runtimeCapabilities, optimizationType = "") {
  return getDeploymentBackendOptions(runtimeCapabilities, true, optimizationType).some((item) => item.value !== "cpu");
}

function resolveDeploymentProviderPreference(currentPreference, runtimeCapabilities, useGpu, optimizationType = "") {
  const options = getDeploymentBackendOptions(runtimeCapabilities, useGpu, optimizationType);
  if (!options.length) {
    return "cpu";
  }
  if (options.some((item) => item.value === currentPreference)) {
    return currentPreference;
  }
  return options[0].value;
}

function getSuggestedDeploymentPreset(modelId, details, recommendation = null) {
  const optimization = getOptimizationSummaryForModel(modelId, details);
  const useGpu = isCpuPinnedOptimization(optimization.optimizationType)
    ? false
    : (recommendation?.recommended_use_gpu ?? true);
  return {
    useGpu,
    gpuIds: useGpu ? (recommendation?.recommended_gpu_ids || []) : [],
    deviceLabel: recommendation?.recommended_device_label || (useGpu ? "GPU" : "CPU"),
    reason: optimization.optimizationLabel
      ? `当前模型为${optimization.optimizationLabel}版本，可继续按推荐设备部署。`
      : "当前模型按部署策略推荐设备创建。",
  };
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

function normalizePredictionFeedbackReason(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "missing_too_many" || normalized === "false_positive_too_many" || normalized === "latency_high") {
    return normalized;
  }
  return "";
}

function parsePredictionFeedbackReasons(value) {
  if (Array.isArray(value)) {
    return Array.from(
      new Set(value.map((item) => normalizePredictionFeedbackReason(item)).filter(Boolean)),
    );
  }
  return Array.from(
    new Set(
      String(value || "")
        .split(",")
        .map((item) => normalizePredictionFeedbackReason(item))
        .filter(Boolean),
    ),
  );
}

function serializePredictionFeedbackReasons(value) {
  const order = ["missing_too_many", "false_positive_too_many", "latency_high"];
  const reasonSet = new Set(parsePredictionFeedbackReasons(value));
  return order.filter((item) => reasonSet.has(item)).join(",");
}

function formatPredictionFeedbackReasonLabel(value) {
  const normalized = normalizePredictionFeedbackReason(value);
  if (normalized === "missing_too_many") {
    return "漏抽多";
  }
  if (normalized === "false_positive_too_many") {
    return "误抽多";
  }
  if (normalized === "latency_high") {
    return "延迟高";
  }
  return value ? String(value) : "";
}

function formatPredictionFeedbackReasonSummary(value) {
  const labels = parsePredictionFeedbackReasons(value).map((item) => formatPredictionFeedbackReasonLabel(item));
  return labels.join("、");
}

function formatPredictionFeedbackSummary(status, note) {
  const statusLabel = formatPredictionFeedbackLabel(status);
  const reasonLabel = formatPredictionFeedbackReasonSummary(note);
  if (normalizePredictionFeedbackStatus(status) === "unsatisfied" && reasonLabel) {
    return `${statusLabel} · ${reasonLabel}`;
  }
  return statusLabel;
}

function formatArchiveSourceLabel(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "task_routing" || normalized === "task_routing_decision") {
    return "任务调度";
  }
  return "在线推理";
}

function getArchiveImpactSignals(archive) {
  const status = normalizePredictionFeedbackStatus(archive?.feedback_status);
  const reasons = parsePredictionFeedbackReasons(archive?.feedback_note);
  if (status === "satisfied") {
    return ["满意反馈", "整体正向"];
  }
  if (status === "unsatisfied") {
    const signals = [];
    if (reasons.includes("missing_too_many")) {
      signals.push("覆盖率承压");
    }
    if (reasons.includes("false_positive_too_many")) {
      signals.push("准确率/稳定性承压");
    }
    if (reasons.includes("latency_high")) {
      signals.push("时延维度承压");
    }
    return signals.length ? signals : ["不满意", "存在负向校准"];
  }
  return ["无显式反馈", "仅留痕"];
}

function collectArchiveModelHints(archive) {
  const values = [
    archive?.model_id,
    archive?.response_payload?.model_id,
    ...(archive?.task_routing_selected_model_ids || []),
    ...(archive?.response_payload?.selected_model_ids || []),
  ];
  return values.map((item) => String(item || "").trim().toLowerCase()).filter(Boolean);
}

function inferArchiveLanguage(archive) {
  const payloadLanguage = String(archive?.response_payload?.language || "").trim().toLowerCase();
  if (payloadLanguage === "en" || payloadLanguage === "zh") {
    return payloadLanguage;
  }
  const hints = collectArchiveModelHints(archive);
  if (hints.some((item) => item.includes("-en-"))) {
    return "en";
  }
  if (hints.some((item) => item.includes("-zh-"))) {
    return "zh";
  }
  return "";
}

function inferArchiveDomain(archive) {
  const hints = [
    ...collectArchiveModelHints(archive),
    String(archive?.scene || "").trim().toLowerCase(),
    String(archive?.task_routing_objective || "").trim().toLowerCase(),
  ];
  if (hints.some((item) => item.includes("equipmentmix15k") || item.includes("industrial"))) {
    return "industrial";
  }
  return "general";
}

function archiveMatchesTaskRoutingScene(archive, sceneValue) {
  const language = inferArchiveLanguage(archive);
  const domain = inferArchiveDomain(archive);
  if (sceneValue === "en_general") {
    return language === "en";
  }
  if (sceneValue === "zh_general") {
    return language === "zh" && domain !== "industrial";
  }
  if (sceneValue === "zh_industrial") {
    return language === "zh" && domain === "industrial";
  }
  return true;
}

function formatWeightSnapshot(snapshot) {
  const entries = Object.entries(snapshot || {});
  if (!entries.length) {
    return "-";
  }
  return entries
    .sort((left, right) => Number(right[1] || 0) - Number(left[1] || 0))
    .map(([key, value]) => `${key}=${Number(value).toFixed(2)}`)
    .join(" / ");
}

function getWeightSnapshotEntries(snapshot) {
  return Object.entries(snapshot || {}).sort((left, right) => Number(right[1] || 0) - Number(left[1] || 0));
}

function getSignedWeightSnapshotEntries(snapshot) {
  return Object.entries(snapshot || {})
    .filter(([, value]) => Math.abs(Number(value) || 0) > 0.000001)
    .sort((left, right) => Math.abs(Number(right[1] || 0)) - Math.abs(Number(left[1] || 0)));
}

function formatWeightDimensionLabel(value) {
  const labels = {
    accuracy: "准确率",
    latency: "时延",
    coverage: "覆盖率",
    stability: "稳定性",
    resource: "资源",
    domain: "领域",
  };
  return labels[String(value || "").trim()] || value || "-";
}

function formatObjectiveLabel(value) {
  const labels = {
    accuracy: "高准确",
    coverage: "高覆盖",
    latency: "低时延",
    balanced: "综合平衡",
    resource: "低成本",
    domain: "领域优先",
  };
  return labels[String(value || "").trim()] || value || "-";
}

function formatSignedNumber(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (number > 0) {
    return `+${number.toFixed(digits)}`;
  }
  return number.toFixed(digits);
}

function formatSignedPercentDelta(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (number > 0) {
    return `+${number.toFixed(digits)}%`;
  }
  return `${number.toFixed(digits)}%`;
}

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

const executionPathNotes = {
  none: "尚未形成真实调度路径，通常出现在页面初始状态或没有可用 deployment 时。",
  single_low_latency: "只选一个低时延 deployment，优先追求快速返回，适合交互式在线请求。",
  parallel_multi_union: "多个 deployment 并行执行，最后按并集补全结果，适合更看重召回的任务。",
  single_low_cost: "只选一个低资源 deployment，优先控制 GPU/CPU 成本，适合资源紧张场景。",
  cost_aware_primary_plus_reviewer: "先用低成本主模型出结果，再让一个复核模型做补充或校验，兼顾成本与稳定性。",
  domain_priority_single: "优先选择领域最匹配的单模型直出，适合工业文本这类领域约束较强的输入。",
  domain_priority_crosscheck: "先选领域模型，再让其他候选做交叉校验，适合需要更稳妥结果的领域场景。",
  single_balanced: "只选一个综合表现较均衡的 deployment，作为准确率、时延和资源之间的折中路径。",
  consensus_with_backfill: "先保留多模型共识结果，再用补全策略追回遗漏三元组，适合平衡准确与覆盖。",
  single_high_confidence: "优先选择高置信度 deployment，适合论文展示或对误抽更敏感的任务。",
  multi_consensus_vote: "多个 deployment 同时执行，以投票或共识方式输出结果，适合高准确或低误抽场景。",
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

const fusionPolicyNotes = {
  empty: "当前没有发生融合，通常对应未执行或单次流程未产出可融合结果。",
  single_primary: "只保留主模型输出，不再和其他候选结果做合并。",
  union_all: "把多个模型结果直接做并集，优点是覆盖高，缺点是误抽风险也会增大。",
  primary_with_fallback: "以主模型结果为主，再用次模型做回填，适合在保证稳定性的前提下补召回。",
  domain_priority: "优先保留领域模型的结果，其余模型只做补充，适合工业场景这类领域文本。",
  consensus_backfill: "先保留共识三元组，再将高价值遗漏结果补入，适合兼顾准确和覆盖。",
  majority_vote: "以多数模型共同支持的结果为主，优先压低误抽，适合高准确需求。",
};

function formatExecutionPathLabel(value) {
  return executionPathLabels[String(value || "").trim()] || value || "-";
}

function formatFusionPolicyLabel(value) {
  return fusionPolicyLabels[String(value || "").trim()] || value || "-";
}

function formatExecutionPathNote(value) {
  return executionPathNotes[String(value || "").trim()] || "该路径当前没有补充说明。";
}

function formatFusionPolicyNote(value) {
  return fusionPolicyNotes[String(value || "").trim()] || "该融合策略当前没有补充说明。";
}

function buildWeightSnapshotChipClass(variantClass, value, withSign = false) {
  const classNames = ["weight-snapshot-chip"];
  if (variantClass) {
    classNames.push(variantClass);
  }
  if (withSign) {
    const numericValue = Number(value || 0);
    if (numericValue > 0) {
      classNames.push("weight-snapshot-chip-positive");
    } else if (numericValue < 0) {
      classNames.push("weight-snapshot-chip-negative");
    } else {
      classNames.push("weight-snapshot-chip-neutral");
    }
  }
  return classNames.join(" ");
}

function renderWeightSnapshotGrid(snapshot, keyPrefix, { variantClass = "", signed = false } = {}) {
  const entries = getWeightSnapshotEntries(snapshot || {});
  if (!entries.length) {
    return <span className="table-note">无</span>;
  }
  return (
    <div className="weight-snapshot-grid">
      {entries.map(([key, value]) => (
        <span
          className={buildWeightSnapshotChipClass(variantClass, value, signed)}
          key={`${keyPrefix}-${key}`}
        >
          <strong>{signed ? formatWeightDimensionLabel(key) : key}</strong>
          <span>{signed ? formatSignedNumber(value, 3) : Number(value).toFixed(2)}</span>
        </span>
      ))}
    </div>
  );
}

function buildSnapshotDelta(currentSnapshot, baselineSnapshot) {
  const keys = new Set([
    ...Object.keys(currentSnapshot || {}),
    ...Object.keys(baselineSnapshot || {}),
  ]);
  const delta = {};
  keys.forEach((key) => {
    const value = Number((currentSnapshot || {})[key] || 0) - Number((baselineSnapshot || {})[key] || 0);
    if (Math.abs(value) > 0.000001) {
      delta[key] = value;
    }
  });
  return delta;
}

function applySnapshotDelta(baselineSnapshot, deltaSnapshot) {
  const keys = new Set([
    ...Object.keys(baselineSnapshot || {}),
    ...Object.keys(deltaSnapshot || {}),
  ]);
  const nextSnapshot = {};
  keys.forEach((key) => {
    const value = Number((baselineSnapshot || {})[key] || 0) + Number((deltaSnapshot || {})[key] || 0);
    if (Math.abs(value) > 0.000001) {
      nextSnapshot[key] = value;
    }
  });
  return nextSnapshot;
}

function formatDeploymentPolicyActionLabel(value) {
  const labels = {
    scan_only: "策略扫描",
    rebalance_only: "资源受限优化 · 局部重平衡",
    rebalance_gpu: "资源受限优化 · GPU 重平衡",
    migrate_to_cpu: "资源受限优化 · 迁移到 CPU",
    skip_cpu_migration: "资源受限优化 · 跳过 CPU 迁移",
    recover_to_gpu: "资源恢复优化 · 回迁到 GPU",
    recover_profile: "资源恢复优化 · profile 优化",
    restart_high_gpu_memory: "显存守护 · 自动重启",
    gpu_pool_update: "GPU 范围变更",
    gpu_pool_scope_change: "GPU 范围变更 · 忙碌保护",
    gpu_pool_reassign_gpu: "GPU 范围变更 · 换卡迁移",
    gpu_pool_fallback_cpu: "GPU 范围变更 · CPU 兜底",
    gpu_pool_promote_gpu: "GPU 范围变更 · 回迁到 GPU",
  };
  return labels[String(value || "").trim()] || value || "-";
}

function buildPaginationItems(currentPage, totalPages) {
  const safeCurrentPage = Math.max(1, Math.min(currentPage, totalPages));
  const start = Math.max(1, safeCurrentPage - 2);
  const end = Math.min(totalPages, safeCurrentPage + 2);
  const items = [];
  for (let page = start; page <= end; page += 1) {
    items.push(page);
  }
  return items;
}

function getStickyTopOffset() {
  if (typeof document === "undefined") {
    return 0;
  }
  const topbar = document.querySelector(".topbar");
  const topbarHeight = topbar ? topbar.getBoundingClientRect().height : 0;
  return topbarHeight + 16;
}

function scrollElementToViewportTop(element) {
  if (typeof window === "undefined" || !element) {
    return;
  }
  const targetTop = window.scrollY + element.getBoundingClientRect().top - getStickyTopOffset();
  window.scrollTo({
    top: Math.max(0, targetTop),
    behavior: "smooth",
  });
}

function resetPageScrollToTop() {
  if (typeof window === "undefined") {
    return;
  }
  window.scrollTo(0, 0);
  if (typeof document !== "undefined") {
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }
}

function PaginationBar({ currentPage, totalPages, pageSize, totalItems, onChange }) {
  if (totalPages <= 1) {
    return null;
  }
  const pageItems = buildPaginationItems(currentPage, totalPages);

  function handlePageChange(nextPage, event) {
    onChange(nextPage);
    const tableWrap = event?.currentTarget?.closest(".table-wrap");
    if (tableWrap) {
      window.requestAnimationFrame(() => {
        scrollElementToViewportTop(tableWrap);
      });
    }
  }

  return (
    <div className="pagination-bar">
      <div className="pagination-meta">
        共 {totalItems} 条，每页 {pageSize} 条，当前第 {currentPage} / {totalPages} 页
      </div>
      <div className="actions pagination-actions">
        <button disabled={currentPage <= 1} onClick={(event) => handlePageChange(1, event)} type="button">首页</button>
        <button disabled={currentPage <= 1} onClick={(event) => handlePageChange(currentPage - 1, event)} type="button">上一页</button>
        {pageItems.map((page) => (
          <button
            key={page}
            className={page === currentPage ? "button-active" : ""}
            onClick={(event) => handlePageChange(page, event)}
            type="button"
          >
            {page}
          </button>
        ))}
        <button disabled={currentPage >= totalPages} onClick={(event) => handlePageChange(currentPage + 1, event)} type="button">下一页</button>
        <button disabled={currentPage >= totalPages} onClick={(event) => handlePageChange(totalPages, event)} type="button">末页</button>
      </div>
    </div>
  );
}

function useTablePagination(items, pageSize = 5) {
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(page, totalPages);

  useEffect(() => {
    if (page !== safePage) {
      setPage(safePage);
    }
  }, [page, safePage]);

  const pagedItems = useMemo(
    () => items.slice((safePage - 1) * pageSize, safePage * pageSize),
    [items, pageSize, safePage],
  );

  return {
    page: safePage,
    setPage,
    totalPages,
    pageSize,
    totalItems: items.length,
    pagedItems,
  };
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

function formatProviderPreferenceLabel(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (normalized === "cuda") {
    return "CUDA Execution Provider";
  }
  if (normalized === "cpu") {
    return "CPUExecutionProvider";
  }
  return normalized || "自动";
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

function resolveBenchmarkSceneForModel(model) {
  const language = String(model?.language || "").trim().toLowerCase();
  const datasetName = String(model?.dataset_name || "").trim().toLowerCase();
  if (language === "en") {
    return "en_general";
  }
  if (
    datasetName.includes("industrial") ||
    datasetName.includes("equipmentmix") ||
    datasetName.includes("mix15k") ||
    datasetName.includes("mfgdominant")
  ) {
    return "zh_industrial";
  }
  return "zh_general";
}

function formatBenchmarkSceneLabel(value) {
  const matched = BENCHMARK_SCENE_OPTIONS.find((item) => item.value === String(value || "").trim());
  return matched?.label || value || "未标注场景";
}

function buildInitialBenchmarkForm(deployments, models = []) {
  const running = deployments.find((item) => item.status === "running") || deployments[0];
  const matchingModel = models.find((item) => item.model_id === running?.model_id);
  return {
    deploymentId: running?.deployment_id || "",
    scene: resolveBenchmarkSceneForModel(matchingModel),
    text: "",
    label: "",
    iterations: 5,
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
    objective: "accuracy",
    language: "auto",
    maxModels: "",
    scene: "",
    modelFamily: "",
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
    "models.import": "上传模型",
    "deployments.view": "查看部署",
    "deployments.manage": "管理部署",
    "benchmarks.view": "查看基准",
    "benchmarks.manage": "运行基准",
    "predictions.view": "查看结果反馈",
    "predictions.feedback": "提交结果反馈",
    "task_routing.run": "运行任务调度",
    "strategies.manage": "运行策略",
    "resources.view": "查看资源",
    "archive.export": "导出归档",
    "users.view": "查看权限",
    "users.manage": "管理用户",
  };
  return mapping[permissionId] || permissionId || "-";
}

function getRoleScopeSummary(roleId) {
  const mapping = {
    admin: "可访问全部模块，负责模型接入、部署治理、任务调度分析、即时学习反馈查看、资源运维和系统管理。",
    user: "可查看资产概览、模型库、能力矩阵、制品包、部署记录与基准结果，可上传模型、运行推理实验与任务调度、提交结果反馈；任务调度页默认采用简化视图，不显示解释区和即时学习反馈，也不能进入调度历史、表现变化、部署策略、运行资源、日志中心和系统管理。",
  };
  return mapping[roleId] || "当前角色的页面范围按权限项动态决定。";
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
  if (canAccessPath("/", currentPermissions, currentUser)) {
    return "/";
  }
  const visibleGroups = getVisibleNavGroups(currentPermissions, currentUser);
  return visibleGroups[0]?.items[0]?.path || "/";
}

export default function App() {
  const navigate = useNavigate();
  const [data, setData] = useState(initialData);
  const [authSession, setAuthSession] = useState(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [loginForm, setLoginForm] = useState(buildInitialLoginForm());
  const [loadStatus, setLoadStatus] = useState("正在加载");
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState({ type: "ok", text: "" });
  const [actionBusy, setActionBusy] = useState({});
  const [modelImportForm, setModelImportForm] = useState(buildInitialModelImportForm(null));
  const [modelOptimizationForm, setModelOptimizationForm] = useState(buildInitialModelOptimizationForm([], null));
  const [modelTransferItems, setModelTransferItems] = useState([]);
  const [deploymentForm, setDeploymentForm] = useState(buildInitialDeploymentForm([], {}));
  const [benchmarkForm, setBenchmarkForm] = useState(buildInitialBenchmarkForm([], []));
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
      const fallback = (modelId, profileName, useGpu, providerPreference) => ({
        ...current,
        modelId,
        profileName,
        useGpu,
        providerPreference,
      });
      if (current.modelId && data.models.some((item) => item.model_id === current.modelId)) {
        const profiles = Object.keys(data.modelDetails[current.modelId]?.resolved_profiles || {});
        const optimization = getOptimizationSummaryForModel(current.modelId, data.modelDetails);
        const allowGpu = hasGpuDeploymentBackend(data.runtimeCapabilities, optimization.optimizationType);
        const nextUseGpu = current.useGpu && allowGpu;
        const nextProviderPreference = resolveDeploymentProviderPreference(
          current.providerPreference,
          data.runtimeCapabilities,
          nextUseGpu,
          optimization.optimizationType,
        );
        if (profiles.includes(current.profileName)) {
          if (nextUseGpu === current.useGpu && nextProviderPreference === current.providerPreference) {
            return current;
          }
          return {
            ...current,
            useGpu: nextUseGpu,
            providerPreference: nextProviderPreference,
            gpuIdsText: nextUseGpu ? current.gpuIdsText : "",
          };
        }
        return fallback(
          current.modelId,
          profiles.includes("default") ? "default" : profiles[0] || "",
          nextUseGpu,
          nextProviderPreference,
        );
      }
      const initial = buildInitialDeploymentForm(data.models, data.modelDetails);
      const initialOptimization = getOptimizationSummaryForModel(initial.modelId, data.modelDetails);
      const initialUseGpu = hasGpuDeploymentBackend(data.runtimeCapabilities, initialOptimization.optimizationType);
      return {
        ...initial,
        useGpu: initialUseGpu,
        providerPreference: resolveDeploymentProviderPreference(
          initial.providerPreference,
          data.runtimeCapabilities,
          initialUseGpu,
          initialOptimization.optimizationType,
        ),
      };
    });
  }, [data.models, data.modelDetails, data.runtimeCapabilities]);

  useEffect(() => {
    if (!deploymentForm.modelId) {
      return;
    }
    const optimization = getOptimizationSummaryForModel(deploymentForm.modelId, data.modelDetails);
    const shouldUseGpu = hasGpuDeploymentBackend(data.runtimeCapabilities, optimization.optimizationType) && deploymentForm.useGpu;
    const nextProviderPreference = resolveDeploymentProviderPreference(
      deploymentForm.providerPreference,
      data.runtimeCapabilities,
      shouldUseGpu,
      optimization.optimizationType,
    );
    if (
      shouldUseGpu !== deploymentForm.useGpu
      || nextProviderPreference !== deploymentForm.providerPreference
      || (!shouldUseGpu && deploymentForm.gpuIdsText)
    ) {
      setDeploymentForm((current) => (
        current.modelId === deploymentForm.modelId
          ? {
            ...current,
            useGpu: shouldUseGpu,
            providerPreference: nextProviderPreference,
            gpuIdsText: shouldUseGpu ? current.gpuIdsText : "",
          }
          : current
      ));
    }
  }, [data.modelDetails, data.runtimeCapabilities, deploymentForm.modelId, deploymentForm.providerPreference, deploymentForm.useGpu, deploymentForm.gpuIdsText]);

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
    if (!data.models.length || !data.optimizationOptions?.options?.length) {
      return;
    }
    setModelOptimizationForm((current) => {
      const safeModelId =
        current.modelId && data.models.some((item) => item.model_id === current.modelId)
          ? current.modelId
          : (data.models.find((item) => item.predict_ready) || data.models[0])?.model_id || "";
      const safeOptimizationType =
        current.optimizationType &&
        data.optimizationOptions.options.some((item) => item.code === current.optimizationType)
          ? current.optimizationType
          : (data.optimizationOptions.options[0]?.code || "ort_graph");
      return {
        ...current,
        modelId: safeModelId,
        optimizationType: safeOptimizationType,
      };
    });
  }, [data.models, data.optimizationOptions]);

  useEffect(() => {
    setBenchmarkForm((current) => {
      if (current.deploymentId && data.deployments.some((item) => item.deployment_id === current.deploymentId)) {
        return current;
      }
      return buildInitialBenchmarkForm(data.deployments, data.models);
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
      await logoutSession().catch(() => null);
      setAuthSession({ authenticated: false, current_user: null, current_permissions: [] });
      setLoadStatus("等待登录");
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
      navigate("/", { replace: true });
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

    const requestedModelId = payloadOverride?.model_id || deploymentForm.modelId;
    const optimizationType = getOptimizationSummaryForModel(requestedModelId, data.modelDetails).optimizationType;
    const useGpu = isCpuPinnedOptimization(optimizationType)
      ? false
      : (payloadOverride?.use_gpu ?? deploymentForm.useGpu);
    if (!useGpu) {
      gpuIds = [];
    }

    const resolvedUseGpu = useGpu;
    const payload = {
      ...payloadOverride,
      model_id: requestedModelId,
      profile_name: resolveProfileName(
        requestedModelId,
        data.modelDetails,
        payloadOverride?.profile_name || deploymentForm.profileName,
      ),
      launch_runtime: payloadOverride?.launch_runtime ?? deploymentForm.launchRuntime,
      use_gpu: resolvedUseGpu,
      gpu_ids: resolvedUseGpu ? gpuIds : [],
      provider_preference: resolveDeploymentProviderPreference(
        payloadOverride?.provider_preference || deploymentForm.providerPreference,
        data.runtimeCapabilities,
        resolvedUseGpu,
        optimizationType,
      ),
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

  async function handleOptimizeModel() {
    if (!modelOptimizationForm.modelId) {
      setNotice({ type: "error", text: "请先选择一个待优化模型。" });
      return;
    }
    if (!modelOptimizationForm.optimizationType) {
      setNotice({ type: "error", text: "请先选择一个优化方式。" });
      return;
    }
    const selectedModel = data.models.find((item) => item.model_id === modelOptimizationForm.modelId);
    const task = buildModelTransferTask({
      taskId: `optimize-${modelOptimizationForm.modelId}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      direction: "optimize",
      phase: "running",
      percent: null,
      loadedBytes: 0,
      totalBytes: null,
      fileName: selectedModel?.model_name || modelOptimizationForm.modelId,
      modelId: modelOptimizationForm.modelId,
      detail: "平台正在生成优化制品，请稍候。",
    });
    addModelTransferTask(task);
    setActionBusy((current) => ({ ...current, "optimize-model": true }));
    setNotice({ type: "ok", text: "" });
    try {
      const result = await optimizeModelPackage(modelOptimizationForm.modelId, {
        optimization_type: modelOptimizationForm.optimizationType,
        alias: modelOptimizationForm.alias.trim() || undefined,
      });
      await refreshAll({ silent: true });
      setModelOptimizationForm((current) => ({
        ...current,
        alias: "",
      }));
      updateModelTransferTask(task.taskId, {
        direction: "optimize",
        phase: "completed",
        percent: 100,
        modelId: result.model_id,
        detail: `优化制品已生成：${result.model_id}`,
      });
      scheduleModelTransferCleanup(task.taskId);
      setNotice({
        type: "ok",
        text: `优化制品已生成：${result.model_id}（${result.optimization_label}）。现在它已经作为正式模型接入平台，可继续部署和基准测试。`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updateModelTransferTask(task.taskId, {
        direction: "optimize",
        phase: "failed",
        detail: message,
      });
      scheduleModelTransferCleanup(task.taskId);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, "optimize-model": false }));
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
          scene: benchmarkForm.scene || undefined,
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
    if (!window.confirm(`确认删除结果反馈记录 ${archiveId} 吗？`)) {
      return;
    }
    await runAction(
      `delete-prediction-${archiveId}`,
      () => deletePredictionArchive(archiveId),
      `结果反馈记录已删除：${archiveId}`,
    );
    setPredictResult((current) => (current?.archive_id === archiveId ? null : current));
  }

  async function handleUpdatePredictionArchiveFeedback(archiveId, feedbackStatus, feedbackNote = null) {
    const nextStatus = normalizePredictionFeedbackStatus(feedbackStatus);
    const statusLabel = formatPredictionFeedbackLabel(nextStatus);
    const nextFeedbackNote = nextStatus === "unsatisfied" ? serializePredictionFeedbackReasons(feedbackNote) || null : null;
    const detailLabel = nextFeedbackNote ? `（${formatPredictionFeedbackReasonSummary(nextFeedbackNote)}）` : "";
    const result = await runAction(
      `prediction-feedback-${archiveId}`,
      () =>
        updatePredictionArchiveFeedback(archiveId, {
          feedback_status: nextStatus,
          feedback_note: nextFeedbackNote,
        }),
      `结果反馈已更新为“${statusLabel}${detailLabel}”。`,
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
      const normalizedModelFamily = String(taskRoutingForm.modelFamily || "").trim().toLowerCase();
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
        max_models: String(taskRoutingForm.maxModels || "").trim()
          ? Math.max(1, Math.min(5, Number(taskRoutingForm.maxModels) || 3))
          : undefined,
        model_family: normalizedModelFamily || undefined,
        options,
      });
      setTaskRoutingResult(result);
      await refreshAll({ silent: true });
      setNotice({ type: "ok", text: `任务调度策略已执行并归档：${result.archive_id || result.decision_id || ""}` });
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

  async function handleApplyGpuPoolUpdate(allowedGpuIds) {
    setActionBusy((current) => ({ ...current, "gpu-pool-update": true }));
    setNotice({ type: "ok", text: "" });
    try {
      const [result, policyResult, automationResult] = await Promise.all([
        updateDeploymentPolicyGpuPool({
          allowed_gpu_ids: allowedGpuIds,
          run_policy_optimization: true,
        }),
        getDeploymentPolicy().catch(() => null),
        getDeploymentPolicyAutomation().catch(() => null),
      ]);
      if (policyResult) {
        setDeploymentPolicy(policyResult);
      }
      if (automationResult) {
        setDeploymentPolicyAutomation(automationResult);
      }
      await refreshAll({ silent: true });
      setNotice({
        type: "ok",
        text:
          result.detail ||
          `GPU 使用范围已更新，当前可用 GPU：${(result.state?.effective_gpu_ids || []).join(",") || "仅 CPU"}`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setNotice({ type: "error", text: message });
    } finally {
      setActionBusy((current) => ({ ...current, "gpu-pool-update": false }));
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
    if (!window.confirm(`确认删除模型制品 ${modelId} 吗？关联部署、基准和结果反馈记录会一并删除。`)) {
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
        text: `模型制品已删除：${modelId}，同步清理部署 ${result.deleted_deployment_count} 条、基准 ${result.deleted_benchmark_count} 条、结果反馈记录 ${result.deleted_prediction_archive_count} 条。`,
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
      model_score_summaries: [],
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
      onApplyGpuPoolUpdate={handleApplyGpuPoolUpdate}
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
      modelOptimizationForm={modelOptimizationForm}
      modelTransferItems={modelTransferItems}
      onDismissModelTransfer={removeModelTransferTask}
      onModelImportFormChange={setModelImportForm}
      onModelOptimizationFormChange={setModelOptimizationForm}
      onOptimizeModel={handleOptimizeModel}
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
  return <Navigate replace to="/login" />;
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
  const deploymentPolicyAutoRefreshPathRef = useRef("");
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
    onApplyGpuPoolUpdate,
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
    modelOptimizationForm,
    modelTransferItems,
    onDismissModelTransfer,
    onModelImportFormChange,
    onModelOptimizationFormChange,
    onOptimizeModel,
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

  useEffect(() => {
    resetPageScrollToTop();
  }, [location.pathname]);

  useEffect(() => {
    if (location.pathname !== "/deployment-policy") {
      deploymentPolicyAutoRefreshPathRef.current = "";
      return;
    }
    if (deploymentPolicyAutoRefreshPathRef.current === location.pathname) {
      return;
    }
    deploymentPolicyAutoRefreshPathRef.current = location.pathname;
    void onLoadDeploymentPolicy();
  }, [location.pathname, onLoadDeploymentPolicy]);

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
                onUpdatePredictionArchiveFeedback={onUpdatePredictionArchiveFeedback}
                onTaskRoutingFormChange={onTaskRoutingFormChange}
                taskRoutingForm={taskRoutingForm}
                taskRoutingResult={taskRoutingResult}
              />,
            )}
            {renderGuardedRoute(
              "/task-routing-history",
              <TaskRoutingHistoryPage
                derived={derived}
              />,
            )}
            {renderGuardedRoute(
              "/deployment-performance",
              <DeploymentPerformancePage
                derived={derived}
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
                modelOptimizationForm={modelOptimizationForm}
                modelTransferItems={modelTransferItems}
                onDismissModelTransfer={onDismissModelTransfer}
                onDeleteModelPackage={onDeleteModelPackage}
                onExportModelPackage={onExportModelPackage}
                onImportModel={onImportModel}
                onModelImportFormChange={onModelImportFormChange}
                onModelOptimizationFormChange={onModelOptimizationFormChange}
                onOptimizeModel={onOptimizeModel}
              />,
            )}
            {renderGuardedRoute(
              "/artifacts",
              <ArtifactsPage
                currentUser={currentUser}
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
                data={data}
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
            {renderGuardedRoute(
              "/resources",
              <ResourcesPage
                actionBusy={actionBusy}
                derived={derived}
                onApplyGpuPoolUpdate={onApplyGpuPoolUpdate}
              />,
            )}
            {renderGuardedRoute("/logs", <LogsPage derived={derived} selectedLogs={selectedLogs} />)}
            <Route element={<Navigate replace to="/login" />} path="*" />
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

function RelationGraphCard({
  emptyNote = "先完成一次推理，平台会根据三元组自动绘制关系图谱。",
  emptyTitle = "还没有可展示的图谱",
  kicker = "关系图谱",
  note = "根据本次结果自动生成小图谱，支持滚轮缩放、拖拽平移与节点数量调节。",
  title = "图谱视图",
  triples,
}) {
  const [nodeLimit, setNodeLimit] = useState(20);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragStart, setDragStart] = useState(null);

  const graphData = useMemo(() => {
    const sourceTriples = Array.isArray(triples) ? triples : [];
    if (!sourceTriples.length) {
      return { nodes: [], edges: [], totalNodeCount: 0 };
    }

    const nodes = [];
    const nodeIndex = new Map();
    const edges = [];

    for (const triple of sourceTriples) {
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
  }, [nodeLimit, triples]);

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
        kicker={kicker}
        title={title}
        note={note}
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
        <EmptyState icon="G" title={emptyTitle} note={emptyNote} />
      )}
    </article>
  );
}

function PredictionGraphCard({ predictResult }) {
  return (
    <RelationGraphCard
      emptyNote="先完成一次在线推理，平台会根据三元组自动绘制关系图谱。"
      emptyTitle="还没有可展示的图谱"
      kicker="关系图谱"
      note="根据本次预测结果自动生成小图谱，支持滚轮缩放、拖拽平移与节点数量调节。"
      title="图谱视图"
      triples={predictResult?.triples || []}
    />
  );
}

function PredictionArchiveFeedbackControls({ actionBusy, archive, canFeedback = true, onUpdate }) {
  const currentStatus = normalizePredictionFeedbackStatus(archive?.feedback_status);
  const currentReasons = parsePredictionFeedbackReasons(archive?.feedback_note);
  const currentReasonKey = serializePredictionFeedbackReasons(currentReasons);
  const [draftStatus, setDraftStatus] = useState(currentStatus);
  const [draftReasons, setDraftReasons] = useState(currentReasons);
  const options = [
    { value: "satisfied", label: "满意" },
    { value: "unsatisfied", label: "不满意" },
    { value: "pending", label: "暂定" },
  ];
  const unsatisfiedReasonOptions = [
    { value: "missing_too_many", label: "漏抽多" },
    { value: "false_positive_too_many", label: "误抽多" },
    { value: "latency_high", label: "延迟高" },
  ];
  const busy = actionBusy?.[`prediction-feedback-${archive.archive_id}`];
  const draftReasonKey = serializePredictionFeedbackReasons(draftReasons);
  const dirty =
    draftStatus !== currentStatus ||
    (draftStatus === "unsatisfied" && draftReasonKey !== currentReasonKey) ||
    (currentStatus === "unsatisfied" && draftStatus !== "unsatisfied");

  useEffect(() => {
    setDraftStatus(currentStatus);
    setDraftReasons(currentReasons);
  }, [archive?.archive_id, currentStatus, currentReasonKey]);

  if (!canFeedback) {
    return <div className="table-note">当前账号仅可查看反馈结果，不能修改反馈。</div>;
  }

  return (
    <div className="archive-feedback-panel">
      <div className="actions archive-feedback-actions">
        {options.map((item) => (
          <button
            key={item.value}
            className={draftStatus === item.value ? "button-active" : ""}
            disabled={busy}
            onClick={() => {
              setDraftStatus(item.value);
              if (item.value !== "unsatisfied") {
                setDraftReasons([]);
              }
            }}
            type="button"
          >
            {item.label}
          </button>
        ))}
      </div>
      {draftStatus === "unsatisfied" ? (
        <div className="archive-feedback-reason-shell">
          <div className="table-note">不满意原因（可多选，可留空）：</div>
          <div className="actions archive-feedback-reason-actions">
            {unsatisfiedReasonOptions.map((item) => (
              <button
                key={item.value}
                className={draftReasons.includes(item.value) ? "button-active" : ""}
                disabled={busy}
                onClick={() => {
                  setDraftReasons((current) =>
                    current.includes(item.value)
                      ? current.filter((reason) => reason !== item.value)
                      : [...current, item.value],
                  );
                }}
                type="button"
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      ) : null}
      <div className="actions archive-feedback-save-row">
        <button
          className="btn-primary"
          disabled={busy || !dirty}
          onClick={() => void onUpdate(archive.archive_id, draftStatus, draftStatus === "unsatisfied" ? draftReasons : null)}
          type="button"
        >
          保存反馈
        </button>
        <button
          disabled={busy || !dirty}
          onClick={() => {
            setDraftStatus(currentStatus);
            setDraftReasons(currentReasons);
          }}
          type="button"
        >
          恢复当前值
        </button>
      </div>
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
  const predictArchivePagination = useTablePagination(sortByCreatedDesc(derived.predictArchives), 5);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageArchives = derived.currentAccessUser?.is_admin;
  const canExportArchives = currentPermissions.includes("archive.export");
  const canFeedbackArchives = currentPermissions.includes("predictions.feedback");
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
                  value={formatPredictionFeedbackSummary(currentArchive?.feedback_status, currentArchive?.feedback_note)}
                  note={currentArchive?.feedback_updated_at || "默认暂定，可在下方结果反馈区中修改"}
                />
              </div>
              {currentArchive ? (
                <div className="banner banner-muted">
                  当前结果反馈：{formatPredictionFeedbackSummary(currentArchive.feedback_status, currentArchive.feedback_note)}。如果这次结果已经可直接用于业务，可标记“满意”；如果结果明显不符合预期，可先标记“不满意”，再补充“漏抽多”、“误抽多”和/或“延迟高”；暂时不判断就保留“暂定”。
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
          kicker="结果反馈"
          title="单次预测反馈"
          note="每次在线推理成功后，平台会自动写入反馈记录，方便后续导出 JSON、截图、回填论文案例，并补充满意度反馈。"
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
                {predictArchivePagination.pagedItems.map((item) => (
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
                        <div>{formatPredictionFeedbackSummary(item.feedback_status, item.feedback_note)}</div>
                        <div className="table-note">{item.feedback_updated_at || "未补充反馈"}</div>
                        {item.feedback_note ? <div className="table-note">原因：{formatPredictionFeedbackReasonSummary(item.feedback_note)}</div> : null}
                        <PredictionArchiveFeedbackControls
                          actionBusy={actionBusy}
                          archive={item}
                          canFeedback={canFeedbackArchives}
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
            <PaginationBar
              currentPage={predictArchivePagination.page}
              totalPages={predictArchivePagination.totalPages}
              pageSize={predictArchivePagination.pageSize}
              totalItems={predictArchivePagination.totalItems}
              onChange={predictArchivePagination.setPage}
            />
          </div>
        ) : (
          <EmptyState icon="A" title="暂无结果反馈" note="完成一次在线推理后，结果会自动沉淀到这里。" />
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
  const userMatrixPagination = useTablePagination(users, 5);
  const governancePagination = useTablePagination(users, 5);

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
        <SectionHeader kicker="权限构成" title="角色与权限映射" note="每个角色都绑定了一组固定权限。这里不仅展示权限项，还会说明这个角色在平台里实际能进入哪些页面、能做哪些操作。" />
        <div className="list-grid">
          {roles.map((role) => (
            <article className="list-card" key={role.role_id}>
              <h3>{role.role_name}</h3>
              <div className="table-note">{role.description}</div>
              <div className="detail-note">{getRoleScopeSummary(role.role_id)}</div>
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
              {userMatrixPagination.pagedItems.map((item) => {
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
          <PaginationBar
            currentPage={userMatrixPagination.page}
            totalPages={userMatrixPagination.totalPages}
            pageSize={userMatrixPagination.pageSize}
            totalItems={userMatrixPagination.totalItems}
            onChange={userMatrixPagination.setPage}
          />
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
  const governancePagination = useTablePagination(users, 5);

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
              {governancePagination.pagedItems.map((item) => (
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
          <PaginationBar
            currentPage={governancePagination.page}
            totalPages={governancePagination.totalPages}
            pageSize={governancePagination.pageSize}
            totalItems={governancePagination.totalItems}
            onChange={governancePagination.setPage}
          />
        </div>
      </section>
    </>
  );
}

function TaskRoutingPage({
  actionBusy,
  derived,
  onRunTaskRouting,
  onUpdatePredictionArchiveFeedback,
  onTaskRoutingFormChange,
  taskRoutingForm,
  taskRoutingResult,
}) {
  const currentPermissions = derived.currentAccessPermissions || [];
  const isAdminTaskRoutingView = Boolean(derived.currentAccessUser?.is_admin) || currentPermissions.includes("strategies.manage");
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
    model_score_summaries: [],
    weight_profiles: [],
    recent_decisions: [],
    insights: [],
  };
  const sceneOptions = [
    { value: "", label: "自动判定", note: "不传场景标识时，平台会根据文本语言与领域线索自动推断：英文走英文通用，中文工业文本走工业场景，其余中文文本走中文通用场景。" },
    { value: "en_general", label: "英文通用场景", note: "只在英文 deployment 中筛选，优先英文模型。" },
    { value: "zh_general", label: "中文通用场景", note: "只在中文通用 deployment 中筛选，优先使用 DuIE 一类通用中文模型。" },
    { value: "zh_industrial", label: "中文工业设备场景", note: "优先使用带 equipmentmix15k 标识的工业专用模型；若通用中文模型历史表现更强，也允许参与调度。" },
  ];
  const modelFamilyOptions = useMemo(() => {
    const families = Array.from(
      new Set((derived.models || []).map((item) => String(item.model_family || "").trim().toLowerCase()).filter(Boolean)),
    ).sort();
    return [
      { value: "", label: "全部种类", note: "不限定模型种类，平台按当前目标与场景在全部候选 deployment 中调度。" },
      ...families.map((family) => ({
        value: family,
        label: family,
        note: `只在 ${family} 家族下做任务调度。`,
      })),
    ];
  }, [derived.models]);
  const completedBenchmarks = sortByCreatedDesc(derived.benchmarks).filter((item) => item.status === "completed");
  const benchmarkedModelCount = new Set(completedBenchmarks.map((item) => item.model_id)).size;
  const calibrationArchiveCount = derived.predictArchives.length;
  const selectedObjective = candidateOptions.find((item) => item.value === taskRoutingForm.objective) || candidateOptions[0];
  const selectedScene = sceneOptions.find((item) => item.value === taskRoutingForm.scene) || sceneOptions[0];
  const selectedModelFamily = modelFamilyOptions.find((item) => item.value === taskRoutingForm.modelFamily) || modelFamilyOptions[0];
  const effectiveSceneValue = String(taskRoutingResult?.context_features?.scene || taskRoutingForm.scene || "").trim();
  const effectiveScene = sceneOptions.find((item) => item.value === effectiveSceneValue) || sceneOptions[0];
  const hasCurrentRun = Boolean(taskRoutingResult);
  const canFeedbackArchives = currentPermissions.includes("predictions.feedback");
  const selectedDeployments = taskRoutingResult?.selected_deployments || [];
  const fusedTriples = taskRoutingResult?.fused_result?.triples || [];
  const currentTaskRoutingArchive = useMemo(() => {
    if (!hasCurrentRun) {
      return null;
    }
    const targetArchiveId = String(taskRoutingResult?.archive_id || "").trim();
    const targetDecisionId = String(
      taskRoutingResult?.decision_id || taskRoutingResult?.task_routing_decision_id || "",
    ).trim();
    return (
      sortByCreatedDesc(derived.predictArchives || []).find((item) => {
        const archiveId = String(item?.archive_id || "").trim();
        const decisionId = String(item?.task_routing_decision_id || "").trim();
        return (targetArchiveId && archiveId === targetArchiveId) || (targetDecisionId && decisionId === targetDecisionId);
      }) || null
    );
  }, [derived.predictArchives, hasCurrentRun, taskRoutingResult]);
  const hasCurrentExplicitFeedback = Boolean(
    currentTaskRoutingArchive
    && normalizePredictionFeedbackStatus(currentTaskRoutingArchive.feedback_status) !== "pending",
  );
  const topWeightEntry = Object.entries(taskRoutingResult?.objective_weights || {}).sort((a, b) => b[1] - a[1])[0];
  const currentWeightProfile = useMemo(
    () => (feedbackOverview.weight_profiles || []).find((item) => item.objective === taskRoutingForm.objective) || null,
    [feedbackOverview.weight_profiles, taskRoutingForm.objective],
  );
  const objectiveWeightRows = useMemo(
    () =>
      candidateOptions.map((item) => ({
        ...item,
        profile: (feedbackOverview.weight_profiles || []).find((profile) => profile.objective === item.value) || null,
      })),
    [candidateOptions, feedbackOverview.weight_profiles],
  );
  const currentModelScoreSummaries = useMemo(
    () => {
      if (!hasCurrentRun) {
        return [];
      }
      return (feedbackOverview.model_score_summaries || [])
        .filter((item) => item.objective === taskRoutingForm.objective)
        .filter((item) => !effectiveSceneValue || item.scene === effectiveSceneValue);
    },
    [effectiveSceneValue, feedbackOverview.model_score_summaries, hasCurrentRun, taskRoutingForm.objective],
  );
  const sceneMatchedArchives = useMemo(
    () => effectiveSceneValue
      ? sortByCreatedDesc(derived.predictArchives).filter((item) => archiveMatchesTaskRoutingScene(item, effectiveSceneValue))
      : sortByCreatedDesc(derived.predictArchives),
    [derived.predictArchives, effectiveSceneValue],
  );
  const sceneFeedbackSummary = useMemo(() => {
    const summary = {
      archiveCount: sceneMatchedArchives.length,
      satisfiedCount: 0,
      unsatisfiedCount: 0,
      missingTooManyCount: 0,
      falsePositiveTooManyCount: 0,
      latencyHighCount: 0,
    };
    sceneMatchedArchives.forEach((item) => {
      const status = normalizePredictionFeedbackStatus(item.feedback_status);
      if (status === "satisfied") {
        summary.satisfiedCount += 1;
      }
      if (status === "unsatisfied") {
        summary.unsatisfiedCount += 1;
      }
      const reasons = parsePredictionFeedbackReasons(item.feedback_note);
      if (reasons.includes("missing_too_many")) {
        summary.missingTooManyCount += 1;
      }
      if (reasons.includes("false_positive_too_many")) {
        summary.falsePositiveTooManyCount += 1;
      }
      if (reasons.includes("latency_high")) {
        summary.latencyHighCount += 1;
      }
    });
    return summary;
  }, [sceneMatchedArchives]);
  const sceneCalibrationNarratives = useMemo(() => {
    const narratives = [];
    if (!sceneFeedbackSummary.archiveCount) {
      return ["当前场景还没有可直接引用的在线归档反馈，调度层暂以 benchmark 和基础权重为主。"];
    }
    if (effectiveScene.value === "zh_industrial" && (sceneFeedbackSummary.missingTooManyCount || sceneFeedbackSummary.falsePositiveTooManyCount)) {
      narratives.push(
        `中文工业设备场景近期出现 ${sceneFeedbackSummary.missingTooManyCount} 条“漏抽多”和 ${sceneFeedbackSummary.falsePositiveTooManyCount} 条“误抽多”反馈，系统会进一步提高领域权重，并优先考虑 equipmentmix15k 路线模型。`,
      );
    }
    if (sceneFeedbackSummary.missingTooManyCount) {
      narratives.push("“漏抽多”反馈会推动覆盖权重上调，让高召回模型或多模型融合路径更容易获得更高优先级。");
    }
    if (sceneFeedbackSummary.falsePositiveTooManyCount) {
      narratives.push("“误抽多”反馈会推动准确率与稳定性权重上调，系统会压低误报偏多候选的排序分数。");
    }
    if (sceneFeedbackSummary.latencyHighCount) {
      narratives.push("“延迟高”反馈会推动时延与资源权重上调，系统会更偏向低 P95 的 deployment 或更轻量的 profile。");
    }
    if (currentWeightProfile && Object.keys(currentWeightProfile.unsupervised_adjustments || {}).length) {
      narratives.push(
        `当前“${selectedObjective.label}”目标已吸收无用户反馈校准信号：${Object.entries(currentWeightProfile.unsupervised_adjustments || {}).map(([key, value]) => `${key}=${Number(value).toFixed(2)}`).join(" / ")}。`,
      );
    }
    if (currentWeightProfile && Object.keys(currentWeightProfile.feedback_adjustments || {}).length) {
      narratives.push(
        `当前“${selectedObjective.label}”目标已吸收用户反馈校准信号：${Object.entries(currentWeightProfile.feedback_adjustments || {}).map(([key, value]) => `${key}=${Number(value).toFixed(2)}`).join(" / ")}。`,
      );
    }
    if (!narratives.length) {
      narratives.push("当前场景已有在线归档样本，但还没有形成明显的反馈驱动偏置，调度层暂以运行结果与 benchmark 作为主要证据。");
    }
    return narratives;
  }, [currentWeightProfile, effectiveScene.value, sceneFeedbackSummary, selectedObjective.label]);
  const candidatePagination = useTablePagination(selectedDeployments, 5);
  const modelScoreSummaryPagination = useTablePagination(currentModelScoreSummaries, 5);
  const strategyModeLabels = {
    none: "未执行",
    "multi-objective-adaptive": "多目标自适应",
  };
  function formatStrategyModeLabel(value) {
    return strategyModeLabels[String(value || "").trim()] || value || "-";
  }
  const activeWeightSnapshot = taskRoutingResult?.objective_weights || {};
  const activeWeightEntries = Object.entries(activeWeightSnapshot).sort((left, right) => Number(right[1] || 0) - Number(left[1] || 0));
  const activeTopWeightEntry = activeWeightEntries[0];
  const activeBenchmarkAdjustments = currentWeightProfile?.benchmark_adjustments || {};
  const activeUnsupervisedAdjustments = currentWeightProfile?.unsupervised_adjustments || {};
  const activeFeedbackAdjustments = currentWeightProfile?.feedback_adjustments || {};
  const selectedDeploymentIds = useMemo(
    () => new Set((selectedDeployments || []).map((item) => String(item.deployment_id || "").trim()).filter(Boolean)),
    [selectedDeployments],
  );
  const selectedModelIds = useMemo(
    () => new Set((selectedDeployments || []).map((item) => String(item.model_id || "").trim()).filter(Boolean)),
    [selectedDeployments],
  );
  const currentRunModelScoreRows = useMemo(() => {
    if (!hasCurrentRun) {
      return [];
    }
    const rows = (currentModelScoreSummaries || []).filter((item) => {
      const deploymentId = String(item.deployment_id || "").trim();
      const modelId = String(item.model_id || "").trim();
      return selectedDeploymentIds.has(deploymentId) || selectedModelIds.has(modelId);
    });
    return rows.sort((left, right) => {
      const leftSelected = selectedDeploymentIds.has(String(left.deployment_id || "").trim()) ? 1 : 0;
      const rightSelected = selectedDeploymentIds.has(String(right.deployment_id || "").trim()) ? 1 : 0;
      if (rightSelected !== leftSelected) {
        return rightSelected - leftSelected;
      }
      const rightDelta = Math.abs(Number(right.score_delta || 0));
      const leftDelta = Math.abs(Number(left.score_delta || 0));
      if (rightDelta !== leftDelta) {
        return rightDelta - leftDelta;
      }
      return String(left.deployment_id || "").localeCompare(String(right.deployment_id || ""));
    });
  }, [currentModelScoreSummaries, hasCurrentRun, selectedDeploymentIds, selectedModelIds]);
  const currentRunModelScoreDisplayRows = useMemo(
    () => currentRunModelScoreRows.map((item) => (
      hasCurrentExplicitFeedback
        ? item
        : {
          ...item,
          feedback_score_delta: 0,
          feedback_score_breakdown: {},
          change_note: item.change_note || "当前展示的是本次运行命中的 deployment；你还没有提交本次结果反馈，因此“有反馈”列暂不计入本次影响。",
        }
    )),
    [currentRunModelScoreRows, hasCurrentExplicitFeedback],
  );
  const currentRunModelScorePagination = useTablePagination(currentRunModelScoreDisplayRows, 5);
  const currentOperationFeedbackRows = useMemo(() => {
    if (!hasCurrentRun) {
      return [];
    }
    const rows = [];
    rows.push({
      operation: "运行任务调度",
      triggerTime: taskRoutingResult?.archive_id ? currentTaskRoutingArchive?.created_at || "-" : "-",
      status: "已执行",
      focus: "本次运行权重快照",
      unsupervisedSnapshot: taskRoutingResult?.adaptive_unsupervised_adjustments || {},
      feedbackSnapshot: {},
      note:
        ((taskRoutingResult?.adaptation_reasons || []).slice(0, 2).join("；"))
        || "这一步反映的是本次调度运行时根据文本、场景和历史证据即时生成的权重修正。当前尚未提交本次结果反馈，因此这里不展示“有用户反馈修正”。",
    });
    if (currentTaskRoutingArchive && normalizePredictionFeedbackStatus(currentTaskRoutingArchive.feedback_status) !== "pending") {
      rows.push({
        operation: "提交结果反馈",
        triggerTime: currentTaskRoutingArchive.feedback_updated_at || "-",
        status: formatPredictionFeedbackSummary(currentTaskRoutingArchive.feedback_status, currentTaskRoutingArchive.feedback_note),
        focus: "反馈已写入当前累计学习状态",
        unsupervisedSnapshot: currentWeightProfile?.unsupervised_adjustments || {},
        feedbackSnapshot: currentWeightProfile?.feedback_adjustments || {},
        note:
          normalizePredictionFeedbackStatus(currentTaskRoutingArchive.feedback_status) === "satisfied"
            ? "这一步显示的是当前累计反馈修正状态。满意反馈会让系统继续维持或增强当前更合适的权重方向。"
            : `这一步显示的是当前累计反馈修正状态。${formatPredictionFeedbackReasonSummary(currentTaskRoutingArchive.feedback_note) || "当前反馈已进入学习过程。"}会持续作用到后续调度。`,
      });
    }
    return rows;
  }, [currentTaskRoutingArchive, currentWeightProfile?.feedback_adjustments, currentWeightProfile?.unsupervised_adjustments, hasCurrentRun, taskRoutingResult]);
  const currentOperationFeedbackPagination = useTablePagination(currentOperationFeedbackRows, 5);

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="创新点一"
          title="任务级模型调度策略"
          note="这一页只保留任务调度真正需要的内容：填写参数、执行调度、查看结果、提交反馈，以及观察这次操作带来的即时学习反馈。"
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
        <div className="banner banner-muted">使用顺序很简单：先填参数，再运行任务调度，然后依次看调度结果、候选部署、结果反馈，以及这次操作带来的即时学习反馈。</div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="操作区"
          title="运行参数"
          note="这是任务调度唯一需要你主动填写的区域。外部 API 风格调用时，场景标识和最大模型数都可以不传；平台会自动补默认策略。"
        />
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
              max="5"
              min="1"
              onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, maxModels: event.target.value }))}
              type="number"
              value={taskRoutingForm.maxModels}
            />
            <small className="detail-note">可留空。留空时后端默认按 3 处理；它是“上限”，不是一定会选满，低时延场景通常仍只会选 1 个。</small>
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
            <span>模型种类</span>
            <select
              onChange={(event) => onTaskRoutingFormChange((current) => ({ ...current, modelFamily: event.target.value }))}
              value={taskRoutingForm.modelFamily}
            >
              {modelFamilyOptions.map((item) => (
                <option key={`task-routing-family-${item.value || "all"}`} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <small className="detail-note">{selectedModelFamily.note}</small>
          </label>
          <label className="stack-field">
            <span>模型范围</span>
            <div className="stack-field-readonly">
              {selectedScene.value === "en_general"
                ? `${selectedModelFamily.value ? `${selectedModelFamily.label} / ` : ""}英文 deployment`
                : selectedScene.value === "zh_general"
                  ? `${selectedModelFamily.value ? `${selectedModelFamily.label} / ` : ""}中文通用 deployment（DuIE 路线）`
                  : selectedScene.value === "zh_industrial"
                    ? `${selectedModelFamily.value ? `${selectedModelFamily.label} / ` : ""}中文工业 deployment（IndustrialEquipmentMix 路线）`
                    : `${selectedModelFamily.value ? `${selectedModelFamily.label} / ` : ""}自动按文本语言与领域判定`}
            </div>
            <small className="detail-note">这里会随场景和模型种类自动收紧范围；如果两者都不传，就按文本语言、领域和目标偏好自动筛候选。</small>
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
          <div className="actions inline-actions stack-field-wide">
            <button
              className="btn-primary"
              disabled={actionBusy["task-routing"]}
              onClick={() => void onRunTaskRouting()}
              type="button"
            >
              {actionBusy["task-routing"] ? "执行中" : "运行任务调度"}
            </button>
          </div>
        </div>
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
                <MetricCard title="归档编号" value={currentTaskRoutingArchive?.archive_id || taskRoutingResult.archive_id || "-"} note="本次调度结果已自动写入任务调度归档" />
                <MetricCard
                  title="当前反馈"
                  value={formatPredictionFeedbackSummary(currentTaskRoutingArchive?.feedback_status, currentTaskRoutingArchive?.feedback_note)}
                  note={currentTaskRoutingArchive?.feedback_updated_at || "可在下方“结果反馈”区直接补充"}
                />
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

        <RelationGraphCard
          emptyNote="运行一次任务调度后，平台会根据融合后的最终三元组自动绘制图谱。"
          emptyTitle="还没有可展示的调度图谱"
          kicker="图谱区"
          note="根据本次任务调度融合结果自动生成小图谱，支持滚轮缩放、拖拽平移与节点数量调节。"
          title="融合结果图谱"
          triples={fusedTriples}
        />
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="结果反馈"
          title="本次调度反馈"
          note="任务调度执行成功后，可以直接在这里给出满意度反馈。该反馈会进入在线学习流程，参与后续调度权重与模型评分修正。"
        />
        {currentTaskRoutingArchive ? (
          <>
            <div className="metrics">
              <MetricCard title="归档编号" value={currentTaskRoutingArchive.archive_id} note={currentTaskRoutingArchive.created_at || "-"} />
              <MetricCard title="决策编号" value={currentTaskRoutingArchive.task_routing_decision_id || taskRoutingResult?.decision_id || "-"} note="任务调度留痕编号" />
              <MetricCard title="当前反馈" value={formatPredictionFeedbackSummary(currentTaskRoutingArchive.feedback_status, currentTaskRoutingArchive.feedback_note)} note={currentTaskRoutingArchive.feedback_updated_at || "尚未补充"} />
              <MetricCard title="不满意原因" value={currentTaskRoutingArchive.feedback_note ? formatPredictionFeedbackReasonSummary(currentTaskRoutingArchive.feedback_note) : "无"} note="支持多选，可留空" />
            </div>
            <div className="banner banner-muted">
              如果这次任务调度的融合结果已经可直接用于业务，可标记“满意”；如果你觉得输出有明显问题，可标记“不满意”，再选择“漏抽多”“误抽多”和/或“延迟高”。这份反馈会直接作用到后续任务调度学习过程。
            </div>
            <PredictionArchiveFeedbackControls
              actionBusy={actionBusy}
              archive={currentTaskRoutingArchive}
              canFeedback={canFeedbackArchives}
              onUpdate={onUpdatePredictionArchiveFeedback}
            />
          </>
        ) : (
          <EmptyState
            icon="F"
            title="还没有可反馈的调度结果"
            note="先完成一次任务调度，平台生成归档后，这里才会开放直接反馈。"
          />
        )}
      </section>

      {isAdminTaskRoutingView ? (
        <>
          <section className="page-shell card section-shell">
            <SectionHeader kicker="解释区" title="候选部署与执行明细" note="这一表专门回答“为什么选中这些 deployment”。这里的评分对象是 deployment 实例，不是抽象模型名；同一模型的不同 deployment 或不同 profile 会分别计算评分。" />
            {taskRoutingResult ? (
              <>
                <div className="banner banner-muted">
                  这张表看什么：看平台为什么选中这些 deployment，没有选中谁。重点看 `score`、`得分拆解`、`领域匹配` 和最后一列 `说明`。这里每一行都是一个 deployment 实例。
                </div>
                <div className="banner banner-muted">
                  术语提示：`得分拆解` 代表各 deployment 在准确、覆盖、时延等维度的评分贡献；`领域匹配` 表示它与当前文本领域的贴合度；`资源级别` 表示其运行成本或资源档位。若某个 deployment 缺少自身 benchmark 或归档证据，系统才会有限借用同模型的同类证据做补充。
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
                      {candidatePagination.pagedItems.map((item) => (
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
                  <PaginationBar
                    currentPage={candidatePagination.page}
                    totalPages={candidatePagination.totalPages}
                    pageSize={candidatePagination.pageSize}
                    totalItems={candidatePagination.totalItems}
                    onChange={candidatePagination.setPage}
                  />
                </div>
                {(taskRoutingResult.decision_trace || []).length ? <div className="logs">{(taskRoutingResult.decision_trace || []).join("\n")}</div> : null}
              </>
            ) : (
              <div className="logs">运行后这里会给出候选排序、实际选中的 deployment，以及各模型的执行结果摘要。</div>
            )}
          </section>

          <section className="page-shell card section-shell">
            <SectionHeader kicker="即时反馈" title="本次学习反馈" note="这一块只回答一个问题：你刚刚在任务调度页做的“运行任务调度”和“提交结果反馈”，到底给权重和模型表现带来了什么变化。" />
            {hasCurrentRun ? (
              <>
                <div className="score-evolution-block">
                  <div className="table-subhead">表 1：本次操作带来的学习反馈</div>
                  <div className="banner banner-muted">这张表看什么：一行对应一次任务调度页内的关键操作。`运行任务调度` 只看本次运行时即时生成的无用户反馈修正；`提交结果反馈` 才看你这次满意度与原因如何进入当前累计学习状态。</div>
                  <div className="table-wrap">
                    <table className="weight-profile-table">
                      <thead>
                        <tr>
                          <th>操作</th>
                          <th>时间 / 状态</th>
                          <th>影响焦点</th>
                          <th>无用户反馈修正</th>
                          <th>有用户反馈修正</th>
                          <th>说明</th>
                        </tr>
                      </thead>
                      <tbody>
                        {currentOperationFeedbackPagination.pagedItems.length ? currentOperationFeedbackPagination.pagedItems.map((item) => (
                          <tr key={`${item.operation}-${item.triggerTime}-${item.status}`}>
                            <td>
                              <div className="archive-cell-stack">
                                <div className="archive-main">{item.operation}</div>
                                <div className="table-note">{selectedObjective.label}</div>
                                <div className="table-note">{selectedScene.label}</div>
                              </div>
                            </td>
                            <td>
                              <div className="archive-cell-stack">
                                <div>{item.triggerTime}</div>
                                <div className="table-note">{item.status}</div>
                              </div>
                            </td>
                            <td>{item.focus}</td>
                            <td>
                              {Object.keys(item.unsupervisedSnapshot || {}).length
                                ? renderWeightSnapshotGrid(item.unsupervisedSnapshot || {}, `operation-unsupervised-${item.operation}`, {
                                  variantClass: "weight-snapshot-chip-unsupervised",
                                  signed: true,
                                })
                                : <span className="table-note">当前暂无该类修正</span>}
                            </td>
                            <td>
                              {Object.keys(item.feedbackSnapshot || {}).length
                                ? renderWeightSnapshotGrid(item.feedbackSnapshot || {}, `operation-feedback-${item.operation}`, {
                                  variantClass: "weight-snapshot-chip-feedback",
                                  signed: true,
                                })
                                : <span className="table-note">当前暂无该类修正</span>}
                            </td>
                            <td className="limitation-cell">{item.note}</td>
                          </tr>
                        )) : (
                          <tr>
                            <td className="table-note" colSpan="6">当前还没有可展示的本次学习反馈记录。</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                    <PaginationBar
                      currentPage={currentOperationFeedbackPagination.page}
                      totalPages={currentOperationFeedbackPagination.totalPages}
                      pageSize={currentOperationFeedbackPagination.pageSize}
                      totalItems={currentOperationFeedbackPagination.totalItems}
                      onChange={currentOperationFeedbackPagination.setPage}
                    />
                  </div>
                </div>

                <div className="score-evolution-block">
                  <div className="table-subhead">表 2：本次影响的 Deployment 表现反馈</div>
                  <div className="banner banner-muted">这张表看什么：只看本次任务调度实际选中或命中的 deployment。若你还没有提交本次结果反馈，表里的“有反馈”列会保持为空；只有你真正提交本次反馈后，这一列才展示本次反馈带来的影响。</div>
                  <div className="table-wrap">
                    <table className="weight-profile-table score-evolution-table score-evolution-primary-table">
                      <thead>
                        <tr>
                          <th>Deployment / 模型</th>
                          <th>本次关系</th>
                          <th>评分概览</th>
                          <th>表现层反馈</th>
                          <th>权重层反馈</th>
                          <th>说明</th>
                        </tr>
                      </thead>
                      <tbody>
                        {currentRunModelScorePagination.pagedItems.length ? currentRunModelScorePagination.pagedItems.map((item) => {
                          const deltaPositive = Number(item.score_delta || 0) > 0;
                          const deltaNegative = Number(item.score_delta || 0) < 0;
                          const selectedNow = selectedDeploymentIds.has(String(item.deployment_id || "").trim());
                          return (
                            <tr key={`current-run-score-${item.objective}-${item.scene}-${item.deployment_id}`}>
                              <td className="score-evolution-cell-identity">
                                <div className="archive-cell-stack">
                                  <strong className="archive-main">{item.deployment_id}</strong>
                                  <div className="table-note">{item.profile_name}</div>
                                  <div className="archive-main">{item.model_id}</div>
                                  <div className="table-note">{item.dataset_name || "-"}</div>
                                </div>
                              </td>
                              <td>
                                <div className="archive-cell-stack">
                                  <div className="archive-main">{selectedNow ? "本次已选中" : "同模型关联"}</div>
                                  <div className="table-note">{formatObjectiveLabel(item.objective)}</div>
                                  <div className="table-note">{item.scene || "-"}</div>
                                </div>
                              </td>
                              <td>
                                <div className="score-overview-stack">
                                  <div className="score-overview-row">
                                    <span className="weight-snapshot-chip weight-snapshot-chip-baseline score-overview-chip">
                                      <strong>优化前</strong>
                                      <span>{Number(item.baseline_score || 0).toFixed(4)}</span>
                                    </span>
                                    <span className="weight-snapshot-chip score-overview-chip">
                                      <strong>当前</strong>
                                      <span>{Number(item.current_score || 0).toFixed(4)}</span>
                                    </span>
                                  </div>
                                  <span className={`score-delta-chip score-overview-delta ${deltaPositive ? "score-delta-positive" : deltaNegative ? "score-delta-negative" : ""}`}>
                                    变化 {formatSignedNumber(item.score_delta || 0, 4)}
                                  </span>
                                </div>
                              </td>
                              <td>
                                <div className="score-adjustment-stack">
                                  <span className={`score-delta-chip ${Number(item.performance_score_delta || 0) > 0 ? "score-delta-positive" : Number(item.performance_score_delta || 0) < 0 ? "score-delta-negative" : ""}`}>
                                    {formatSignedNumber(item.performance_score_delta || 0, 4)}
                                  </span>
                                  <div className="score-adjustment-breakdown-grid">
                                    {getSignedWeightSnapshotEntries(item.performance_delta_breakdown || {}).length ? (
                                      getSignedWeightSnapshotEntries(item.performance_delta_breakdown || {}).map(([key, value]) => (
                                        <span className={`score-adjustment-breakdown-chip ${Number(value || 0) > 0 ? "score-adjustment-breakdown-positive" : Number(value || 0) < 0 ? "score-adjustment-breakdown-negative" : ""}`} key={`current-run-performance-${item.objective}-${item.scene}-${item.deployment_id}-${key}`}>
                                          <strong>{formatWeightDimensionLabel(key)}</strong>
                                          <span>{formatSignedNumber(value, 4)}</span>
                                        </span>
                                      ))
                                    ) : (
                                      <span className="table-note">当前无明显表现层变化</span>
                                    )}
                                  </div>
                                </div>
                              </td>
                              <td>
                                <div className="score-adjustment-stack">
                                  <div className="score-adjustment-breakdown-grid">
                                    {getSignedWeightSnapshotEntries(item.no_feedback_score_breakdown || {}).length ? (
                                      getSignedWeightSnapshotEntries(item.no_feedback_score_breakdown || {}).map(([key, value]) => (
                                        <span className={`score-adjustment-breakdown-chip ${Number(value || 0) > 0 ? "score-adjustment-breakdown-positive" : Number(value || 0) < 0 ? "score-adjustment-breakdown-negative" : ""}`} key={`current-run-no-feedback-${item.objective}-${item.scene}-${item.deployment_id}-${key}`}>
                                          <strong>无反馈 {formatWeightDimensionLabel(key)}</strong>
                                          <span>{formatSignedNumber(value, 4)}</span>
                                        </span>
                                      ))
                                    ) : null}
                                    {getSignedWeightSnapshotEntries(item.feedback_score_breakdown || {}).length ? (
                                      getSignedWeightSnapshotEntries(item.feedback_score_breakdown || {}).map(([key, value]) => (
                                        <span className={`score-adjustment-breakdown-chip ${Number(value || 0) > 0 ? "score-adjustment-breakdown-positive" : Number(value || 0) < 0 ? "score-adjustment-breakdown-negative" : ""}`} key={`current-run-feedback-${item.objective}-${item.scene}-${item.deployment_id}-${key}`}>
                                          <strong>有反馈 {formatWeightDimensionLabel(key)}</strong>
                                          <span>{formatSignedNumber(value, 4)}</span>
                                        </span>
                                      ))
                                    ) : null}
                                    {!getSignedWeightSnapshotEntries(item.no_feedback_score_breakdown || {}).length && !getSignedWeightSnapshotEntries(item.feedback_score_breakdown || {}).length ? (
                                      <span className="table-note">当前无明显权重层反馈</span>
                                    ) : null}
                                  </div>
                                </div>
                              </td>
                              <td className="limitation-cell score-evolution-note-cell">{item.change_note || "暂无变化说明"}</td>
                            </tr>
                          );
                        }) : (
                          <tr>
                            <td className="table-note" colSpan="6">当前这次任务调度还没有可展示的 Deployment 表现反馈。</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                    <PaginationBar
                      currentPage={currentRunModelScorePagination.page}
                      totalPages={currentRunModelScorePagination.totalPages}
                      pageSize={currentRunModelScorePagination.pageSize}
                      totalItems={currentRunModelScorePagination.totalItems}
                      onChange={currentRunModelScorePagination.setPage}
                    />
                  </div>
                </div>
                <div className="actions">
                  <NavLink className="btn-linklike btn-primary" to="/task-routing-history">
                    去调度历史看长期累计分析
                  </NavLink>
                </div>
              </>
            ) : (
              <>
                <EmptyState
                  icon="S"
                  title="尚未生成本次学习反馈"
                  note="下方两张表只用于解释“本次刚执行的任务调度”和“本次刚提交的结果反馈”如何影响学习过程。你还没运行本次调度时，这里不会展示长期累计分析；长期统计请去“调度历史”页查看。"
                />
                <div className="actions">
                  <NavLink className="btn-linklike btn-primary" to="/task-routing-history">
                    打开调度历史页
                  </NavLink>
                </div>
              </>
            )}
          </section>
        </>
      ) : null}

    </>
  );
}

function TaskRoutingHistoryPage({ derived }) {
  const objectiveOptions = [
    { value: "accuracy", label: "高准确" },
    { value: "coverage", label: "高覆盖" },
    { value: "latency", label: "低时延" },
    { value: "balanced", label: "综合平衡" },
    { value: "resource", label: "低成本" },
    { value: "domain", label: "领域优先" },
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
    model_score_summaries: [],
    weight_profiles: [],
    recent_decisions: [],
    insights: [],
  };
  const completedBenchmarks = sortByCreatedDesc(derived.benchmarks).filter((item) => item.status === "completed");
  const calibrationArchiveCount = derived.predictArchives.length;
  const [historyObjectiveFilter, setHistoryObjectiveFilter] = useState("all");
  const [historyDecisionFilter, setHistoryDecisionFilter] = useState("all");
  const [historyModelScoreObjectiveFilter, setHistoryModelScoreObjectiveFilter] = useState("all");
  const [historyModelScoreSceneFilter, setHistoryModelScoreSceneFilter] = useState("all");
  const [historyModelScoreModelFilter, setHistoryModelScoreModelFilter] = useState("all");
  const historyObjectiveOptions = [
    { value: "all", label: "全部目标", count: feedbackOverview.total_decisions || 0 },
    ...objectiveOptions.map((item) => ({
      ...item,
      count: (feedbackOverview.objective_summaries || []).find((summary) => summary.objective === item.value)?.decision_count || 0,
    })),
  ];
  const filteredObjectiveSummaries = historyObjectiveFilter === "all"
    ? (feedbackOverview.objective_summaries || [])
    : (feedbackOverview.objective_summaries || []).filter((item) => item.objective === historyObjectiveFilter);
  const filteredWeightProfiles = historyObjectiveFilter === "all"
    ? (feedbackOverview.weight_profiles || [])
    : (feedbackOverview.weight_profiles || []).filter((item) => item.objective === historyObjectiveFilter);
  const taskRoutingHistoryRows = useMemo(
    () =>
      sortByCreatedDesc(
        (derived.predictArchives || []).filter((item) => String(item.task_routing_decision_id || "").trim()),
      ),
    [derived.predictArchives],
  );
  const filteredTaskRoutingHistoryRows = useMemo(
    () =>
      historyObjectiveFilter === "all"
        ? taskRoutingHistoryRows
        : taskRoutingHistoryRows.filter((item) => String(item.task_routing_objective || "").trim().toLowerCase() === historyObjectiveFilter),
    [historyObjectiveFilter, taskRoutingHistoryRows],
  );
  function getHistoryDecisionKey(item) {
    return String(item?.task_routing_decision_id || item?.archive_id || "").trim();
  }
  const isAllHistoryDecisionSelected = historyDecisionFilter === "all";
  useEffect(() => {
    const availableKeys = filteredTaskRoutingHistoryRows.map((item) => getHistoryDecisionKey(item)).filter(Boolean);
    if (historyDecisionFilter === "all") {
      return;
    }
    if (!availableKeys.length) {
      if (historyDecisionFilter !== "all") {
        setHistoryDecisionFilter("all");
      }
      return;
    }
    if (!availableKeys.includes(historyDecisionFilter)) {
      setHistoryDecisionFilter("all");
    }
  }, [filteredTaskRoutingHistoryRows, historyDecisionFilter]);
  const selectedHistoryDecision = useMemo(
    () => (isAllHistoryDecisionSelected
      ? null
      : filteredTaskRoutingHistoryRows.find((item) => getHistoryDecisionKey(item) === historyDecisionFilter) || null),
    [filteredTaskRoutingHistoryRows, historyDecisionFilter, isAllHistoryDecisionSelected],
  );
  const selectedHistoryDecisionObjective = String(selectedHistoryDecision?.task_routing_objective || "").trim().toLowerCase();
  const selectedHistoryDecisionWeightProfile = useMemo(
    () =>
      selectedHistoryDecisionObjective
        ? (feedbackOverview.weight_profiles || []).find((item) => String(item.objective || "").trim().toLowerCase() === selectedHistoryDecisionObjective) || null
        : null,
    [feedbackOverview.weight_profiles, selectedHistoryDecisionObjective],
  );
  const selectedHistoryDecisionResponsePayload = selectedHistoryDecision?.response_payload || {};
  const selectedHistoryDecisionPayloadWeights =
    selectedHistoryDecisionResponsePayload?.objective_weights ||
    selectedHistoryDecisionResponsePayload?.objective_weight_snapshot ||
    selectedHistoryDecision?.objective_weights ||
    {};
  const selectedHistoryDecisionPayloadAdaptiveAdjustments =
    selectedHistoryDecisionResponsePayload?.adaptive_adjustments ||
    selectedHistoryDecision?.adaptive_adjustments ||
    {};
  const selectedHistoryDecisionPayloadUnsupervisedAdjustments =
    selectedHistoryDecisionResponsePayload?.adaptive_unsupervised_adjustments ||
    selectedHistoryDecision?.task_routing_unsupervised_adjustment_snapshot ||
    selectedHistoryDecision?.adaptive_unsupervised_adjustments ||
    {};
  const selectedHistoryDecisionPayloadFeedbackAdjustments =
    selectedHistoryDecisionResponsePayload?.adaptive_feedback_adjustments ||
    selectedHistoryDecision?.task_routing_feedback_adjustment_snapshot ||
    selectedHistoryDecision?.adaptive_feedback_adjustments ||
    {};
  const selectedHistoryDecisionWeightDelta = useMemo(() => {
    if (Object.keys(selectedHistoryDecisionPayloadWeights || {}).length) {
      return buildSnapshotDelta(selectedHistoryDecisionPayloadWeights, selectedHistoryDecisionWeightProfile?.effective_weights || {});
    }
    if (Object.keys(selectedHistoryDecisionPayloadAdaptiveAdjustments || {}).length) {
      return selectedHistoryDecisionPayloadAdaptiveAdjustments || {};
    }
    return {};
  }, [selectedHistoryDecisionPayloadAdaptiveAdjustments, selectedHistoryDecisionPayloadWeights, selectedHistoryDecisionWeightProfile?.effective_weights]);
  const selectedHistoryDecisionAfterWeights = useMemo(() => {
    if (Object.keys(selectedHistoryDecisionPayloadWeights || {}).length) {
      return selectedHistoryDecisionPayloadWeights;
    }
    if (Object.keys(selectedHistoryDecisionWeightDelta || {}).length) {
      return applySnapshotDelta(selectedHistoryDecisionWeightProfile?.effective_weights || {}, selectedHistoryDecisionWeightDelta);
    }
    return {};
  }, [selectedHistoryDecisionPayloadWeights, selectedHistoryDecisionWeightDelta, selectedHistoryDecisionWeightProfile?.effective_weights]);
  const decisionScopedObjectiveSummaries = useMemo(() => {
    if (!selectedHistoryDecision) {
      return filteredObjectiveSummaries;
    }
    const feedbackStatus = normalizePredictionFeedbackStatus(selectedHistoryDecision.feedback_status);
    return [{
      objective: selectedHistoryDecisionObjective || selectedHistoryDecision?.task_routing_objective || "unknown",
      decision_count: 1,
      feedback_count: feedbackStatus === "pending" ? 0 : 1,
      acceptance_rate: feedbackStatus === "satisfied" ? 1 : feedbackStatus === "unsatisfied" ? 0 : null,
      best_execution_path: selectedHistoryDecision.task_routing_execution_path || "none",
      fusion_policy: selectedHistoryDecision.task_routing_fusion_policy || "empty",
      triple_count: Number(selectedHistoryDecision.triple_count || 0),
      feedback_status: feedbackStatus,
    }];
  }, [filteredObjectiveSummaries, selectedHistoryDecision, selectedHistoryDecisionObjective]);
  const decisionScopedWeightProfiles = useMemo(() => {
    if (!selectedHistoryDecision) {
      return filteredWeightProfiles;
    }
    return [{
      objective: selectedHistoryDecisionObjective || selectedHistoryDecision?.task_routing_objective || "unknown",
      baseline_weights: selectedHistoryDecisionWeightProfile?.effective_weights || {},
      decision_delta: selectedHistoryDecisionWeightDelta || {},
      decision_unsupervised_adjustments: selectedHistoryDecisionPayloadUnsupervisedAdjustments || {},
      decision_feedback_adjustments: selectedHistoryDecisionPayloadFeedbackAdjustments || {},
      effective_weights: selectedHistoryDecisionAfterWeights || {},
      reasons: selectedHistoryDecisionWeightProfile?.reasons || [],
      data_ready: Boolean(Object.keys(selectedHistoryDecisionAfterWeights || {}).length || Object.keys(selectedHistoryDecisionWeightDelta || {}).length),
      split_data_ready: Boolean(
        Object.keys(selectedHistoryDecisionPayloadUnsupervisedAdjustments || {}).length ||
        Object.keys(selectedHistoryDecisionPayloadFeedbackAdjustments || {}).length
      ),
    }];
  }, [
    filteredWeightProfiles,
    selectedHistoryDecision,
    selectedHistoryDecisionAfterWeights,
    selectedHistoryDecisionObjective,
    selectedHistoryDecisionPayloadFeedbackAdjustments,
    selectedHistoryDecisionPayloadUnsupervisedAdjustments,
    selectedHistoryDecisionWeightDelta,
    selectedHistoryDecisionWeightProfile?.effective_weights,
    selectedHistoryDecisionWeightProfile?.reasons,
  ]);
  const baseHistoryModelScoreRows = useMemo(() => {
    let rows = [...(feedbackOverview.model_score_summaries || [])];
    if (historyObjectiveFilter !== "all") {
      rows = rows.filter((item) => String(item.objective || "").trim().toLowerCase() === historyObjectiveFilter);
    }
    if (selectedHistoryDecision) {
      const selectedModels = new Set((selectedHistoryDecision.task_routing_selected_model_ids || []).map((item) => String(item || "").trim()).filter(Boolean));
      const selectedScene = String(selectedHistoryDecision.scene || "").trim();
      const selectedObjective = String(selectedHistoryDecision.task_routing_objective || "").trim().toLowerCase();
      if (selectedObjective) {
        rows = rows.filter((item) => String(item.objective || "").trim().toLowerCase() === selectedObjective);
      }
      if (selectedScene) {
        rows = rows.filter((item) => String(item.scene || "").trim() === selectedScene);
      }
      if (selectedModels.size) {
        rows = rows.filter((item) => selectedModels.has(String(item.model_id || "").trim()));
      }
    }
    return rows.sort((left, right) => {
      const rightDelta = Math.abs(Number(right.score_delta || 0));
      const leftDelta = Math.abs(Number(left.score_delta || 0));
      if (rightDelta !== leftDelta) {
        return rightDelta - leftDelta;
      }
      return String(left.deployment_id || "").localeCompare(String(right.deployment_id || ""));
    });
  }, [feedbackOverview.model_score_summaries, historyObjectiveFilter, selectedHistoryDecision]);
  const historyModelScoreSceneOptions = useMemo(() => {
    const uniqueScenes = Array.from(new Set(baseHistoryModelScoreRows.map((item) => String(item.scene || "").trim()).filter(Boolean)));
    return ["all", ...uniqueScenes];
  }, [baseHistoryModelScoreRows]);
  const historyModelScoreModelOptions = useMemo(() => {
    const uniqueModels = Array.from(new Set(baseHistoryModelScoreRows.map((item) => String(item.model_id || "").trim()).filter(Boolean)));
    return ["all", ...uniqueModels];
  }, [baseHistoryModelScoreRows]);
  useEffect(() => {
    if (historyModelScoreObjectiveFilter !== "all" && !objectiveOptions.some((item) => item.value === historyModelScoreObjectiveFilter)) {
      setHistoryModelScoreObjectiveFilter("all");
    }
  }, [historyModelScoreObjectiveFilter, objectiveOptions]);
  useEffect(() => {
    if (historyModelScoreSceneFilter !== "all" && !historyModelScoreSceneOptions.includes(historyModelScoreSceneFilter)) {
      setHistoryModelScoreSceneFilter("all");
    }
  }, [historyModelScoreSceneFilter, historyModelScoreSceneOptions]);
  useEffect(() => {
    if (historyModelScoreModelFilter !== "all" && !historyModelScoreModelOptions.includes(historyModelScoreModelFilter)) {
      setHistoryModelScoreModelFilter("all");
    }
  }, [historyModelScoreModelFilter, historyModelScoreModelOptions]);
  const filteredHistoryModelScoreRows = useMemo(() => {
    let rows = [...baseHistoryModelScoreRows];
    if (historyModelScoreObjectiveFilter !== "all") {
      rows = rows.filter((item) => String(item.objective || "").trim().toLowerCase() === historyModelScoreObjectiveFilter);
    }
    if (historyModelScoreSceneFilter !== "all") {
      rows = rows.filter((item) => String(item.scene || "").trim() === historyModelScoreSceneFilter);
    }
    if (historyModelScoreModelFilter !== "all") {
      rows = rows.filter((item) => String(item.model_id || "").trim() === historyModelScoreModelFilter);
    }
    return rows;
  }, [baseHistoryModelScoreRows, historyModelScoreModelFilter, historyModelScoreObjectiveFilter, historyModelScoreSceneFilter]);
  const filteredExecutionPathSummaries = useMemo(() => {
    const grouped = new Map();
    filteredTaskRoutingHistoryRows.forEach((item) => {
      const key = String(item.task_routing_execution_path || "").trim();
      if (!key) {
        return;
      }
      const current = grouped.get(key) || {
        execution_path: key,
        decision_count: 0,
        feedback_count: 0,
        objective_counter: new Map(),
      };
      current.decision_count += 1;
      if (normalizePredictionFeedbackStatus(item.feedback_status) !== "pending") {
        current.feedback_count += 1;
      }
      const objectiveKey = String(item.task_routing_objective || "").trim().toLowerCase();
      if (objectiveKey) {
        current.objective_counter.set(objectiveKey, (current.objective_counter.get(objectiveKey) || 0) + 1);
      }
      grouped.set(key, current);
    });
    return Array.from(grouped.values())
      .map((item) => ({
        execution_path: item.execution_path,
        decision_count: item.decision_count,
        feedback_count: item.feedback_count,
        objective_labels: Array.from(item.objective_counter.entries())
          .sort((left, right) => right[1] - left[1])
          .slice(0, 3)
          .map(([objective, count]) => `${formatObjectiveLabel(objective)} ${count}`),
      }))
      .sort((left, right) => right.decision_count - left.decision_count || right.feedback_count - left.feedback_count || left.execution_path.localeCompare(right.execution_path));
  }, [filteredTaskRoutingHistoryRows]);
  const filteredFusionPolicySummaries = useMemo(() => {
    const grouped = new Map();
    filteredTaskRoutingHistoryRows.forEach((item) => {
      const key = String(item.task_routing_fusion_policy || "").trim() || "empty";
      const current = grouped.get(key) || {
        fusion_policy: key,
        decision_count: 0,
        feedback_count: 0,
        triple_total: 0,
        objective_counter: new Map(),
      };
      current.decision_count += 1;
      current.triple_total += Number(item.triple_count || 0);
      if (normalizePredictionFeedbackStatus(item.feedback_status) !== "pending") {
        current.feedback_count += 1;
      }
      const objectiveKey = String(item.task_routing_objective || "").trim().toLowerCase();
      if (objectiveKey) {
        current.objective_counter.set(objectiveKey, (current.objective_counter.get(objectiveKey) || 0) + 1);
      }
      grouped.set(key, current);
    });
    return Array.from(grouped.values())
      .map((item) => ({
        fusion_policy: item.fusion_policy,
        decision_count: item.decision_count,
        feedback_count: item.feedback_count,
        avg_triple_count: item.decision_count ? item.triple_total / item.decision_count : 0,
        objective_labels: Array.from(item.objective_counter.entries())
          .sort((left, right) => right[1] - left[1])
          .slice(0, 3)
          .map(([objective, count]) => `${formatObjectiveLabel(objective)} ${count}`),
      }))
      .sort((left, right) => right.decision_count - left.decision_count || right.feedback_count - left.feedback_count || left.fusion_policy.localeCompare(right.fusion_policy));
  }, [filteredTaskRoutingHistoryRows]);
  const objectiveSummaryPagination = useTablePagination(decisionScopedObjectiveSummaries, 5);
  const weightProfilePagination = useTablePagination(decisionScopedWeightProfiles, 5);
  const historyModelScorePagination = useTablePagination(filteredHistoryModelScoreRows, 5);
  const executionPathPagination = useTablePagination(filteredExecutionPathSummaries, 5);
  const fusionPolicyPagination = useTablePagination(filteredFusionPolicySummaries, 5);

  return (
    <>
      <section className="page-shell card section-shell score-evolution-section">
        <SectionHeader
          kicker={selectedHistoryDecision ? "单条决策视角" : "历史累计视角"}
          title={selectedHistoryDecision ? "任务调度单条决策分析" : "任务调度历史累计总览"}
          note={selectedHistoryDecision
            ? "当前已选中一条真实决策记录。下方表 1 和表 2 将不再展示长期累计结果，而是改为这条决策本身带来的目标概览与评分权重变化。"
            : "这一页只看长期沉淀，不看本次临时输入。它回答平台长期在哪些目标下运行更多、已经形成了怎样的调度权重，以及常见执行路径与最近真实决策长什么样。"}
        />
        <div className="metrics">
          <MetricCard title="历史决策总数" value={feedbackOverview.total_decisions || 0} note="累计沉淀的任务调度记录" />
          <MetricCard title="历史反馈总数" value={feedbackOverview.total_feedback || 0} note="任务调度反馈与结果反馈共同支撑长期校准" />
          <MetricCard title="Benchmark 证据" value={completedBenchmarks.length} note="长期参与权重修正的基准记录" />
          <MetricCard title="结果反馈证据" value={calibrationArchiveCount} note="自动留痕与用户反馈形成的历史样本" />
        </div>
        <div className="objective-filter-shell">
          <div className="table-note">调度目标分类：</div>
          <div className="actions objective-filter-actions">
            {historyObjectiveOptions.map((item) => (
              <button
                key={`history-objective-${item.value}`}
                className={historyObjectiveFilter === item.value ? "button-active" : ""}
                onClick={() => setHistoryObjectiveFilter(item.value)}
                type="button"
              >
                {item.label}
                <span className="objective-filter-count">{item.count}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="form-grid">
          <label className="stack-field">
            <span>决策记录筛选</span>
            <select
              onChange={(event) => setHistoryDecisionFilter(event.target.value)}
              value={historyDecisionFilter}
            >
              <option value="all">全部记录</option>
              {filteredTaskRoutingHistoryRows.map((item) => {
                const decisionKey = getHistoryDecisionKey(item);
                return (
                  <option key={`history-decision-${decisionKey}`} value={decisionKey}>
                    {`${decisionKey} | ${formatObjectiveLabel(item.task_routing_objective)} | ${item.created_at || "-"}`}
                  </option>
                );
              })}
            </select>
            <small className="detail-note">
              这里用于查看某一条真实决策记录；也支持切回“全部记录”。选中单条后，表 1 和表 2 会切成“单条决策影响视角”，同时页面下方也会同步聚焦这条记录的执行路径、融合策略、选中模型，以及对应模型部署的表现变化。
            </small>
          </label>
        </div>
        <div className="banner banner-muted">
          {selectedHistoryDecision
            ? "阅读顺序建议：先看“单条决策详情”，再看表 1 的本次决策影响概览和表 2 的本次决策评分权重变化；随后看“Deployment 表现变化”，最后再用执行路径与融合策略核对这条记录在平台中的定位。"
            : "阅读顺序建议：先看“单条决策详情”，再看“各目标历史累计概览”和“目标调度评分权重（历史累计）”；随后看“历史 Deployment 表现变化”，最后再用“执行路径分布”和“融合策略历史”核对长期统计。"}
        </div>
        <div className="relation-list">
          {(feedbackOverview.insights || []).map((item, index) => (
            <div className="relation-item" key={`${item}-${index + 1}`}>
              {item}
            </div>
          ))}
        </div>
      </section>

      <section className="page-shell card section-shell score-evolution-section">
        <SectionHeader
          kicker="单条记录"
          title="单条决策详情"
          note="这里专门展开上方筛选器当前选中的那一条任务调度记录，方便你单独核对这次调度到底用了什么路径、什么融合策略、选了哪些模型，以及后续有没有收到反馈。"
        />
        {selectedHistoryDecision ? (
          <>
            <div className="metrics">
              <MetricCard
                title="决策编号"
                value={selectedHistoryDecision.task_routing_decision_id || selectedHistoryDecision.archive_id || "-"}
                note={`时间 ${selectedHistoryDecision.created_at || "-"}`}
              />
              <MetricCard
                title="任务调度目标"
                value={formatObjectiveLabel(selectedHistoryDecision.task_routing_objective)}
                note={`${selectedHistoryDecision.scene || "-"} / ${formatLanguageLabel(selectedHistoryDecision.inferred_language)}`}
              />
              <MetricCard
                title="执行路径"
                value={formatExecutionPathLabel(selectedHistoryDecision.task_routing_execution_path)}
                note={selectedHistoryDecision.task_routing_execution_path || "-"}
              />
              <MetricCard
                title="融合策略"
                value={formatFusionPolicyLabel(selectedHistoryDecision.task_routing_fusion_policy)}
                note={selectedHistoryDecision.task_routing_fusion_policy || "-"}
              />
            </div>
            <div className="detail-grid">
              <article className="card detail-card">
                <div className="table-subhead">记录内容概览</div>
                <div className="relation-list">
                  <div className="relation-item">
                    <strong>选中模型</strong>
                    <br />
                    {(selectedHistoryDecision.task_routing_selected_model_ids || []).join(" / ") || "当前记录未写入选中模型。"}
                  </div>
                  <div className="relation-item">
                    <strong>三元组产出</strong>
                    <br />
                    {`本次共输出 ${selectedHistoryDecision.triple_count || 0} 个三元组。`}
                  </div>
                  <div className="relation-item">
                    <strong>执行路径说明</strong>
                    <br />
                    {formatExecutionPathNote(selectedHistoryDecision.task_routing_execution_path)}
                  </div>
                  <div className="relation-item">
                    <strong>融合策略说明</strong>
                    <br />
                    {formatFusionPolicyNote(selectedHistoryDecision.task_routing_fusion_policy)}
                  </div>
                </div>
              </article>

              <article className="card detail-card">
                <div className="table-subhead">反馈与留痕状态</div>
                <div className="relation-list">
                  <div className="relation-item">
                    <strong>反馈状态</strong>
                    <br />
                    {formatPredictionFeedbackSummary(selectedHistoryDecision.feedback_status, selectedHistoryDecision.feedback_note)}
                  </div>
                  <div className="relation-item">
                    <strong>不满意原因</strong>
                    <br />
                    {selectedHistoryDecision.feedback_note
                      ? formatPredictionFeedbackReasonSummary(selectedHistoryDecision.feedback_note)
                      : "当前记录还没有补充不满意原因。"}
                  </div>
                  <div className="relation-item">
                    <strong>归档编号</strong>
                    <br />
                    {selectedHistoryDecision.archive_id || "-"}
                  </div>
                  <div className="relation-item">
                    <strong>历史定位</strong>
                    <br />
                    这条记录来自任务调度真实留痕，可用于回看单次调度在长期学习过程中的具体输入证据。
                  </div>
                </div>
              </article>
            </div>
          </>
        ) : (
          <EmptyState
            icon="D"
            title={filteredTaskRoutingHistoryRows.length ? "当前处于全部记录视角" : "当前没有可查看的决策详情"}
            note={filteredTaskRoutingHistoryRows.length
              ? "你当前选择的是“全部记录”，因此这里不展开单条决策详情。若要查看某次任务调度的具体路径、融合策略和反馈状态，请在上方筛选器中选中某一条记录。"
              : "先在上方选择一个任务调度目标分类，或等待平台产生至少一条任务调度历史记录。"}
          />
        )}
      </section>

      <section className="page-shell card section-shell score-evolution-section">
        <div className="score-evolution-block">
          <div className="table-subhead">{selectedHistoryDecision ? "表 1：本次决策影响概览" : "表 1：各目标历史累计概览"}</div>
          <div className="banner banner-muted">
            {selectedHistoryDecision
              ? "这张表看什么：当前只看你选中的这一条决策记录。它不再统计长期累计，而是直接展示这次决策属于哪个目标、这次有没有反馈、采用了哪条执行路径，以及融合输出的大致规模。"
              : "这张表看什么：一行就是一个任务调度目标。顶部切到某个目标分类后，这里会只保留该目标的历史累计结果；切回“全部目标”则恢复全量目标对比。"}
          </div>
          <div className="table-wrap">
            <table className="weight-profile-table">
              <thead>
                <tr>
                  <th>任务调度目标</th>
                  <th>{selectedHistoryDecision ? "本次决策数" : "历史决策数"}</th>
                  <th>{selectedHistoryDecision ? "本次反馈数" : "历史反馈数"}</th>
                  <th>{selectedHistoryDecision ? "本次反馈结果" : "接受率"}</th>
                  <th>{selectedHistoryDecision ? "本次执行路径" : "主参考执行路径"}</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {objectiveSummaryPagination.pagedItems.map((item) => (
                  <tr key={item.objective}>
                    <td>
                      <div className="archive-cell-stack">
                        <div className="archive-main">{formatObjectiveLabel(item.objective)}</div>
                        <div className="table-note">{item.objective}</div>
                      </div>
                    </td>
                    <td>{item.decision_count}</td>
                    <td>{item.feedback_count}</td>
                    <td>{selectedHistoryDecision ? formatPredictionFeedbackLabel(item.feedback_status) : formatPercent(item.acceptance_rate)}</td>
                    <td>
                      <div>{formatExecutionPathLabel(item.best_execution_path)}</div>
                      <div className="table-note">{item.best_execution_path || "-"}</div>
                    </td>
                    <td className="limitation-cell">
                      {selectedHistoryDecision
                        ? `这条决策采用“${formatExecutionPathLabel(item.best_execution_path)}”路径，融合策略为“${formatFusionPolicyLabel(item.fusion_policy)}”，最终输出 ${item.triple_count || 0} 个三元组。`
                        : item.decision_count > 0
                          ? `该目标已有 ${item.decision_count} 条历史调度记录，长期主要参考“${formatExecutionPathLabel(item.best_execution_path)}”路径。`
                          : "当前还没有该目标的历史调度记录。"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PaginationBar
              currentPage={objectiveSummaryPagination.page}
              totalPages={objectiveSummaryPagination.totalPages}
              pageSize={objectiveSummaryPagination.pageSize}
              totalItems={objectiveSummaryPagination.totalItems}
              onChange={objectiveSummaryPagination.setPage}
            />
          </div>
        </div>

        <div className="score-evolution-block">
          <div className="table-subhead">{selectedHistoryDecision ? "表 2：本次决策造成的评分权重变化" : "表 2：目标调度评分权重（历史累计）"}</div>
          <div className="banner banner-muted">
            {selectedHistoryDecision
              ? "这张表看什么：当前只看这条决策本身。它展示该目标在“决策前稳定权重”基础上，本次调度分别叠加了多少无用户反馈校准修正和有用户反馈校准修正，并给出本次真正落地使用的权重快照。"
              : "这张表看什么：这是调度历史页里的“目标调度评分权重”版本，口径是长期累计而不是单次运行。顶部如果选择了某个调度目标分类，这里就只展示该目标长期学到的权重；如果保持“全部目标”，就能横向比较六类目标的长期打分规则差异。"}
          </div>
          <div className="table-wrap">
            <table className="weight-profile-table">
              <thead>
                <tr>
                  <th>任务调度目标</th>
                  <th>{selectedHistoryDecision ? "决策前稳定权重" : "当前有效权重"}</th>
                  <th>{selectedHistoryDecision ? "本次无用户反馈校准修正" : "Benchmark 修正"}</th>
                  <th>{selectedHistoryDecision ? "本次有用户反馈校准修正" : "无用户反馈校准修正"}</th>
                  <th>{selectedHistoryDecision ? "决策后实际权重" : "有用户反馈校准修正"}</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {weightProfilePagination.pagedItems.map((item) => (
                  <tr key={item.objective}>
                    <td>
                      <div className="archive-cell-stack">
                        <div className="archive-main">{formatObjectiveLabel(item.objective)}</div>
                        <div className="table-note">{item.objective}</div>
                      </div>
                    </td>
                    <td>
                      {selectedHistoryDecision
                        ? renderWeightSnapshotGrid(item.baseline_weights || {}, `${item.objective}-baseline`)
                        : (
                          <>
                            {renderWeightSnapshotGrid(item.effective_weights || {}, `${item.objective}-effective`)}
                            <div className="table-note">{(item.reasons || []).slice(0, 2).join("；") || "暂无额外说明"}</div>
                          </>
                        )}
                    </td>
                    <td>
                      {selectedHistoryDecision
                        ? item.split_data_ready
                          ? Object.keys(item.decision_unsupervised_adjustments || {}).length
                            ? renderWeightSnapshotGrid(item.decision_unsupervised_adjustments || {}, `${item.objective}-decision-unsupervised`, {
                              variantClass: "weight-snapshot-chip-unsupervised",
                              signed: true,
                            })
                            : <span className="table-note">本次未触发该类修正</span>
                          : <span className="table-note">{item.data_ready ? "旧记录未拆分" : "未记录"}</span>
                        : renderWeightSnapshotGrid(item.benchmark_adjustments || {}, `${item.objective}-benchmark`, {
                          variantClass: "weight-snapshot-chip-benchmark",
                          signed: true,
                        })}
                    </td>
                    <td>
                      {selectedHistoryDecision
                        ? item.split_data_ready
                          ? Object.keys(item.decision_feedback_adjustments || {}).length
                            ? renderWeightSnapshotGrid(item.decision_feedback_adjustments || {}, `${item.objective}-decision-feedback`, {
                              variantClass: "weight-snapshot-chip-feedback",
                              signed: true,
                            })
                            : <span className="table-note">本次未触发该类修正</span>
                          : <span className="table-note">{item.data_ready ? "旧记录未拆分" : "未记录"}</span>
                        : renderWeightSnapshotGrid(item.unsupervised_adjustments || {}, `${item.objective}-unsupervised`, {
                          variantClass: "weight-snapshot-chip-unsupervised",
                          signed: true,
                        })}
                    </td>
                    <td>
                      {selectedHistoryDecision ? (
                        renderWeightSnapshotGrid(item.effective_weights || {}, `${item.objective}-after`)
                      ) : (
                        renderWeightSnapshotGrid(item.feedback_adjustments || {}, `${item.objective}-feedback`, {
                          variantClass: "weight-snapshot-chip-feedback",
                          signed: true,
                        })
                      )}
                    </td>
                    <td>
                      {selectedHistoryDecision ? (
                        <div className="archive-cell-stack">
                          <div className="table-note">
                            {item.split_data_ready
                              ? "当前优先使用这条归档记录中真实写入的“无用户反馈 / 有用户反馈”两类修正快照。"
                              : item.data_ready
                              ? "这条记录写入了最终权重快照，但生成时还是旧格式，未单独落盘两类修正，因此这里只能提示为“未拆分”。"
                              : "这条决策记录没有写入足够的权重字段，因此当前无法完整还原本次权重变化。"}
                          </div>
                          <div className="table-note">
                            {(item.reasons || []).slice(0, 2).join("；") || "暂无额外说明"}
                          </div>
                        </div>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PaginationBar
              currentPage={weightProfilePagination.page}
              totalPages={weightProfilePagination.totalPages}
              pageSize={weightProfilePagination.pageSize}
              totalItems={weightProfilePagination.totalItems}
              onChange={weightProfilePagination.setPage}
            />
          </div>
        </div>
      </section>

      <section className="page-shell card section-shell score-evolution-section">
        <SectionHeader
          kicker="历史部署"
          title="Deployment 表现变化"
          note={selectedHistoryDecision
            ? "当前按单条调度记录过滤，只展示这次任务调度实际用到的 deployment 表现变化。这里的评分单位始终是 deployment 实例。"
            : "当前处于全部记录视角，展示当前目标分类下全部 deployment 的历史表现变化。这里的评分单位始终是 deployment 实例，而不是抽象模型名。"}
        />
        <div className="banner banner-muted">
          这张表看什么：看 deployment 在历史学习后的评分变化。若上方选择“全部记录”，这里是当前目标分类下的全量 deployment；若选中某条决策记录，这里会缩成这次调度实际用到的 deployment。
        </div>
        <div className="form-grid">
          <label className="stack-field">
            <span>目标 Filter</span>
            <select
              onChange={(event) => setHistoryModelScoreObjectiveFilter(event.target.value)}
              value={historyModelScoreObjectiveFilter}
            >
              <option value="all">全部目标</option>
              {objectiveOptions.map((item) => (
                <option key={`history-model-score-objective-${item.value}`} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>场景 Filter</span>
            <select
              onChange={(event) => setHistoryModelScoreSceneFilter(event.target.value)}
              value={historyModelScoreSceneFilter}
            >
              <option value="all">全部场景</option>
              {historyModelScoreSceneOptions.filter((item) => item !== "all").map((item) => (
                <option key={`history-model-score-scene-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>模型 Filter</span>
            <select
              onChange={(event) => setHistoryModelScoreModelFilter(event.target.value)}
              value={historyModelScoreModelFilter}
            >
              <option value="all">全部模型</option>
              {historyModelScoreModelOptions.filter((item) => item !== "all").map((item) => (
                <option key={`history-model-score-model-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="table-wrap">
          <table className="weight-profile-table score-evolution-table score-evolution-primary-table">
            <thead>
              <tr>
                <th>Deployment / 模型</th>
                <th>目标 / 场景</th>
                <th>评分概览</th>
                <th>当前得分拆解</th>
                <th>表现层净变化</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {historyModelScorePagination.pagedItems.length ? historyModelScorePagination.pagedItems.map((item) => {
                const deltaPositive = Number(item.score_delta || 0) > 0;
                const deltaNegative = Number(item.score_delta || 0) < 0;
                return (
                  <tr key={`history-model-score-${item.objective}-${item.scene}-${item.deployment_id}`}>
                    <td className="score-evolution-cell-identity">
                      <div className="archive-cell-stack">
                        <strong className="archive-main">{item.deployment_id}</strong>
                        <div className="table-note">{item.profile_name}</div>
                        <div className="archive-main">{item.model_id}</div>
                        <div className="table-note">{item.dataset_name || "-"}</div>
                      </div>
                    </td>
                    <td>
                      <div>{formatObjectiveLabel(item.objective)}</div>
                      <div className="table-note">{item.scene || "-"}</div>
                    </td>
                    <td>
                      <div className="score-overview-stack">
                        <div className="score-overview-row">
                          <span className="weight-snapshot-chip weight-snapshot-chip-baseline score-overview-chip">
                            <strong>优化前</strong>
                            <span>{Number(item.baseline_score || 0).toFixed(4)}</span>
                          </span>
                          <span className="weight-snapshot-chip score-overview-chip">
                            <strong>当前</strong>
                            <span>{Number(item.current_score || 0).toFixed(4)}</span>
                          </span>
                        </div>
                        <span className={`score-delta-chip score-overview-delta ${deltaPositive ? "score-delta-positive" : deltaNegative ? "score-delta-negative" : ""}`}>
                          变化 {formatSignedNumber(item.score_delta || 0, 4)}
                        </span>
                      </div>
                    </td>
                    <td>
                      <div className="score-breakdown-compact-grid score-breakdown-compact-grid-wide">
                        {getWeightSnapshotEntries(item.current_breakdown || {}).length ? (
                          getWeightSnapshotEntries(item.current_breakdown || {}).map(([key, value]) => (
                            <span className="score-breakdown-compact-chip" key={`history-score-${item.deployment_id}-breakdown-${key}`}>
                              <strong>{formatWeightDimensionLabel(key)}</strong>
                              <span>{Number(value).toFixed(2)}</span>
                            </span>
                          ))
                        ) : (
                          <span className="table-note">无</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <div className="score-breakdown-compact-grid score-breakdown-compact-grid-wide">
                        {getSignedWeightSnapshotEntries(item.performance_delta_breakdown || {}).length ? (
                          getSignedWeightSnapshotEntries(item.performance_delta_breakdown || {}).map(([key, value]) => (
                            <span className={`score-breakdown-compact-chip ${Number(value || 0) > 0 ? "score-adjustment-breakdown-positive" : Number(value || 0) < 0 ? "score-adjustment-breakdown-negative" : ""}`} key={`history-score-${item.deployment_id}-performance-${key}`}>
                              <strong>{formatWeightDimensionLabel(key)}</strong>
                              <span>{formatSignedNumber(value, 4)}</span>
                            </span>
                          ))
                        ) : (
                          <span className="table-note">无变化</span>
                        )}
                      </div>
                    </td>
                    <td className="limitation-cell score-evolution-note-cell">{item.change_note || "暂无变化说明"}</td>
                  </tr>
                );
              }) : (
                <tr>
                  <td className="table-note" colSpan="6">
                    {selectedHistoryDecision
                      ? "当前选中的这条调度记录下，还没有匹配到对应模型部署的表现变化数据。"
                      : "当前筛选条件下，还没有可展示的模型部署表现变化数据。"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          <PaginationBar
            currentPage={historyModelScorePagination.page}
            totalPages={historyModelScorePagination.totalPages}
            pageSize={historyModelScorePagination.pageSize}
            totalItems={historyModelScorePagination.totalItems}
            onChange={historyModelScorePagination.setPage}
          />
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="历史路径" title="执行路径分布" note="这张表看的是平台长期更常用哪种调度方式完成任务，例如单模型直出还是多模型融合。" />
          <div className="banner banner-muted">一行就是一种执行路径。顶部如果选中了某个调度目标分类，这里就统计该目标下的路径分布；如果保持“全部目标”，则展示全平台的历史累计路径分布。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>执行路径</th>
                  <th>中文说明</th>
                  <th>历史决策数</th>
                  <th>历史反馈数</th>
                  <th>关联目标分类</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {executionPathPagination.pagedItems.map((item) => (
                  <tr key={item.execution_path}>
                    <td>{item.execution_path}</td>
                    <td>{formatExecutionPathLabel(item.execution_path)}</td>
                    <td>{item.decision_count}</td>
                    <td>{item.feedback_count}</td>
                    <td>{item.objective_labels?.join(" / ") || "-"}</td>
                    <td className="limitation-cell">
                      {item.decision_count > 0
                        ? `该路径历史上已出现 ${item.decision_count} 次，可用于观察平台长期更偏好的执行组织方式。`
                        : "暂无路径记录。"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PaginationBar
              currentPage={executionPathPagination.page}
              totalPages={executionPathPagination.totalPages}
              pageSize={executionPathPagination.pageSize}
              totalItems={executionPathPagination.totalItems}
              onChange={executionPathPagination.setPage}
            />
          </div>
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="融合历史" title="融合策略历史" note="这张表按融合策略做长期汇总，更适合观察平台历史上更常使用哪种结果合成方式。" />
          <div className="banner banner-muted">一行就是一种融合策略。顶部切换调度目标分类后，这里会同步只保留该目标下的融合策略历史；如果保持“全部目标”，则展示全平台累计的融合方式分布。</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>融合策略</th>
                  <th>历史决策数</th>
                  <th>历史反馈数</th>
                  <th>平均三元组数</th>
                  <th>关联目标分类</th>
                  <th>说明</th>
                </tr>
              </thead>
              <tbody>
                {fusionPolicyPagination.pagedItems.map((item) => (
                  <tr key={item.fusion_policy}>
                    <td>
                      <div>{formatFusionPolicyLabel(item.fusion_policy)}</div>
                      <div className="table-note">{item.fusion_policy || "-"}</div>
                    </td>
                    <td>{item.decision_count}</td>
                    <td>{item.feedback_count}</td>
                    <td>{Number(item.avg_triple_count || 0).toFixed(2)}</td>
                    <td>{item.objective_labels?.join(" / ") || "-"}</td>
                    <td className="limitation-cell">{formatFusionPolicyNote(item.fusion_policy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PaginationBar
              currentPage={fusionPolicyPagination.page}
              totalPages={fusionPolicyPagination.totalPages}
              pageSize={fusionPolicyPagination.pageSize}
              totalItems={fusionPolicyPagination.totalItems}
              onChange={fusionPolicyPagination.setPage}
            />
          </div>
        </article>
      </section>
    </>
  );
}

function DeploymentPerformancePage({ derived }) {
  const deployments = derived.deployments || [];
  const feedbackOverview = derived.taskRoutingFeedbackOverview || { model_score_summaries: [] };
  const modelScoreSummaries = feedbackOverview.model_score_summaries || [];
  const taskRoutingArchives = (derived.predictArchives || []).filter(
    (item) =>
      item.archive_source === "task_routing" ||
      item.archive_source === "task_routing_decision" ||
      String(item.task_routing_decision_id || "").trim(),
  );
  const experimentArchives = (derived.predictArchives || []).filter((item) => item.archive_source !== "task_routing");
  const datasetCatalog = derived.datasetCatalog || { datasets: DEFAULT_PREDICT_DATASETS };

  const [modelFilter, setModelFilter] = useState("all");
  const [deploymentFilter, setDeploymentFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [taskSceneFilter, setTaskSceneFilter] = useState("all");
  const [datasetSceneFilter, setDatasetSceneFilter] = useState("all");
  const [recordDeploymentFilter, setRecordDeploymentFilter] = useState("all");
  const [recordModelFilter, setRecordModelFilter] = useState("all");
  const [recordTypeFilter, setRecordTypeFilter] = useState("all");
  const [recordObjectiveFilter, setRecordObjectiveFilter] = useState("all");
  const [recordTaskSceneFilter, setRecordTaskSceneFilter] = useState("all");
  const [recordDatasetSceneFilter, setRecordDatasetSceneFilter] = useState("all");

  const datasetLabelMap = useMemo(() => {
    const map = new Map();
    (datasetCatalog.datasets || []).forEach((item) => {
      const key = String(item.dataset_key || "").trim();
      if (key) {
        map.set(key, item.label || item.dataset_name || key);
      }
    });
    return map;
  }, [datasetCatalog.datasets]);

  function formatDatasetSceneLabel(value) {
    const key = String(value || "").trim();
    return datasetLabelMap.get(key) || key || "-";
  }

  function collectAverageBreakdown(items, key) {
    const bucket = new Map();
    items.forEach((item) => {
      const breakdown = item?.[key] || {};
      Object.entries(breakdown).forEach(([dimension, value]) => {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          return;
        }
        const current = bucket.get(dimension) || { total: 0, count: 0 };
        current.total += numeric;
        current.count += 1;
        bucket.set(dimension, current);
      });
    });
    return Object.fromEntries(
      Array.from(bucket.entries()).map(([dimension, item]) => [dimension, item.count ? item.total / item.count : 0]),
    );
  }

  const modelOptions = useMemo(
    () => ["all", ...Array.from(new Set(deployments.map((item) => String(item.model_id || "").trim()).filter(Boolean))).sort()],
    [deployments],
  );
  const deploymentOptions = useMemo(
    () => ["all", ...Array.from(new Set(deployments.map((item) => String(item.deployment_id || "").trim()).filter(Boolean))).sort()],
    [deployments],
  );
  const statusOptions = useMemo(
    () => ["all", ...Array.from(new Set(deployments.map((item) => String(item.status || "").trim()).filter(Boolean))).sort()],
    [deployments],
  );
  const taskSceneOptions = useMemo(
    () => ["all", "none", ...Array.from(new Set(modelScoreSummaries.map((item) => String(item.scene || "").trim()).filter(Boolean))).sort()],
    [modelScoreSummaries],
  );
  const datasetSceneOptions = useMemo(
    () => ["all", "none", ...Array.from(new Set(experimentArchives.map((item) => String(item.scene || "").trim()).filter(Boolean))).sort()],
    [experimentArchives],
  );

  useEffect(() => {
    if (!modelOptions.includes(modelFilter)) {
      setModelFilter("all");
    }
  }, [modelFilter, modelOptions]);
  useEffect(() => {
    if (!deploymentOptions.includes(deploymentFilter)) {
      setDeploymentFilter("all");
    }
  }, [deploymentFilter, deploymentOptions]);
  useEffect(() => {
    if (!statusOptions.includes(statusFilter)) {
      setStatusFilter("all");
    }
  }, [statusFilter, statusOptions]);
  useEffect(() => {
    if (!taskSceneOptions.includes(taskSceneFilter)) {
      setTaskSceneFilter("all");
    }
  }, [taskSceneFilter, taskSceneOptions]);
  useEffect(() => {
    if (!datasetSceneOptions.includes(datasetSceneFilter)) {
      setDatasetSceneFilter("all");
    }
  }, [datasetSceneFilter, datasetSceneOptions]);

  const rows = useMemo(() => {
    return deployments.map((deployment) => {
      const deploymentId = String(deployment.deployment_id || "").trim();
      const deploymentModelId = String(deployment.model_id || "").trim();
      const routingRows = modelScoreSummaries.filter((item) => String(item.deployment_id || "").trim() === deploymentId);
      const experimentRows = experimentArchives.filter((item) => String(item.deployment_id || "").trim() === deploymentId);
      const taskScenes = Array.from(new Set(routingRows.map((item) => String(item.scene || "").trim()).filter(Boolean)));
      const datasetScenes = Array.from(new Set(experimentRows.map((item) => String(item.scene || "").trim()).filter(Boolean)));
      const avgCurrentBreakdown = collectAverageBreakdown(routingRows, "current_breakdown");
      const avgPerformanceDeltaBreakdown = collectAverageBreakdown(routingRows, "performance_delta_breakdown");
      const avgBaselineScore = meanBy(routingRows, (item) => item.baseline_score);
      const avgCurrentScore = meanBy(routingRows, (item) => item.current_score);
      const avgScoreDelta = meanBy(routingRows, (item) => item.score_delta);
      const avgLatencyMs = meanBy(experimentRows, (item) => item.latency_ms);
      const avgTripleCount = meanBy(experimentRows, (item) => item.triple_count);
      const taskObjectiveLabels = Array.from(new Set(routingRows.map((item) => String(item.objective || "").trim()).filter(Boolean)));
      const latestExperimentAt = sortByCreatedDesc(experimentRows)[0]?.created_at || "-";
      const matchedTaskRoutingArchives = sortByCreatedDesc(taskRoutingArchives).filter((item) => {
        const archiveDeploymentId = String(item.deployment_id || "").trim();
        if (archiveDeploymentId && archiveDeploymentId === deploymentId) {
          return true;
        }
        const modelHints = collectArchiveModelHints(item);
        return modelHints.includes(deploymentModelId.toLowerCase());
      });

      return {
        deployment,
        routingRows,
        matchedTaskRoutingArchives,
        experimentRows,
        taskScenes,
        datasetScenes,
        taskObjectiveLabels,
        avgCurrentBreakdown,
        avgPerformanceDeltaBreakdown,
        avgBaselineScore,
        avgCurrentScore,
        avgScoreDelta,
        avgLatencyMs,
        avgTripleCount,
        latestExperimentAt,
        hasTaskRouting: routingRows.length > 0,
        hasExperiment: experimentRows.length > 0,
        modelId: deploymentModelId,
      };
    });
  }, [deployments, experimentArchives, modelScoreSummaries, taskRoutingArchives]);

  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (modelFilter !== "all" && row.modelId !== modelFilter) {
        return false;
      }
      if (deploymentFilter !== "all" && String(row.deployment.deployment_id || "").trim() !== deploymentFilter) {
        return false;
      }
      if (statusFilter !== "all" && String(row.deployment.status || "").trim() !== statusFilter) {
        return false;
      }
      if (taskSceneFilter === "none" && row.hasTaskRouting) {
        return false;
      }
      if (taskSceneFilter !== "all" && taskSceneFilter !== "none" && !row.taskScenes.includes(taskSceneFilter)) {
        return false;
      }
      if (datasetSceneFilter === "none" && row.hasExperiment) {
        return false;
      }
      if (datasetSceneFilter !== "all" && datasetSceneFilter !== "none" && !row.datasetScenes.includes(datasetSceneFilter)) {
        return false;
      }
      return true;
    }).sort((left, right) => {
      if (left.deployment.status === "running" && right.deployment.status !== "running") {
        return -1;
      }
      if (left.deployment.status !== "running" && right.deployment.status === "running") {
        return 1;
      }
      return String(left.deployment.deployment_id || "").localeCompare(String(right.deployment.deployment_id || ""));
    });
  }, [datasetSceneFilter, deploymentFilter, modelFilter, rows, statusFilter, taskSceneFilter]);

  const recordRows = useMemo(() => {
    return rows.flatMap((row) => {
      const deploymentId = String(row.deployment.deployment_id || "").trim();
      const taskRoutingRecordRows = row.matchedTaskRoutingArchives.map((archive) => {
        const archiveScene = String(archive.scene || "").trim();
        const archiveObjective = String(archive.task_routing_objective || "").trim();
        const matchedSummary =
          row.routingRows.find(
            (item) =>
              String(item.scene || "").trim() === archiveScene &&
              String(item.objective || "").trim() === archiveObjective,
          ) ||
          row.routingRows.find((item) => String(item.scene || "").trim() === archiveScene) ||
          row.routingRows[0] ||
          null;
        const directMatched = String(archive.deployment_id || "").trim() === deploymentId;
        const performanceDeltaBreakdown = matchedSummary?.performance_delta_breakdown || {};
        const performanceDeltaFromBreakdown = Object.values(performanceDeltaBreakdown).reduce(
          (sum, value) => sum + (Number(value) || 0),
          0,
        );
        return {
          recordType: "task_routing",
          recordId: archive.task_routing_decision_id || archive.archive_id || "-",
          deploymentId,
          modelId: row.deployment.model_id,
          profileName: row.deployment.profile_name,
          status: row.deployment.status,
          createdAt: archive.created_at || "-",
          taskScene: archiveScene || "none",
          datasetScene: "none",
          objective: archiveObjective || "all",
          sourceLabel: formatArchiveSourceLabel(archive.archive_source),
          summaryLabel: formatPredictionFeedbackSummary(archive.feedback_status, archive.feedback_note),
          impactSignals: getArchiveImpactSignals(archive),
          matchedScoreDelta: matchedSummary?.score_delta ?? null,
          matchedPerformanceScoreDelta:
            matchedSummary?.performance_score_delta ??
            (Object.keys(performanceDeltaBreakdown).length ? performanceDeltaFromBreakdown : null),
          matchedPerformanceDeltaBreakdown: performanceDeltaBreakdown,
          latencyMs: archive.latency_ms ?? null,
          tripleCount: archive.triple_count ?? null,
          note: directMatched ? "通过 deployment_id 直接命中" : "通过任务调度记录中的模型提示命中",
        };
      });
      const experimentRecordRows = row.experimentRows.map((archive) => ({
        recordType: "experiment",
        recordId: archive.archive_id || "-",
        deploymentId,
        modelId: row.deployment.model_id,
        profileName: row.deployment.profile_name,
        status: row.deployment.status,
        createdAt: archive.created_at || "-",
        taskScene: "none",
        datasetScene: String(archive.scene || "").trim() || "none",
        objective: "all",
        sourceLabel: formatArchiveSourceLabel(archive.archive_source),
        summaryLabel: formatPredictionFeedbackSummary(archive.feedback_status, archive.feedback_note),
        impactSignals: getArchiveImpactSignals(archive),
        matchedScoreDelta: null,
        matchedPerformanceScoreDelta: null,
        matchedPerformanceDeltaBreakdown: {},
        latencyMs: archive.latency_ms ?? null,
        tripleCount: archive.triple_count ?? null,
        note: "通过 deployment_id 直接命中推理实验归档",
      }));
      return [...taskRoutingRecordRows, ...experimentRecordRows];
    }).sort((left, right) => String(right.createdAt || "").localeCompare(String(left.createdAt || "")));
  }, [rows]);

  const recordObjectiveOptions = useMemo(
    () => ["all", ...Array.from(new Set(recordRows.map((item) => String(item.objective || "").trim()).filter((item) => item && item !== "all"))).sort()],
    [recordRows],
  );
  const recordModelOptions = useMemo(
    () => ["all", ...Array.from(new Set(recordRows.map((item) => String(item.modelId || "").trim()).filter(Boolean))).sort()],
    [recordRows],
  );
  const recordDeploymentOptions = useMemo(() => {
    const scopedRows = recordModelFilter === "all"
      ? recordRows
      : recordRows.filter((item) => String(item.modelId || "").trim() === recordModelFilter);
    return ["all", ...Array.from(new Set(scopedRows.map((item) => String(item.deploymentId || "").trim()).filter(Boolean))).sort()];
  }, [recordModelFilter, recordRows]);
  const recordTaskSceneOptions = useMemo(
    () => ["all", "none", ...Array.from(new Set(recordRows.map((item) => String(item.taskScene || "").trim()).filter((item) => item && item !== "none"))).sort()],
    [recordRows],
  );
  const recordDatasetSceneOptions = useMemo(
    () => ["all", "none", ...Array.from(new Set(recordRows.map((item) => String(item.datasetScene || "").trim()).filter((item) => item && item !== "none"))).sort()],
    [recordRows],
  );
  useEffect(() => {
    if (!recordModelOptions.includes(recordModelFilter)) {
      setRecordModelFilter("all");
    }
  }, [recordModelFilter, recordModelOptions]);
  useEffect(() => {
    if (!recordDeploymentOptions.includes(recordDeploymentFilter)) {
      setRecordDeploymentFilter("all");
    }
  }, [recordDeploymentFilter, recordDeploymentOptions]);
  useEffect(() => {
    if (!recordObjectiveOptions.includes(recordObjectiveFilter)) {
      setRecordObjectiveFilter("all");
    }
  }, [recordObjectiveFilter, recordObjectiveOptions]);
  useEffect(() => {
    if (!recordTaskSceneOptions.includes(recordTaskSceneFilter)) {
      setRecordTaskSceneFilter("all");
    }
  }, [recordTaskSceneFilter, recordTaskSceneOptions]);
  useEffect(() => {
    if (!recordDatasetSceneOptions.includes(recordDatasetSceneFilter)) {
      setRecordDatasetSceneFilter("all");
    }
  }, [recordDatasetSceneFilter, recordDatasetSceneOptions]);

  const filteredRecordRows = useMemo(() => {
    return recordRows.filter((row) => {
      if (recordModelFilter !== "all" && row.modelId !== recordModelFilter) {
        return false;
      }
      if (recordDeploymentFilter !== "all" && row.deploymentId !== recordDeploymentFilter) {
        return false;
      }
      if (recordTypeFilter !== "all" && row.recordType !== recordTypeFilter) {
        return false;
      }
      if (recordObjectiveFilter !== "all" && row.objective !== recordObjectiveFilter) {
        return false;
      }
      if (recordTaskSceneFilter === "none" && row.taskScene !== "none") {
        return false;
      }
      if (recordTaskSceneFilter !== "all" && recordTaskSceneFilter !== "none" && row.taskScene !== recordTaskSceneFilter) {
        return false;
      }
      if (recordDatasetSceneFilter === "none" && row.datasetScene !== "none") {
        return false;
      }
      if (recordDatasetSceneFilter !== "all" && recordDatasetSceneFilter !== "none" && row.datasetScene !== recordDatasetSceneFilter) {
        return false;
      }
      return true;
    });
  }, [recordDatasetSceneFilter, recordDeploymentFilter, recordModelFilter, recordObjectiveFilter, recordRows, recordTaskSceneFilter, recordTypeFilter]);

  const pagination = useTablePagination(filteredRows, 5);
  const recordPagination = useTablePagination(filteredRecordRows, 5);
  const noTaskRoutingCount = rows.filter((item) => !item.hasTaskRouting).length;
  const noExperimentCount = rows.filter((item) => !item.hasExperiment).length;

  return (
    <>
      <section className="page-shell card section-shell score-evolution-section">
        <SectionHeader
          kicker="全量总表"
          title="Deployment 表现变化总览"
          note="这页统一看所有 Deployment 在任务调度与推理实验两条链路中的表现层变化。这里的评分对象是 deployment 实例；即使某个 Deployment 从未参与过任务调度或推理实验，也会保留在总表里。"
        />
        <div className="metrics">
          <MetricCard title="Deployment 总数" value={rows.length} note="包含当前平台全部 deployment" />
          <MetricCard title="无任务调度记录" value={noTaskRoutingCount} note="没有进入任务调度评分快照的 deployment" />
          <MetricCard title="无推理实验记录" value={noExperimentCount} note="没有推理实验归档的 deployment" />
          <MetricCard title="当前筛选结果" value={filteredRows.length} note="经过筛选后仍保留的 deployment 数量" />
        </div>
        <div className="form-grid">
          <label className="stack-field">
            <span>模型</span>
            <select onChange={(event) => setModelFilter(event.target.value)} value={modelFilter}>
              <option value="all">全部模型</option>
              {modelOptions.filter((item) => item !== "all").map((item) => (
                <option key={`deployment-performance-model-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>Deployment</span>
            <select onChange={(event) => setDeploymentFilter(event.target.value)} value={deploymentFilter}>
              <option value="all">全部 Deployment</option>
              {deploymentOptions.filter((item) => item !== "all").map((item) => (
                <option key={`deployment-performance-deployment-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>Deployment 状态</span>
            <select onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
              <option value="all">全部状态</option>
              {statusOptions.filter((item) => item !== "all").map((item) => (
                <option key={`deployment-performance-status-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>任务调度场景</span>
            <select onChange={(event) => setTaskSceneFilter(event.target.value)} value={taskSceneFilter}>
              <option value="all">全部场景</option>
              <option value="none">无任务调度记录</option>
              {taskSceneOptions.filter((item) => !["all", "none"].includes(item)).map((item) => (
                <option key={`deployment-performance-task-scene-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>推理实验数据集场景</span>
            <select onChange={(event) => setDatasetSceneFilter(event.target.value)} value={datasetSceneFilter}>
              <option value="all">全部场景</option>
              <option value="none">无推理实验记录</option>
              {datasetSceneOptions.filter((item) => !["all", "none"].includes(item)).map((item) => (
                <option key={`deployment-performance-dataset-scene-${item}`} value={item}>
                  {formatDatasetSceneLabel(item)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="banner banner-muted">
          当前过滤策略保留四层：模型、Deployment 状态、任务调度场景、推理实验数据集场景。这样既能筛出某个 Deployment 在哪些调度场景下发生变化，也能对照看到推理实验记录。
        </div>
      </section>

      <section className="page-shell card section-shell score-evolution-section">
        <SectionHeader
          kicker="表现层"
          title="所有 Deployment 表现变化"
          note="表内同时展示任务调度表现概览、表现层拆解、表现层净变化，以及造成这些变化的任务调度记录与推理记录。这里每一行对应一个 deployment 实例；变化来源记录默认各展示最近 3 条。"
        />
        <div className="table-wrap">
          <table className="weight-profile-table score-evolution-table score-evolution-primary-table deployment-performance-table">
            <thead>
              <tr>
                <th>Deployment / 模型</th>
                <th>场景覆盖</th>
                <th>任务调度表现概览</th>
                <th>当前表现层拆解</th>
                <th>表现层净变化</th>
                <th>变化来源记录</th>
              </tr>
            </thead>
            <tbody>
              {pagination.pagedItems.length ? pagination.pagedItems.map((row) => {
                const deltaPositive = Number(row.avgScoreDelta || 0) > 0;
                const deltaNegative = Number(row.avgScoreDelta || 0) < 0;
                const recentTaskRoutingArchives = row.matchedTaskRoutingArchives.slice(0, 3);
                const recentExperimentArchives = sortByCreatedDesc(row.experimentRows).slice(0, 3);
                return (
                  <tr key={row.deployment.deployment_id}>
                    <td className="score-evolution-cell-identity">
                      <div className="archive-cell-stack">
                        <strong className="archive-main">{row.deployment.deployment_id}</strong>
                        <div className="table-note">{row.deployment.status || "-"}</div>
                        <div className="archive-main">{row.deployment.model_id}</div>
                        <div className="table-note">{row.deployment.profile_name || "-"}</div>
                      </div>
                    </td>
                    <td>
                      <div className="archive-cell-stack">
                        <div className="table-note">任务调度</div>
                        {row.taskScenes.length ? (
                          <div className="tag-list">
                            {row.taskScenes.map((item) => <span className="tag" key={`${row.deployment.deployment_id}-task-scene-${item}`}>{item}</span>)}
                          </div>
                        ) : (
                          <span className="table-note">无任务调度记录</span>
                        )}
                        <div className="table-note">目标：{row.taskObjectiveLabels.length ? row.taskObjectiveLabels.join(" / ") : "-"}</div>
                        <div className="table-note">推理实验</div>
                        {row.datasetScenes.length ? (
                          <div className="tag-list">
                            {row.datasetScenes.map((item) => <span className="tag" key={`${row.deployment.deployment_id}-dataset-scene-${item}`}>{formatDatasetSceneLabel(item)}</span>)}
                          </div>
                        ) : (
                          <span className="table-note">无推理实验记录</span>
                        )}
                      </div>
                    </td>
                    <td>
                      {row.hasTaskRouting ? (
                        <div className="score-overview-stack">
                          <div className="score-overview-row">
                            <span className="weight-snapshot-chip weight-snapshot-chip-baseline score-overview-chip">
                              <strong>优化前</strong>
                              <span>{Number(row.avgBaselineScore || 0).toFixed(4)}</span>
                            </span>
                            <span className="weight-snapshot-chip score-overview-chip">
                              <strong>当前</strong>
                              <span>{Number(row.avgCurrentScore || 0).toFixed(4)}</span>
                            </span>
                          </div>
                          <span className={`score-delta-chip score-overview-delta ${deltaPositive ? "score-delta-positive" : deltaNegative ? "score-delta-negative" : ""}`}>
                            变化 {formatSignedNumber(row.avgScoreDelta || 0, 4)}
                          </span>
                        </div>
                      ) : (
                        <span className="table-note">无任务调度评分快照</span>
                      )}
                    </td>
                    <td>
                      <div className="score-breakdown-compact-grid score-breakdown-compact-grid-wide">
                        {getWeightSnapshotEntries(row.avgCurrentBreakdown || {}).length ? (
                          getWeightSnapshotEntries(row.avgCurrentBreakdown || {}).map(([key, value]) => (
                            <span className="score-breakdown-compact-chip" key={`${row.deployment.deployment_id}-current-breakdown-${key}`}>
                              <strong>{formatWeightDimensionLabel(key)}</strong>
                              <span>{Number(value).toFixed(2)}</span>
                            </span>
                          ))
                        ) : (
                          <span className="table-note">无</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <div className="score-breakdown-compact-grid score-breakdown-compact-grid-wide">
                        {getSignedWeightSnapshotEntries(row.avgPerformanceDeltaBreakdown || {}).length ? (
                          getSignedWeightSnapshotEntries(row.avgPerformanceDeltaBreakdown || {}).map(([key, value]) => (
                            <span className={`score-breakdown-compact-chip ${Number(value || 0) > 0 ? "score-adjustment-breakdown-positive" : Number(value || 0) < 0 ? "score-adjustment-breakdown-negative" : ""}`} key={`${row.deployment.deployment_id}-performance-delta-${key}`}>
                              <strong>{formatWeightDimensionLabel(key)}</strong>
                              <span>{formatSignedNumber(value, 4)}</span>
                            </span>
                          ))
                        ) : (
                          <span className="table-note">无</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <div className="archive-cell-stack">
                        <div className="archive-main">任务调度记录 {row.matchedTaskRoutingArchives.length}</div>
                        {recentTaskRoutingArchives.length ? recentTaskRoutingArchives.map((item) => (
                          <div className="table-note" key={`${row.deployment.deployment_id}-task-archive-${item.archive_id || item.task_routing_decision_id}`}>
                            {item.task_routing_decision_id || item.archive_id || "-"} · {item.created_at || "-"} · {item.scene || "-"}
                          </div>
                        )) : (
                          <div className="table-note">未命中任务调度归档</div>
                        )}
                        <div className="table-note">默认展示最近 3 条任务调度记录</div>
                        <div className="archive-main">推理记录 {row.experimentRows.length}</div>
                        {recentExperimentArchives.length ? recentExperimentArchives.map((item) => (
                          <div className="table-note" key={`${row.deployment.deployment_id}-experiment-archive-${item.archive_id || item.created_at}`}>
                            {item.archive_id || "-"} · {formatDatasetSceneLabel(item.scene)} · {item.created_at || "-"}
                          </div>
                        )) : (
                          <div className="table-note">未命中推理实验归档</div>
                        )}
                        <div className="table-note">默认展示最近 3 条推理记录</div>
                        <div className="table-note">平均延迟 {row.avgLatencyMs !== null ? `${Number(row.avgLatencyMs).toFixed(2)} ms` : "-"}</div>
                        <div className="table-note">平均三元组 {row.avgTripleCount !== null ? Number(row.avgTripleCount).toFixed(2) : "-"}</div>
                        <div className="table-note">最近实验 {row.latestExperimentAt}</div>
                        <div className="table-note">
                          {!row.hasTaskRouting && !row.hasExperiment
                            ? "无任务调度、无推理实验"
                            : !row.hasTaskRouting
                              ? "仅有推理实验记录"
                              : !row.hasExperiment
                                ? "仅有任务调度记录"
                                : "两条链路均有记录"}
                        </div>
                      </div>
                    </td>
                  </tr>
                );
              }) : (
                <tr>
                  <td className="table-note" colSpan="6">当前筛选条件下没有可展示的 Deployment 表现变化记录。</td>
                </tr>
              )}
            </tbody>
          </table>
          <PaginationBar
            currentPage={pagination.page}
            totalPages={pagination.totalPages}
            pageSize={pagination.pageSize}
            totalItems={pagination.totalItems}
            onChange={pagination.setPage}
          />
        </div>
      </section>

      <section className="page-shell card section-shell score-evolution-section">
        <SectionHeader
          kicker="记录级"
          title="Deployment 记录级影响明细"
          note="这张表一行就是一条真实记录。筛选顺序改成先选模型，再选这个模型下的 Deployment，用来更直接定位某个 Deployment 的表现变化到底由哪条任务调度记录或推理记录带来。"
        />
        <div className="form-grid">
          <label className="stack-field">
            <span>模型</span>
            <select onChange={(event) => setRecordModelFilter(event.target.value)} value={recordModelFilter}>
              <option value="all">全部模型</option>
              {recordModelOptions.filter((item) => item !== "all").map((item) => (
                <option key={`deployment-records-model-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>Deployment</span>
            <select onChange={(event) => setRecordDeploymentFilter(event.target.value)} value={recordDeploymentFilter}>
              <option value="all">全部 Deployment</option>
              {recordDeploymentOptions.filter((item) => item !== "all").map((item) => (
                <option key={`deployment-records-deployment-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <small className="detail-note">这里会跟随上一个“模型”筛选，只展示该模型下的 Deployment。</small>
          </label>
          <label className="stack-field">
            <span>记录类型</span>
            <select onChange={(event) => setRecordTypeFilter(event.target.value)} value={recordTypeFilter}>
              <option value="all">全部类型</option>
              <option value="task_routing">任务调度</option>
              <option value="experiment">推理实验</option>
            </select>
          </label>
          <label className="stack-field">
            <span>调度目标</span>
            <select onChange={(event) => setRecordObjectiveFilter(event.target.value)} value={recordObjectiveFilter}>
              <option value="all">全部目标</option>
              {recordObjectiveOptions.filter((item) => item !== "all").map((item) => (
                <option key={`deployment-records-objective-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>任务调度场景</span>
            <select onChange={(event) => setRecordTaskSceneFilter(event.target.value)} value={recordTaskSceneFilter}>
              <option value="all">全部场景</option>
              <option value="none">非任务调度记录</option>
              {recordTaskSceneOptions.filter((item) => !["all", "none"].includes(item)).map((item) => (
                <option key={`deployment-records-task-scene-${item}`} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>推理实验数据集场景</span>
            <select onChange={(event) => setRecordDatasetSceneFilter(event.target.value)} value={recordDatasetSceneFilter}>
              <option value="all">全部场景</option>
              <option value="none">非推理实验记录</option>
              {recordDatasetSceneOptions.filter((item) => !["all", "none"].includes(item)).map((item) => (
                <option key={`deployment-records-dataset-scene-${item}`} value={item}>
                  {formatDatasetSceneLabel(item)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="banner banner-muted">
          这张表看什么：如果你想知道“为什么某个 Deployment 最近表现变了”，就看这里。任务调度记录会带出它所属的评分上下文和净变化；推理实验记录会带出反馈信号、延迟和三元组信息。
        </div>
        <div className="table-wrap">
          <table className="weight-profile-table score-evolution-table deployment-record-table">
            <thead>
              <tr>
                <th>记录类型 / 编号</th>
                <th>Deployment / 模型</th>
                <th>目标 / 场景</th>
                <th>记录内容</th>
                <th>表现层净变化</th>
                <th>命中说明</th>
              </tr>
            </thead>
            <tbody>
              {recordPagination.pagedItems.length ? recordPagination.pagedItems.map((row) => (
                <tr key={`${row.recordType}-${row.deploymentId}-${row.recordId}-${row.createdAt}`}>
                  <td>
                    <div className="archive-cell-stack">
                      <div className="archive-main">{row.sourceLabel}</div>
                      <div className="table-note">{row.recordId}</div>
                      <div className="table-note">{row.createdAt}</div>
                    </div>
                  </td>
                  <td>
                    <div className="archive-cell-stack">
                      <div className="archive-main">{row.deploymentId}</div>
                      <div className="table-note">{row.status || "-"}</div>
                      <div>{row.modelId}</div>
                      <div className="table-note">{row.profileName || "-"}</div>
                    </div>
                  </td>
                  <td>
                    <div className="archive-cell-stack">
                      <div className="table-note">目标：{row.objective === "all" ? "-" : row.objective}</div>
                      <div className="table-note">任务调度场景：{row.taskScene === "none" ? "-" : row.taskScene}</div>
                      <div className="table-note">数据集场景：{row.datasetScene === "none" ? "-" : formatDatasetSceneLabel(row.datasetScene)}</div>
                    </div>
                  </td>
                  <td>
                    <div className="archive-cell-stack">
                      <div>{row.summaryLabel}</div>
                      <div className="tag-list">
                        {row.impactSignals.map((item) => (
                          <span className="tag" key={`${row.recordType}-${row.recordId}-${item}`}>{item}</span>
                        ))}
                      </div>
                      <div className="table-note">延迟：{row.latencyMs !== null && row.latencyMs !== undefined ? `${Number(row.latencyMs).toFixed(2)} ms` : "-"}</div>
                      <div className="table-note">三元组：{row.tripleCount !== null && row.tripleCount !== undefined ? row.tripleCount : "-"}</div>
                    </div>
                  </td>
                  <td>
                    {row.recordType === "task_routing" ? (
                      <div className="archive-cell-stack">
                        <div className={`score-delta-chip ${Number(row.matchedPerformanceScoreDelta || 0) > 0 ? "score-delta-positive" : Number(row.matchedPerformanceScoreDelta || 0) < 0 ? "score-delta-negative" : ""}`}>
                          净变化 {formatSignedNumber(row.matchedPerformanceScoreDelta || 0, 4)}
                        </div>
                        <div className="table-note">
                          评分变化 {formatSignedNumber(row.matchedScoreDelta || 0, 4)}
                        </div>
                        <div className="score-breakdown-compact-grid score-breakdown-compact-grid-wide">
                          {getSignedWeightSnapshotEntries(row.matchedPerformanceDeltaBreakdown || {}).length ? (
                            getSignedWeightSnapshotEntries(row.matchedPerformanceDeltaBreakdown || {}).map(([key, value]) => (
                              <span className={`score-breakdown-compact-chip ${Number(value || 0) > 0 ? "score-adjustment-breakdown-positive" : Number(value || 0) < 0 ? "score-adjustment-breakdown-negative" : ""}`} key={`${row.recordType}-${row.recordId}-impact-${key}`}>
                                <strong>{formatWeightDimensionLabel(key)}</strong>
                                <span>{formatSignedNumber(value, 4)}</span>
                              </span>
                            ))
                          ) : (
                            <span className="table-note">未找到更细的表现变化拆解</span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="archive-cell-stack">
                        <div className="table-note">推理实验记录通常不直接沉淀为单条“表现层净变化”数值，而是通过反馈、延迟和三元组统计逐步作用到 Deployment 表现层。</div>
                        <div className="tag-list">
                          {row.impactSignals.map((item) => (
                            <span className="tag" key={`${row.recordType}-${row.recordId}-signal-${item}`}>{item}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </td>
                  <td className="limitation-cell">{row.note}</td>
                </tr>
              )) : (
                <tr>
                  <td className="table-note" colSpan="6">当前筛选条件下没有可展示的记录级影响明细。</td>
                </tr>
              )}
            </tbody>
          </table>
          <PaginationBar
            currentPage={recordPagination.page}
            totalPages={recordPagination.totalPages}
            pageSize={recordPagination.pageSize}
            totalItems={recordPagination.totalItems}
            onChange={recordPagination.setPage}
          />
        </div>
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
  const visiblePolicyDevices = (deploymentPolicy?.gpu_snapshot?.devices || []).filter((item) => item.platform_enabled !== false);
  const freestGpu = visiblePolicyDevices.length
    ? [...visiblePolicyDevices].sort(
        (left, right) =>
          ((right.memory_total_mb || 0) - (right.memory_used_mb || 0)) -
          ((left.memory_total_mb || 0) - (left.memory_used_mb || 0)),
      )[0]
    : null;
  const allowedGpuIds = deploymentPolicy?.gpu_snapshot?.platform_allowed_gpu_ids || [];
  const disabledGpuIds = deploymentPolicy?.gpu_snapshot?.platform_disabled_gpu_ids || [];

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
      const isRecoveryToGpu = !deployment.use_gpu && recommendation.recommended_use_gpu;
      const isConstraintToCpu =
        deployment.use_gpu && !recommendation.recommended_use_gpu && recommendation.gpu_pressure_action === "migrate_to_cpu";
      const isConstraintGpuRebalance =
        deployment.use_gpu && recommendation.recommended_use_gpu && gpuChanged;
      const isRecoveryProfile =
        profileChanged &&
        (
          (!deployment.use_gpu && !recommendation.recommended_use_gpu) ||
          (deployment.use_gpu && recommendation.recommended_use_gpu)
        );
      let optimizationAction = "保持现状";
      if (isRecoveryToGpu) {
        optimizationAction = "资源恢复优化：回迁到 GPU";
      } else if (isConstraintToCpu) {
        optimizationAction = "资源受限优化：迁移到 CPU";
      } else if (isConstraintGpuRebalance) {
        optimizationAction = "资源受限优化：换卡重平衡";
      } else if (isRecoveryProfile) {
        optimizationAction = "资源恢复优化：调整 Profile";
      }
      return {
        deployment,
        recommendation,
        currentGpuLabel,
        targetGpuLabel,
        optimizationAction,
        profileChanged,
        gpuChanged,
        isRecoveryToGpu,
        isConstraintToCpu,
        isConstraintGpuRebalance,
        isRecoveryProfile,
      };
    })
    .filter(Boolean);
  const resourceRecoveryCandidateCount = deploymentPlanRows.filter(
    (row) => row.isRecoveryToGpu || row.isRecoveryProfile,
  ).length;
  const policyActionPagination = useTablePagination(deploymentPolicyAutomation?.recent_actions || [], 5);
  const deploymentPlanPagination = useTablePagination(deploymentPlanRows, 5);

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="创新点二"
          title="跨层协同部署策略"
          note="在基础热度、冷启动风险和 GPU 压力之外，引入任务调度价值信号与部署优化历史信号，生成跨层协同的部署推荐。"
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
            "这里展示的不是简单“热门上好卡”的规则，而是把任务调度价值、维度贡献、显存重启历史和 CPU 回退历史一起纳入的跨层协同部署策略。"}
        </div>
        <div className="detail-note">
          当前部署策略只会在平台允许范围内推荐 GPU。
          当前可用 GPU：{allowedGpuIds.length ? allowedGpuIds.join(", ") : "仅 CPU"}；
          已禁用 GPU：{disabledGpuIds.length ? disabledGpuIds.join(", ") : "无"}。
          若在运行资源页一次性调整多张卡，这里会在下一次分析时直接反映最新范围。
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
          部署优化现在是双向闭环。
          资源受限时，优先做 GPU 内部换卡与重平衡，必要时将低优先级 deployment 迁移到 CPU；
          资源恢复后，会重新基于最新评分生成推荐，再执行 GPU 回迁与 profile 恢复优化，而不是回到历史旧状态。
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
                {policyActionPagination.pagedItems.map((item, index) => (
                  <tr key={`${item.created_at}-${item.model_id}-${index}`}>
                    <td>{item.created_at}</td>
                    <td>{formatDeploymentPolicyActionLabel(item.action_type)}</td>
                    <td>
                      <strong>{item.model_id}</strong>
                      <div className="table-note">{item.deployment_id || "-"}</div>
                      <div className="table-note">{item.action_type}</div>
                    </td>
                    <td>{item.status}</td>
                    <td className="limitation-cell">{item.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PaginationBar
              currentPage={policyActionPagination.page}
              totalPages={policyActionPagination.totalPages}
              pageSize={policyActionPagination.pageSize}
              totalItems={policyActionPagination.totalItems}
              onChange={policyActionPagination.setPage}
            />
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
            <MetricCard title="平台可用 GPU" value={allowedGpuIds.length} note={allowedGpuIds.length ? `GPU ${allowedGpuIds.join(", ")}` : "当前仅允许 CPU"} />
            <MetricCard title="活跃进程" value={deploymentPolicy.gpu_snapshot.process_count || 0} note="nvidia-smi 检测到的活跃进程" />
            <MetricCard title="最空闲 GPU" value={freestGpu ? `GPU ${freestGpu.index}` : "-"} note="按显存余量估算" />
            <MetricCard title="CPU 迁移候选" value={gpuPressureModels.length} note="全局 GPU 高压时建议迁移到 CPU 的模型数" />
            <MetricCard title="恢复优化候选" value={resourceRecoveryCandidateCount} note="资源恢复后可回迁 GPU 或恢复 profile 的 deployment 数" />
          </div>
        ) : (
          <EmptyState icon="G" title="还没有部署建议快照" note="点击上方“刷新部署建议”后，这里会展示实时 GPU 资源视图。" />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="分层概览" title="热温冷模型分层" note="这一层不是只看调用热度，还综合考虑任务调度价值和运行风险，形成更稳健的热/温/冷分层。" />
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
        <SectionHeader kicker="策略建议" title="现有 Deployment 优化方案表" note="每一行对应平台里已经存在的 deployment，联合展示基础需求、协同优先级、调度价值、运行风险与下一步优化动作。" />
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
                {deploymentPlanPagination.pagedItems.map((row) => (
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
                      <div className="table-note">需求等级：{row.recommendation.demand_level} / 协同优先级：{formatFixed(row.recommendation.collaborative_priority_score)}</div>
                      <div className="table-note">基础需求分：{formatFixed(row.recommendation.base_demand_score)} / 调度价值：{formatFixed(row.recommendation.routing_value_score)} / 运行风险：{formatFixed(row.recommendation.operational_risk_score)}</div>
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
                      <div className="table-note">
                        调度选中：{row.recommendation.routing_selected_count} / 采纳：{row.recommendation.routing_adopted_count} / 反馈：{row.recommendation.routing_feedback_count}
                      </div>
                    </td>
                    <td>
                      <div>{row.optimizationAction}</div>
                      <div className="table-note">
                        {row.gpuChanged ? `设备调整为 ${row.targetGpuLabel}` : "设备保持不变"}
                      </div>
                      <div className="table-note">
                        {row.profileChanged ? `Profile 调整为 ${row.recommendation.recommended_profile_name}` : "Profile 保持不变"}
                      </div>
                      <div className="table-note">
                        {row.isRecoveryToGpu || row.isRecoveryProfile
                          ? "这是资源恢复后的最新推荐动作"
                          : row.isConstraintToCpu || row.isConstraintGpuRebalance
                            ? "这是资源受限时的压缩或重平衡动作"
                            : "当前 deployment 已接近最新推荐方案"}
                      </div>
                      <div className="tag-list">
                        {(row.recommendation.policy_tags || []).map((tag) => (
                          <span className="tag" key={`${row.deployment.deployment_id}-${tag}`}>{tag}</span>
                        ))}
                      </div>
                    </td>
                    <td className="limitation-cell">
                      {row.recommendation.rationale}
                      <div className="table-note">{row.recommendation.coordination_note || "当前推荐综合了调度价值与运行风险信号。"}</div>
                      <div className="table-note">
                        历史动作：重启 {row.recommendation.restart_history_count} 次 / 换卡 {row.recommendation.rebalance_history_count} 次 / CPU 回退 {row.recommendation.cpu_migration_history_count} 次 / 失败 {row.recommendation.failed_action_history_count} 次
                      </div>
                      <div className="table-note">
                        重点维度：{row.recommendation.routing_focus_dimensions?.length
                          ? row.recommendation.routing_focus_dimensions.map((item) => formatWeightDimensionLabel(item)).join(" / ")
                          : "暂无显著偏置"}
                      </div>
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
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PaginationBar
              currentPage={deploymentPlanPagination.page}
              totalPages={deploymentPlanPagination.totalPages}
              pageSize={deploymentPlanPagination.pageSize}
              totalItems={deploymentPlanPagination.totalItems}
              onChange={deploymentPlanPagination.setPage}
            />
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
  const canFeedbackArchives = currentPermissions.includes("predictions.feedback");
  const archivePageSize = 5;
  const [predictionArchivePage, setPredictionArchivePage] = useState(1);
  const [taskRoutingArchivePage, setTaskRoutingArchivePage] = useState(1);
  const allArchives = sortByCreatedDesc(derived.predictArchives);
  const feedbackOverview = derived.taskRoutingFeedbackOverview || { recent_decisions: [] };
  const predictionArchives = allArchives.filter((item) => item.archive_source !== "task_routing");
  const taskRoutingArchives = allArchives.filter((item) => item.archive_source === "task_routing");
  const taskRoutingDecisionIds = new Set(
    taskRoutingArchives.map((item) => String(item.task_routing_decision_id || "").trim()).filter(Boolean),
  );
  const supplementalTaskRoutingDecisionRows = (feedbackOverview.recent_decisions || [])
    .filter((item) => {
      const decisionId = String(item?.decision_id || "").trim();
      return decisionId && !taskRoutingDecisionIds.has(decisionId);
    })
    .map((item) => ({
      archive_id: `decision-${item.decision_id}`,
      deployment_id: item.decision_id,
      model_id: "task-routing",
      profile_name: item.objective || "task-routing",
      archive_source: "task_routing_decision",
      service_url: null,
      scene: item.context_domain || null,
      request_text: null,
      request_texts: [],
      options: {},
      latency_ms: null,
      triple_count: item.fused_triple_count ?? null,
      item_count: null,
      response_payload: {},
      task_routing_decision_id: item.decision_id,
      task_routing_objective: item.objective || null,
      task_routing_execution_path: item.execution_path || null,
      task_routing_fusion_policy: item.fusion_policy || null,
      task_routing_weight_snapshot: {},
      task_routing_adjustment_snapshot: {},
      task_routing_selected_model_ids: item.selected_model_ids || [],
      feedback_status: item.accepted === true ? "satisfied" : item.accepted === false ? "unsatisfied" : "pending",
      feedback_note: null,
      feedback_updated_at: item.feedback_status === "completed" ? item.created_at : null,
      created_at: item.created_at,
      is_supplemental_decision: true,
      inferred_language: item.inferred_language || null,
    }));
  const mergedTaskRoutingArchives = sortByCreatedDesc([...taskRoutingArchives, ...supplementalTaskRoutingDecisionRows]);
  const predictionArchiveTotalPages = Math.max(1, Math.ceil(predictionArchives.length / archivePageSize));
  const taskRoutingArchiveTotalPages = Math.max(1, Math.ceil(mergedTaskRoutingArchives.length / archivePageSize));
  const safePredictionArchivePage = Math.min(predictionArchivePage, predictionArchiveTotalPages);
  const safeTaskRoutingArchivePage = Math.min(taskRoutingArchivePage, taskRoutingArchiveTotalPages);
  useEffect(() => {
    if (predictionArchivePage !== safePredictionArchivePage) {
      setPredictionArchivePage(safePredictionArchivePage);
    }
  }, [predictionArchivePage, safePredictionArchivePage]);
  useEffect(() => {
    if (taskRoutingArchivePage !== safeTaskRoutingArchivePage) {
      setTaskRoutingArchivePage(safeTaskRoutingArchivePage);
    }
  }, [taskRoutingArchivePage, safeTaskRoutingArchivePage]);
  const pagedPredictionArchives = predictionArchives.slice(
    (safePredictionArchivePage - 1) * archivePageSize,
    safePredictionArchivePage * archivePageSize,
  );
  const pagedTaskRoutingArchives = taskRoutingArchives.slice(
    (safeTaskRoutingArchivePage - 1) * archivePageSize,
    safeTaskRoutingArchivePage * archivePageSize,
  );
  const pagedMergedTaskRoutingArchives = mergedTaskRoutingArchives.slice(
    (safeTaskRoutingArchivePage - 1) * archivePageSize,
    safeTaskRoutingArchivePage * archivePageSize,
  );
  const archiveSatisfiedCount = allArchives.filter((item) => normalizePredictionFeedbackStatus(item.feedback_status) === "satisfied").length;
  const archiveUnsatisfiedCount = allArchives.filter((item) => normalizePredictionFeedbackStatus(item.feedback_status) === "unsatisfied").length;
  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader kicker="结果反馈" title="结果反馈" note="这里以用户反馈为主组织推理与调度结果，用来回看在线推理结果、补充调度学习证据，并支持论文展示。" />
        <div className="metrics">
          <MetricCard title="最新反馈记录" value={derived.latestPredictArchive?.archive_id || "-"} note={derived.latestPredictArchive?.model_id || "-"} />
          <MetricCard title="反馈记录总数" value={allArchives.length} note="在线推理与任务调度结果自动留痕" />
          <MetricCard title="推理实验反馈" value={predictionArchives.length} note="以 pred 开头，来自在线推理或推理实验" />
          <MetricCard title="任务调度反馈" value={mergedTaskRoutingArchives.length} note="真实记录 + 历史最近决策补充行" />
          <MetricCard title="满意结果" value={archiveSatisfiedCount} note="用户确认当前结果可接受" />
          <MetricCard title="不满意结果" value={archiveUnsatisfiedCount} note="用户明确标记结果不符合预期" />
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="任务调度反馈" title="任务调度反馈列表" note="这一表优先展示真实任务调度记录，同时补充接入历史最近决策记录，便于在同一处查看完整任务调度留痕和反馈。" />
        <div className="banner banner-muted">这张表看什么：一行就是一次任务调度记录。已生成真实留痕的记录保留完整输入、权重快照和反馈入口；没有真实留痕的历史最近决策，也会补成只读记录接进来。</div>
        {mergedTaskRoutingArchives.length ? (
          <div className="table-wrap">
            <table className="archive-table">
              <thead>
                <tr>
                  <th>归档信息</th>
                  <th>目标与场景</th>
                  <th>输入文本</th>
                  <th>调度快照</th>
                  <th>反馈</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {pagedMergedTaskRoutingArchives
                  .map((item) => {
                    const routingModels = (item.task_routing_selected_model_ids || []).join(" / ");
                    const weightSnapshotText = formatWeightSnapshot(item.task_routing_weight_snapshot);
                    const adjustmentSnapshotText = formatWeightSnapshot(item.task_routing_adjustment_snapshot);
                    const displayId = item.task_routing_decision_id || item.archive_id;
                    const isSupplementalDecision = Boolean(item.is_supplemental_decision);
                    return (
                    <tr key={item.archive_id}>
                      <td>
                        <div className="archive-cell-stack">
                          <strong className="archive-main">{displayId}</strong>
                          <div className="table-note">创建时间：{item.created_at}</div>
                          <div className="table-note">来源：{isSupplementalDecision ? "历史最近决策补充" : "任务调度反馈记录"}</div>
                        </div>
                      </td>
                      <td>
                        <div className="archive-cell-stack">
                          <div className="archive-main">{item.task_routing_objective || item.profile_name || "-"}</div>
                          <div className="table-note">执行路径：{formatExecutionPathLabel(item.task_routing_execution_path)}</div>
                          <div className="table-note">融合策略：{formatFusionPolicyLabel(item.task_routing_fusion_policy)}</div>
                          <div className="table-note">场景：{item.scene || "-"}</div>
                          <div className="table-note">语言：{item.inferred_language || "-"}</div>
                          <div className="table-note">参与模型：{routingModels || "-"}</div>
                        </div>
                      </td>
                      <td>
                        <div className="archive-input">
                          {item.request_text || item.request_texts?.[0] || "该历史任务调度记录未保留原始输入文本。"}
                        </div>
                      </td>
                      <td>
                        <div className="archive-cell-stack">
                          <div className="archive-result-grid">
                            <span className="archive-result-chip">延迟 {escapeValue(item.latency_ms)} ms</span>
                            <span className="archive-result-chip">三元组 {escapeValue(item.triple_count)}</span>
                          </div>
                          <div className="table-note">权重快照：{weightSnapshotText}</div>
                          <div className="table-note">调整快照：{adjustmentSnapshotText}</div>
                          <div className="table-note">
                            {isSupplementalDecision ? "该条为最近决策补充记录，优先用于补足历史任务调度留痕。" : "这条记录的反馈会同步给任务调度学习逻辑。"}
                          </div>
                        </div>
                      </td>
                      <td className="archive-feedback-cell">
                        <div className="archive-cell-stack">
                          <div className="archive-main">{formatPredictionFeedbackSummary(item.feedback_status, item.feedback_note)}</div>
                          <div className="table-note">{item.feedback_updated_at || "尚未更新反馈时间"}</div>
                          {item.feedback_note ? <div className="table-note">原因：{formatPredictionFeedbackReasonSummary(item.feedback_note)}</div> : null}
                        </div>
                        {!isSupplementalDecision ? (
                          <PredictionArchiveFeedbackControls
                            actionBusy={actionBusy}
                            archive={item}
                            canFeedback={canFeedbackArchives}
                            onUpdate={onUpdatePredictionArchiveFeedback}
                          />
                        ) : (
                          <div className="table-note">历史补充行当前仅展示决策信息，不提供记录级反馈按钮。</div>
                        )}
                      </td>
                      <td>
                        <div className="actions row-actions">
                          {canExportArchives && !isSupplementalDecision ? (
                            <button onClick={() => onExportPredictionArchive(item)} type="button">导出</button>
                          ) : null}
                          {canManageArchives && !isSupplementalDecision ? (
                            <button onClick={() => void onDeletePredictionArchive(item.archive_id)} type="button">删除</button>
                          ) : null}
                          {isSupplementalDecision ? <span className="table-note">历史补充行</span> : null}
                        </div>
                      </td>
                    </tr>
                  )})}
              </tbody>
            </table>
            <PaginationBar
              currentPage={safeTaskRoutingArchivePage}
              totalPages={taskRoutingArchiveTotalPages}
              pageSize={archivePageSize}
              totalItems={mergedTaskRoutingArchives.length}
              onChange={setTaskRoutingArchivePage}
            />
          </div>
        ) : (
          <EmptyState icon="T" title="暂无任务调度反馈" note="先运行一次任务调度，这里就会出现以 trd 开头的反馈记录。" />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="推理实验反馈" title="推理实验反馈列表" note="这一表只放以 pred 开头的记录，主要回看单次在线推理或推理实验结果，也支持直接补反馈。" />
        <div className="banner banner-muted">这张表看什么：一行就是一次推理实验记录。它保留输入文本、延迟和三元组数量，补上的满意 / 不满意 / 暂定反馈会继续影响任务调度学习过程。</div>
        {predictionArchives.length ? (
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
                {pagedPredictionArchives
                  .map((item) => {
                    return (
                    <tr key={item.archive_id}>
                      <td>
                        <div className="archive-cell-stack">
                          <strong className="archive-main">{item.archive_id}</strong>
                          <div className="table-note">创建时间：{item.created_at}</div>
                          <div className="table-note">来源：推理实验反馈记录</div>
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
                          <div className="table-note">推理实验反馈记录用于回放预测输出、导出 JSON 和补充反馈。</div>
                        </div>
                      </td>
                      <td className="archive-feedback-cell">
                        <div className="archive-cell-stack">
                          <div className="archive-main">{formatPredictionFeedbackSummary(item.feedback_status, item.feedback_note)}</div>
                          <div className="table-note">{item.feedback_updated_at || "尚未更新反馈时间"}</div>
                          {item.feedback_note ? <div className="table-note">原因：{formatPredictionFeedbackReasonSummary(item.feedback_note)}</div> : null}
                        </div>
                        <PredictionArchiveFeedbackControls
                          actionBusy={actionBusy}
                          archive={item}
                          canFeedback={canFeedbackArchives}
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
                  )})}
              </tbody>
            </table>
            <PaginationBar
              currentPage={safePredictionArchivePage}
              totalPages={predictionArchiveTotalPages}
              pageSize={archivePageSize}
              totalItems={predictionArchives.length}
              onChange={setPredictionArchivePage}
            />
          </div>
        ) : (
          <EmptyState icon="A" title="暂无推理实验反馈" note="先执行一次在线推理或推理实验，这里就会出现以 pred 开头的反馈记录。" />
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
  modelOptimizationForm,
  modelTransferItems,
  onDismissModelTransfer,
  onDeleteModelPackage,
  onCreateDeployment,
  onDeploymentFormChange,
  onExportModelPackage,
  onImportModel,
  onModelImportFormChange,
  onModelOptimizationFormChange,
  onOptimizeModel,
}) {
  const [search, setSearch] = useState("");
  const [runtimeFilter, setRuntimeFilter] = useState("all");
  const [originFilter, setOriginFilter] = useState("all");
  const deferredSearch = useDeferredValue(search);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canImportModels = currentPermissions.includes("models.import");
  const canManageDeployments = currentPermissions.includes("deployments.manage");
  const canExportPackages = currentPermissions.includes("archive.export");
  const canManageModels = derived.currentAccessUser?.is_admin;
  const familyOptions = data.importOptions?.families || [];
  const optimizationOptions = data.optimizationOptions?.options || [];
  const runtimeCapabilities = data.runtimeCapabilities;
  const uploadTransferItems = modelTransferItems.filter((item) => item.direction !== "optimize");
  const optimizationTransferItems = modelTransferItems.filter((item) => item.direction === "optimize");
  const selectedFamilyOption = familyOptions.find((item) => item.code === modelImportForm.modelFamily) || familyOptions[0];
  const selectedOptimizationOption =
    optimizationOptions.find((item) => item.code === modelOptimizationForm.optimizationType) || optimizationOptions[0];
  const selectedDeploymentModelSummary = getOptimizationSummaryForModel(deploymentForm.modelId, data.modelDetails);
  const deploymentBackendOptions = getDeploymentBackendOptions(
    runtimeCapabilities,
    deploymentForm.useGpu,
    selectedDeploymentModelSummary.optimizationType,
  );
  const selectedDeploymentBackend =
    deploymentBackendOptions.find((item) => item.value === deploymentForm.providerPreference) || deploymentBackendOptions[0] || null;
  const runtimeCapabilitySummary = useMemo(() => {
    const cudaUsable = isRuntimeProviderUsable(runtimeCapabilities, "CUDAExecutionProvider");
    if (cudaUsable) {
      return "当前环境支持 CUDA EP，平台会显示 CUDA 部署后端。";
    }
    return "当前环境仅确认 CPUExecutionProvider 可用；平台会隐藏不可用的 GPU 执行后端。";
  }, [runtimeCapabilities]);
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
      {canImportModels ? (
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="模型导入"
          title="上传模型制品"
          note="模式A上传原始 ONNX 压缩包后由平台自动补齐标准目录；模式B直接接入已经标准化的 LuckyModels 模型包。"
        />
        <div className="detail-note upload-note">
          导入命名规则：模型家族-语言-数据集[-别名]-版本，例如 <span className="mono">casrel-zh-duie-myexp-v1</span>。
        </div>
        {uploadTransferItems.length ? (
          <div className="transfer-list">
            {uploadTransferItems.map((item) => {
              const transferRunning = item.phase === "running";
              return (
                <div className={`transfer-card transfer-${item.phase}`} key={item.taskId}>
                  <div className="transfer-head">
                    <strong>
                      {item.direction === "download"
                        ? "模型制品下载进度"
                        : "模型制品上传进度"}
                    </strong>
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

      {canManageModels && optimizationOptions.length ? (
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="模型优化"
          title="模型优化工作台"
          note="基于平台内已有 ONNX 制品直接生成优化版本。生成后的制品会作为新的正式模型接入平台，可继续部署、基准测试和删除。"
        />
        <div className="banner banner-muted">
          这块解决的是运行时工具链优化，不是调度策略。当前支持 ORT 图优化、ORT Format、FP16 精度优化和 INT8 动态量化。
          生成后的优化制品会作为新的正式模型接入平台，后续可按普通模型继续部署、基准测试和管理。
        </div>
        {runtimeCapabilities ? (
          <div className="stack-block">
            <div className="section-intro">
              <strong>当前运行时能力看什么</strong>
              <div className="detail-note">
                这里展示服务器上的 onnxruntime 实际可用能力，不是纸面支持。只有这里显示“可正常初始化”的 provider，后续才适合做正式部署入口。
              </div>
            </div>
            <div className="metrics">
              <MetricCard
                title="ONNX Runtime"
                value={runtimeCapabilities.onnxruntime_version || (runtimeCapabilities.onnxruntime_installed ? "已安装" : "未安装")}
                note={runtimeCapabilities.onnxruntime_installed ? "当前平台实际运行版本" : runtimeCapabilities.detail || "当前环境未安装"}
              />
              <MetricCard
                title="ORT Format 导出"
                value={formatBooleanLabel(runtimeCapabilities.features?.ort_format_export)}
                note="支持把 ONNX 制品转成 .ort 格式"
              />
              <MetricCard
                title="Session 配置注入"
                value={formatBooleanLabel(runtimeCapabilities.features?.session_config_entries)}
                note="支持图优化等级、profiling 和 session config entries"
              />
              <MetricCard
                title="探测样本"
                value={runtimeCapabilities.probe_model_path ? "已找到" : "缺失"}
                note={runtimeCapabilities.probe_model_path || "当前没有可用于运行时能力探测的 ONNX 模型"}
              />
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Provider</th>
                    <th>wheel 内置</th>
                    <th>当前环境可用</th>
                    <th>实际激活结果</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  {(runtimeCapabilities.providers || []).map((item) => (
                    <tr key={`runtime-capability-${item.provider}`}>
                      <td><strong>{item.provider}</strong></td>
                      <td>{formatBooleanLabel(item.available_in_wheel)}</td>
                      <td>{formatBooleanLabel(item.runtime_usable)}</td>
                      <td>{Array.isArray(item.active_providers) && item.active_providers.length ? item.active_providers.join(", ") : "-"}</td>
                      <td>{item.detail || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="detail-note">
              {runtimeCapabilitySummary}
              目前平台内已经实际接入的工具链优化优先级建议：先用 ORT 图优化、ORT Format 和 FP16，再根据当前服务器能力选择 CPU 或 CUDA 后端。
            </div>
          </div>
        ) : null}
        <div className="form-grid">
          <label className="stack-field">
            <span>源模型</span>
            <select
              onChange={(event) => onModelOptimizationFormChange((current) => ({ ...current, modelId: event.target.value }))}
              value={modelOptimizationForm.modelId}
            >
              {derived.models.map((model) => (
                <option key={`optimize-${model.model_id}`} value={model.model_id}>
                  {model.model_id} · {model.model_name}
                </option>
              ))}
            </select>
          </label>
          <label className="stack-field">
            <span>优化方式</span>
            <select
              onChange={(event) =>
                onModelOptimizationFormChange((current) => ({ ...current, optimizationType: event.target.value }))
              }
              value={modelOptimizationForm.optimizationType}
            >
              {optimizationOptions.map((item) => (
                <option key={`optimization-${item.code}`} value={item.code}>
                  {item.label}
                </option>
              ))}
            </select>
            <small className="detail-note">{selectedOptimizationOption?.note || "选择一种优化方式后，平台会生成新的模型制品包。"}</small>
          </label>
          <label className="stack-field">
            <span>优化别名</span>
            <input
              onChange={(event) => onModelOptimizationFormChange((current) => ({ ...current, alias: event.target.value }))}
              placeholder={selectedOptimizationOption?.alias_suffix || "例如 fp16"}
              type="text"
              value={modelOptimizationForm.alias}
            />
            <small className="detail-note">留空则自动使用系统别名后缀，生成模型编号时会自动带上源模型信息。</small>
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={actionBusy["optimize-model"] || !modelOptimizationForm.modelId || !modelOptimizationForm.optimizationType}
              onClick={() => void onOptimizeModel()}
              type="button"
            >
              {actionBusy["optimize-model"] ? "生成中" : "生成优化制品"}
            </button>
          </div>
        </div>
        {optimizationTransferItems.length ? (
          <div className="transfer-list">
            {optimizationTransferItems.map((item) => {
              const transferRunning = item.phase === "running";
              return (
                <div className={`transfer-card transfer-${item.phase}`} key={item.taskId}>
                  <div className="transfer-head">
                    <strong>模型制品优化进度</strong>
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
            {selectedDeploymentModelSummary.optimizationLabel ? (
              <small className="detail-note">
                当前所选模型为 {selectedDeploymentModelSummary.optimizationLabel}
                {selectedDeploymentModelSummary.sourceModelId ? `，来源模型 ${selectedDeploymentModelSummary.sourceModelId}` : ""}。
              </small>
            ) : null}
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
            <span>执行后端</span>
            <select
              disabled={!deploymentBackendOptions.length}
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, providerPreference: event.target.value }))}
              value={resolveDeploymentProviderPreference(
                deploymentForm.providerPreference,
                runtimeCapabilities,
                deploymentForm.useGpu,
                selectedDeploymentModelSummary.optimizationType,
              )}
            >
              {deploymentBackendOptions.map((item) => (
                <option key={`model-page-backend-${item.value}`} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <small className="detail-note">
              {selectedDeploymentBackend?.note || "当前环境没有可用的 GPU 执行后端，平台已自动改为 CPU。"}
            </small>
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
              disabled={
                !getDeploymentBackendOptions(runtimeCapabilities, true, selectedDeploymentModelSummary.optimizationType).length
              }
              onChange={(event) =>
                onDeploymentFormChange((current) => ({
                  ...current,
                  useGpu: event.target.checked,
                  providerPreference: resolveDeploymentProviderPreference(
                    current.providerPreference,
                    runtimeCapabilities,
                    event.target.checked,
                    selectedDeploymentModelSummary.optimizationType,
                  ),
                  gpuIdsText: event.target.checked ? current.gpuIdsText : "",
                }))
              }
              type="checkbox"
            />
            使用 GPU
            {!getDeploymentBackendOptions(runtimeCapabilities, true, selectedDeploymentModelSummary.optimizationType).length ? (
              <small className="detail-note">当前服务器环境未通过 CUDA 运行时探测，平台已禁用 GPU 部署入口。</small>
            ) : null}
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
        <SectionHeader kicker="模型库" title="模型资产" note={canManageDeployments ? "模型卡片支持快速查看运行状态，并为可推理模型直接创建部署实例。" : "模型卡片支持查看运行状态、导出制品，并配合上传模型形成模型资产管理入口。"} />
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
                <option value="generated">平台生成</option>
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
            const deploymentPreset = getSuggestedDeploymentPreset(model.model_id, data.modelDetails, recommendation);
            const quickDeployUseGpu = deploymentPreset.useGpu;
            const quickDeployGpuIds = deploymentPreset.gpuIds;
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
                  {detail.package_metadata?.optimization_label ? (
                    <span className="tag">{detail.package_metadata.optimization_label}</span>
                  ) : null}
                  <span className="tag">{manifest.framework || "-"}</span>
                  <span className="tag">{manifest.stage_count || "-" } 个阶段</span>
                </div>
                {detail.package_metadata?.source_model_id ? (
                  <div className="detail-note">
                    来源模型：<span className="mono">{detail.package_metadata.source_model_id}</span>
                    {detail.package_metadata?.optimization_type ? ` · 优化类型：${detail.package_metadata.optimization_type}` : ""}
                  </div>
                ) : null}
                {detail.package_metadata?.optimization_label ? (
                  <div className="detail-note">{deploymentPreset.reason}</div>
                ) : null}
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
                            ? `从模型页快速创建（按部署策略：${recommendation.recommended_device_label || deploymentPreset.deviceLabel}）`
                            : `从模型页快速创建（默认 ${deploymentPreset.deviceLabel}）`,
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
  const capabilityPagination = useTablePagination(filteredRows, 5);

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
                {capabilityPagination.pagedItems.map((item) => (
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
            <PaginationBar
              currentPage={capabilityPagination.page}
              totalPages={capabilityPagination.totalPages}
              pageSize={capabilityPagination.pageSize}
              totalItems={capabilityPagination.totalItems}
              onChange={capabilityPagination.setPage}
            />
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

function ArtifactsPage({ actionBusy, currentUser, data, onDeleteModelPackage, onExportModelManifest, onExportModelProfiles }) {
  const canManageModels = Boolean(currentUser?.is_admin);
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
                {detail.package_metadata?.source_model_id ? (
                  <div className="artifact-meta-row">
                    <span className="artifact-meta-label">source_model</span>
                    <span className="artifact-meta-value mono">{detail.package_metadata.source_model_id}</span>
                  </div>
                ) : null}
                {detail.package_metadata?.optimization_label ? (
                  <div className="artifact-meta-row">
                    <span className="artifact-meta-label">optimization</span>
                    <span className="artifact-meta-value">{detail.package_metadata.optimization_label}</span>
                  </div>
                ) : null}
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
              {detail.summary?.deletable && canManageModels ? (
                <button
                  className="btn-danger"
                  disabled={actionBusy[`delete-model-${detail.summary?.model_id}`]}
                  onClick={() => void onDeleteModelPackage(detail.summary?.model_id)}
                  type="button"
                >
                  {actionBusy[`delete-model-${detail.summary?.model_id}`] ? "删除中" : "删除制品"}
                </button>
              ) : !detail.summary?.deletable ? (
                <button disabled title="本机自带模型不可删除" type="button">本机自带</button>
              ) : (
                <button disabled title="仅管理员可删除用户上传模型" type="button">管理员可删除</button>
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
  data,
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
  const selectedDeploymentModelSummary = getOptimizationSummaryForModel(deploymentForm.modelId, derived.modelDetails);
  const deploymentBackendOptions = getDeploymentBackendOptions(
    data.runtimeCapabilities,
    deploymentForm.useGpu,
    selectedDeploymentModelSummary.optimizationType,
  );
  const selectedDeploymentBackend =
    deploymentBackendOptions.find((item) => item.value === deploymentForm.providerPreference) || deploymentBackendOptions[0] || null;

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
  const deploymentPagination = useTablePagination(filteredDeployments, 5);

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
              {selectedDeploymentModelSummary.optimizationLabel ? (
                <small className="detail-note">
                  当前所选模型为 {selectedDeploymentModelSummary.optimizationLabel}
                  {selectedDeploymentModelSummary.sourceModelId ? `，来源模型 ${selectedDeploymentModelSummary.sourceModelId}` : ""}。
                </small>
              ) : null}
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
            <span>执行后端</span>
            <select
              disabled={!deploymentBackendOptions.length}
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, providerPreference: event.target.value }))}
              value={resolveDeploymentProviderPreference(
                deploymentForm.providerPreference,
                data.runtimeCapabilities,
                deploymentForm.useGpu,
                selectedDeploymentModelSummary.optimizationType,
              )}
            >
              {deploymentBackendOptions.map((item) => (
                <option key={`deploy-backend-${item.value}`} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <small className="detail-note">
              {selectedDeploymentBackend?.note || "当前环境没有可用的 GPU 执行后端，平台已自动改为 CPU。"}
            </small>
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
                disabled={
                  !getDeploymentBackendOptions(data.runtimeCapabilities, true, selectedDeploymentModelSummary.optimizationType).length
                }
	              onChange={(event) =>
	                onDeploymentFormChange((current) => ({
	                  ...current,
                  useGpu: event.target.checked,
                  providerPreference: resolveDeploymentProviderPreference(
                    current.providerPreference,
                    data.runtimeCapabilities,
                    event.target.checked,
                    selectedDeploymentModelSummary.optimizationType,
                  ),
                  gpuIdsText: event.target.checked ? current.gpuIdsText : "",
                }))
              }
              type="checkbox"
	            />
	            使用 GPU
              {!getDeploymentBackendOptions(data.runtimeCapabilities, true, selectedDeploymentModelSummary.optimizationType).length ? (
                <small className="detail-note">当前服务器环境未通过 CUDA 运行时探测，平台已禁用 GPU 部署入口。</small>
              ) : null}
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
            <>
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
                  {deploymentPagination.pagedItems.map((item) => (
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
                        <div className="table-note">后端偏好：{formatProviderPreferenceLabel(item.provider_preference)}</div>
                        <div className="table-note">实际激活：{formatProviderSummary(item)}</div>
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
              <PaginationBar
                currentPage={deploymentPagination.page}
                totalPages={deploymentPagination.totalPages}
                pageSize={deploymentPagination.pageSize}
                totalItems={deploymentPagination.totalItems}
                onChange={deploymentPagination.setPage}
              />
            </>
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
  const [modelFilter, setModelFilter] = useState("all");
  const [sceneFilter, setSceneFilter] = useState("all");
  const [scopeFilter, setScopeFilter] = useState("effective");
  const [compareModelId, setCompareModelId] = useState("all");
  const [optimizationSourceModelId, setOptimizationSourceModelId] = useState("all");
  const [paperModelDraftId, setPaperModelDraftId] = useState("all");
  const [paperModelId, setPaperModelId] = useState("all");
  const deferredSearch = useDeferredValue(search);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageBenchmarks = currentPermissions.includes("benchmarks.manage");

  const completedBenchmarks = useMemo(
    () => sortByCreatedDesc(derived.benchmarks).filter((item) => item.status === "completed"),
    [derived.benchmarks],
  );

  const benchmarkModelOptions = useMemo(
    () => Array.from(new Set((derived.benchmarks || []).map((item) => String(item.model_id || "").trim()).filter(Boolean))).sort((left, right) => left.localeCompare(right)),
    [derived.benchmarks],
  );

  const benchmarkSceneOptions = useMemo(
    () => {
      const observed = Array.from(new Set((derived.benchmarks || []).map((item) => String(item.scene || "").trim()).filter(Boolean)));
      const fixed = BENCHMARK_SCENE_OPTIONS.map((item) => item.value);
      return Array.from(new Set([...fixed, ...observed])).sort((left, right) => left.localeCompare(right));
    },
    [derived.benchmarks],
  );

  useEffect(() => {
    if (modelFilter !== "all" && !benchmarkModelOptions.includes(modelFilter)) {
      setModelFilter("all");
    }
  }, [benchmarkModelOptions, modelFilter]);

  useEffect(() => {
    if (sceneFilter !== "all" && sceneFilter !== "none" && !benchmarkSceneOptions.includes(sceneFilter)) {
      setSceneFilter("all");
    }
  }, [benchmarkSceneOptions, sceneFilter]);

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
        [item.benchmark_id, item.label, item.model_id, item.deployment_id, item.profile_name, item.status, item.scene, item.trigger_source],
        deferredSearch,
      ),
    );

    if (statusFilter !== "all") {
      next = next.filter((item) => item.status === statusFilter);
    }

     if (modelFilter !== "all") {
      next = next.filter((item) => item.model_id === modelFilter);
    }

    if (sceneFilter === "none") {
      next = next.filter((item) => !String(item.scene || "").trim());
    } else if (sceneFilter !== "all") {
      next = next.filter((item) => String(item.scene || "").trim() === sceneFilter);
    }

    if (scopeFilter === "effective") {
      next = next.filter((item) => item.status === "completed" && item.is_effective);
    } else if (scopeFilter === "completed") {
      next = next.filter((item) => item.status === "completed");
    } else if (scopeFilter === "problematic") {
      next = next.filter((item) => item.failure_count > 0 || item.status === "failed");
    } else if (scopeFilter === "recent") {
      next = next.slice(0, 12);
    }

    return next;
  }, [deferredSearch, derived.benchmarks, modelFilter, sceneFilter, scopeFilter, statusFilter]);
  const benchmarkRecordPagination = useTablePagination(filteredBenchmarks, 5);

  const allCompareRows = useMemo(() => {
    return buildLatestPreferredBenchmarkCompareRows(
      completedBenchmarks,
      derived.deployments,
      derived.modelDetails,
      "all",
    );
  }, [completedBenchmarks, derived.deployments, derived.modelDetails]);

  const compareRows = useMemo(() => {
    return buildLatestPreferredBenchmarkCompareRows(
      completedBenchmarks,
      derived.deployments,
      derived.modelDetails,
      compareModelId,
    );
  }, [compareModelId, completedBenchmarks, derived.deployments, derived.modelDetails]);
  const compareRowPagination = useTablePagination(compareRows, 5);

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
  const thesisRowPagination = useTablePagination(thesisRows, 5);

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

  const optimizationGroups = useMemo(() => {
    const grouped = new Map();
    allCompareRows.forEach((row) => {
      const rootModelId = row.comparison_root_model_id || row.model_id;
      const current = grouped.get(rootModelId) || [];
      current.push(row);
      grouped.set(rootModelId, current);
    });
    return grouped;
  }, [allCompareRows]);

  const optimizationSourceOptions = useMemo(() => {
    return ["all", ...Array.from(optimizationGroups.entries())
      .filter(([, items]) => items.some((item) => item.is_generated_variant) && items.length > 1)
      .map(([rootModelId]) => rootModelId)
      .sort((left, right) => left.localeCompare(right))];
  }, [optimizationGroups]);

  useEffect(() => {
    if (!optimizationSourceOptions.includes(optimizationSourceModelId)) {
      setOptimizationSourceModelId("all");
    }
  }, [optimizationSourceModelId, optimizationSourceOptions]);

  const optimizationCompareRows = useMemo(() => {
    return buildOptimizationBenchmarkCompareRows(allCompareRows, optimizationSourceModelId);
  }, [allCompareRows, optimizationSourceModelId]);
  const optimizationComparePagination = useTablePagination(optimizationCompareRows, 5);

  const bestOptimizationLatencyRow = useMemo(() => {
    return [...optimizationCompareRows]
      .filter((item) => item.is_generated_variant && item.latency_gain_pct !== null)
      .sort((left, right) => Number(right.latency_gain_pct) - Number(left.latency_gain_pct))[0] || null;
  }, [optimizationCompareRows]);

  const bestOptimizationThroughputRow = useMemo(() => {
    return [...optimizationCompareRows]
      .filter((item) => item.is_generated_variant && item.throughput_gain_pct !== null)
      .sort((left, right) => Number(right.throughput_gain_pct) - Number(left.throughput_gain_pct))[0] || null;
  }, [optimizationCompareRows]);
  const selectedOptimizationSourceRows = useMemo(() => {
    if (optimizationSourceModelId === "all") {
      return [];
    }
    return optimizationCompareRows.filter((item) => item.source_root_model_id === optimizationSourceModelId);
  }, [optimizationCompareRows, optimizationSourceModelId]);
  const selectedOptimizationSourceSummary = useMemo(() => {
    if (!selectedOptimizationSourceRows.length) {
      return {
        bestLatency: null,
        bestThroughput: null,
        bestSuccess: null,
        bestLatencyGain: null,
        bestThroughputGain: null,
      };
    }
    return {
      bestLatency:
        [...selectedOptimizationSourceRows]
          .filter((item) => item.avg_latency_ms !== null)
          .sort((left, right) => Number(left.avg_latency_ms) - Number(right.avg_latency_ms))[0] || null,
      bestThroughput:
        [...selectedOptimizationSourceRows]
          .filter((item) => item.throughput_qps !== null)
          .sort((left, right) => Number(right.throughput_qps) - Number(left.throughput_qps))[0] || null,
      bestSuccess:
        [...selectedOptimizationSourceRows]
          .filter((item) => item.success_rate !== null)
          .sort((left, right) => Number(right.success_rate) - Number(left.success_rate))[0] || null,
      bestLatencyGain:
        [...selectedOptimizationSourceRows]
          .filter((item) => item.is_generated_variant && item.latency_gain_pct !== null)
          .sort((left, right) => Number(right.latency_gain_pct) - Number(left.latency_gain_pct))[0] || null,
      bestThroughputGain:
        [...selectedOptimizationSourceRows]
          .filter((item) => item.is_generated_variant && item.throughput_gain_pct !== null)
          .sort((left, right) => Number(right.throughput_gain_pct) - Number(left.throughput_gain_pct))[0] || null,
    };
  }, [selectedOptimizationSourceRows]);
  const optimizationChartMaxima = useMemo(() => {
    return {
      latency: Math.max(...selectedOptimizationSourceRows.map((item) => Number(item.avg_latency_ms || 0)), 1),
      throughput: Math.max(...selectedOptimizationSourceRows.map((item) => Number(item.throughput_qps || 0)), 1),
      success: Math.max(...selectedOptimizationSourceRows.map((item) => Number(item.success_rate || 0)), 1),
    };
  }, [selectedOptimizationSourceRows]);

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
        scene: item.scene || "",
        is_effective: item.is_effective ? "yes" : "no",
        trigger_source: item.trigger_source || "",
        superseded_by: item.superseded_by || "",
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

  function handleExportOptimizationCompareRows() {
    if (!optimizationCompareRows.length) {
      return;
    }
    downloadCsvFile(
      "luckymodels-optimization-compare.csv",
      optimizationCompareRows.map((item) => ({
        source_model_id: item.source_root_model_id,
        model_id: item.model_id,
        variant_label: item.variant_label,
        optimization_type: item.optimization_type || "",
        profile_name: item.profile_name,
        device: item.device || "",
        run_count: item.run_count,
        avg_latency_ms: item.avg_latency_ms?.toFixed(3) ?? "",
        latency_gain_pct: item.latency_gain_pct?.toFixed(2) ?? "",
        throughput_qps: item.throughput_qps?.toFixed(3) ?? "",
        throughput_gain_pct: item.throughput_gain_pct?.toFixed(2) ?? "",
        success_rate: item.success_rate?.toFixed(4) ?? "",
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
                onChange={(event) => {
                  const nextDeploymentId = event.target.value;
                  const nextDeployment = derived.deployments.find((item) => item.deployment_id === nextDeploymentId);
                  const nextModel = derived.models.find((item) => item.model_id === nextDeployment?.model_id);
                  onBenchmarkFormChange((current) => ({
                    ...current,
                    deploymentId: nextDeploymentId,
                    scene: resolveBenchmarkSceneForModel(nextModel),
                  }));
                }}
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
              <span>基准场景</span>
              <select
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, scene: event.target.value }))}
                value={benchmarkForm.scene || "zh_general"}
              >
                {BENCHMARK_SCENE_OPTIONS.map((item) => (
                  <option key={`benchmark-create-scene-${item.value}`} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
              <small className="detail-note">这会告诉平台这条 benchmark 代表哪类业务文本，用于后续任务调度和部署策略读取正确证据。</small>
            </label>
            <label className="stack-field">
              <span>基准标签</span>
              <input
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, label: event.target.value }))}
                placeholder="可选，例如：工业场景换卡后复测"
                type="text"
                value={benchmarkForm.label}
              />
              <small className="detail-note">标签主要用于人工区分这次测试的目的，不参与平台调度评分。</small>
            </label>
            <label className="stack-field">
              <span>测试轮次</span>
              <input
                min="1"
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, iterations: event.target.value }))}
                type="number"
                value={benchmarkForm.iterations}
              />
              <small className="detail-note">正式计入统计的请求轮次。轮次越多，结果越稳定，但耗时也更长。</small>
            </label>
            <label className="stack-field">
              <span>预热轮数</span>
              <input
                min="0"
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, warmupRuns: event.target.value }))}
                type="number"
                value={benchmarkForm.warmupRuns}
              />
              <small className="detail-note">预热请求不会计入最终统计，适合先让模型和运行时进入稳定状态。</small>
            </label>
            <label className="stack-field stack-field-wide">
              <span>基准文本</span>
              <textarea
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, text: event.target.value }))}
                rows="4"
                value={benchmarkForm.text}
              />
              <small className="detail-note">建议输入与所选场景一致的代表性文本。若是工业场景，尽量使用设备、故障、制造流程相关文本。</small>
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
              <span>模型</span>
              <select onChange={(event) => setModelFilter(event.target.value)} value={modelFilter}>
                <option value="all">全部模型</option>
                {benchmarkModelOptions.map((item) => (
                  <option key={`benchmark-model-${item}`} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
            <label className="stack-field">
              <span>场景</span>
              <select onChange={(event) => setSceneFilter(event.target.value)} value={sceneFilter}>
                <option value="all">全部场景</option>
                <option value="none">未标注场景</option>
                {benchmarkSceneOptions.map((item) => (
                  <option key={`benchmark-scene-${item}`} value={item}>
                    {formatBenchmarkSceneLabel(item)}
                  </option>
                ))}
              </select>
            </label>
            <label className="stack-field">
              <span>显示范围</span>
              <select onChange={(event) => setScopeFilter(event.target.value)} value={scopeFilter}>
                <option value="recent">最近 12 条</option>
                <option value="effective">仅最新有效</option>
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
                setModelFilter("all");
                setSceneFilter("all");
                setScopeFilter("effective");
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
            <>
              <table>
                <thead>
                  <tr>
                    <th>基准编号</th>
                    <th>模型</th>
                    <th>场景 / 有效性</th>
                    <th>状态</th>
                    <th>吞吐量</th>
                    <th>平均延迟</th>
                    <th>P95</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkRecordPagination.pagedItems.map((item) => (
                    <tr key={item.benchmark_id}>
                      <td>
                        <strong>{item.label || item.benchmark_id}</strong>
                        <div className="mono">{item.benchmark_id}</div>
                        <div className="table-note">{item.created_at || "-"}</div>
                      </td>
                      <td>
                        <div>{item.model_id}</div>
                        <div className="table-note">{item.deployment_id || "-"}</div>
                      </td>
                      <td>
                        <div>{formatBenchmarkSceneLabel(item.scene)}</div>
                        <div className="table-note">
                          {item.is_effective ? "最新有效记录" : item.superseded_by ? `已被 ${item.superseded_by} 覆盖` : "历史记录"}
                        </div>
                        <div className="table-note">触发方式：{item.trigger_source || "manual"}</div>
                      </td>
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
              <PaginationBar
                currentPage={benchmarkRecordPagination.page}
                totalPages={benchmarkRecordPagination.totalPages}
                pageSize={benchmarkRecordPagination.pageSize}
                totalItems={benchmarkRecordPagination.totalItems}
                onChange={benchmarkRecordPagination.setPage}
              />
            </>
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
          note="选择单个模型后，一键生成更适合论文截图的配置模板对比条形图和结论卡片。这里的性能值优先使用各配置模板最新有效 benchmark，没有有效记录时才回退到最新完成记录。"
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
                  当前页面按配置模板选取该模型最新的 benchmark 记录做正式对比，优先使用最新有效记录；如果某个模板还没有有效记录，就回退到该模板最新完成记录，可直接用于论文截图和实验结论整理。
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
          note="按模型比较不同配置模板的最新 benchmark 表现。每个模型/模板组合优先取最新有效记录，没有有效记录时回退到最新完成记录。CSV 导出会由浏览器下载到你本地电脑。"
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
              当前展示 {compareRows.length} 个模型/模板组合。性能值来自各组合的最新优先 benchmark，历史上共参考 {completedBenchmarks.length} 条已完成记录。
            </div>
          </div>
        </div>

        {compareRows.length ? (
          <>
            <div className="metrics">
              <MetricCard
                title="最低平均延迟"
                value={bestLatencyRow ? Number(bestLatencyRow.avg_latency_ms).toFixed(3) : "-"}
                note={bestLatencyRow ? `当前对比中延迟最优：${bestLatencyRow.model_id} · ${bestLatencyRow.profile_name}` : "当前还没有可比较模板"}
              />
              <MetricCard
                title="最高吞吐量"
                value={bestThroughputRow ? Number(bestThroughputRow.throughput_qps).toFixed(3) : "-"}
                note={bestThroughputRow ? `当前对比中吞吐最优：${bestThroughputRow.model_id} · ${bestThroughputRow.profile_name}` : "当前还没有可比较模板"}
              />
              <MetricCard
                title="对比组合数"
                value={compareRows.length}
                note={compareModelId === "all" ? "当前按全部模型的模板组合做正式对比" : `当前仅比较模型 ${compareModelId}`}
              />
              <MetricCard
                title="已完成记录"
                value={completedBenchmarks.length}
                note="仅已完成 benchmark 参与筛选，单个组合优先取最新有效记录"
              />
            </div>
            <div className="table-wrap compare-table-wrap">
              <table>
                <thead>
                <tr>
                  <th>模型</th>
                  <th>配置模板</th>
                  <th>历史完成次数</th>
                  <th>平均延迟</th>
                  <th>P95</th>
                  <th>吞吐量</th>
                  <th>成功率</th>
                  <th>平均批量</th>
                  <th>最新记录</th>
                </tr>
                </thead>
                <tbody>
                  {compareRowPagination.pagedItems.map((item) => (
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
                        <div className="table-note">{item.selection_label} · {formatBenchmarkSceneLabel(item.latest_scene)}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <PaginationBar
                currentPage={compareRowPagination.page}
                totalPages={compareRowPagination.totalPages}
                pageSize={compareRowPagination.pageSize}
                totalItems={compareRowPagination.totalItems}
                onChange={compareRowPagination.setPage}
              />
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
          kicker="优化对比"
          title="原始模型与优化版本对比"
          note="集中展示模型优化工作台生成模型与原始模型的 benchmark 对比结果。按“源模型”归并原始版本及其平台生成的优化版本，直接比较优化前后在最新 benchmark 上的延迟、吞吐和成功率变化。"
          right={
            <div className="actions">
              <button disabled={!optimizationCompareRows.length} onClick={handleExportOptimizationCompareRows} type="button">
                导出优化对比 CSV
              </button>
            </div>
          }
        />
        <div className="filter-panel compare-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>源模型</span>
              <select onChange={(event) => setOptimizationSourceModelId(event.target.value)} value={optimizationSourceModelId}>
                <option value="all">全部源模型</option>
                {optimizationSourceOptions.filter((item) => item !== "all").map((item) => (
                  <option key={`optimization-source-${item}`} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="filter-foot">
            <div className="filter-meta">
              当前纳入 {optimizationCompareRows.length} 个原始/优化版本组合条目。
            </div>
          </div>
        </div>

        {optimizationCompareRows.length ? (
          <>
            {optimizationSourceModelId !== "all" && selectedOptimizationSourceRows.length ? (
              <>
                <div className="paper-summary-grid">
                  <article className="paper-hero-card">
                    <div className="paper-hero-kicker">可用于论文</div>
                    <h3>{optimizationSourceModelId}</h3>
                    <p>
                      当前页面已按所选源模型整理出原始模型与平台生成优化版本的正式 benchmark 对比结果。
                      这里原始模型只保留一个最优对照版本参与展示；各优化类型也只保留该类型下表现最优的一条，
                      再分别与这个对照版本比较。
                      各版本数值优先取最新有效记录，没有有效记录时再回退到最新完成记录，适合直接截图写入优化实验小节。
                    </p>
                    <div className="tag-list">
                      <span className="tag">版本条目：{selectedOptimizationSourceRows.length}</span>
                      <span className="tag">生成版本：{selectedOptimizationSourceRows.filter((item) => item.is_generated_variant).length}</span>
                      <span className="tag">展示规则：原始最优 + 各优化类型最优</span>
                    </div>
                  </article>
                  <MetricCard
                    title="最低平均延迟"
                    value={selectedOptimizationSourceSummary.bestLatency ? formatFixed(selectedOptimizationSourceSummary.bestLatency.avg_latency_ms) : "-"}
                    note={selectedOptimizationSourceSummary.bestLatency ? `${selectedOptimizationSourceSummary.bestLatency.model_id} · ${selectedOptimizationSourceSummary.bestLatency.profile_name}` : "暂无"}
                  />
                  <MetricCard
                    title="最高吞吐量"
                    value={selectedOptimizationSourceSummary.bestThroughput ? formatFixed(selectedOptimizationSourceSummary.bestThroughput.throughput_qps) : "-"}
                    note={selectedOptimizationSourceSummary.bestThroughput ? `${selectedOptimizationSourceSummary.bestThroughput.model_id} · ${selectedOptimizationSourceSummary.bestThroughput.profile_name}` : "暂无"}
                  />
                  <MetricCard
                    title="最佳延迟收益"
                    value={selectedOptimizationSourceSummary.bestLatencyGain ? formatSignedPercentDelta(selectedOptimizationSourceSummary.bestLatencyGain.latency_gain_pct, 1) : "-"}
                    note={selectedOptimizationSourceSummary.bestLatencyGain ? `${selectedOptimizationSourceSummary.bestLatencyGain.model_id} · ${selectedOptimizationSourceSummary.bestLatencyGain.profile_name}` : "暂无优化收益"}
                  />
                  <MetricCard
                    title="最佳吞吐收益"
                    value={selectedOptimizationSourceSummary.bestThroughputGain ? formatSignedPercentDelta(selectedOptimizationSourceSummary.bestThroughputGain.throughput_gain_pct, 1) : "-"}
                    note={selectedOptimizationSourceSummary.bestThroughputGain ? `${selectedOptimizationSourceSummary.bestThroughputGain.model_id} · ${selectedOptimizationSourceSummary.bestThroughputGain.profile_name}` : "暂无优化收益"}
                  />
                </div>

                <div className="paper-chart-grid">
                  <article className="paper-chart-card">
                    <div className="paper-chart-title">平均延迟对比</div>
                    <div className="paper-chart-note">单位毫秒，越低越好</div>
                    <div className="paper-bar-list">
                      {selectedOptimizationSourceRows.map((item) => (
                        <div className="paper-bar-row" key={`${item.source_root_model_id}-${item.model_id}-${item.profile_name}-latency`}>
                          <div className="paper-bar-label">
                            <strong>{item.variant_label || item.model_id}</strong>
                            <span>{item.profile_name} / {item.device || "-"}</span>
                          </div>
                          <div className="paper-bar-track">
                            <div
                              className="paper-bar-fill paper-bar-latency"
                              style={{ width: `${Math.max((Number(item.avg_latency_ms || 0) / optimizationChartMaxima.latency) * 100, 6)}%` }}
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
                      {selectedOptimizationSourceRows.map((item) => (
                        <div className="paper-bar-row" key={`${item.source_root_model_id}-${item.model_id}-${item.profile_name}-throughput`}>
                          <div className="paper-bar-label">
                            <strong>{item.variant_label || item.model_id}</strong>
                            <span>{item.profile_name} / {item.providers || "-"}</span>
                          </div>
                          <div className="paper-bar-track">
                            <div
                              className="paper-bar-fill paper-bar-throughput"
                              style={{ width: `${Math.max((Number(item.throughput_qps || 0) / optimizationChartMaxima.throughput) * 100, 6)}%` }}
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
                      {selectedOptimizationSourceRows.map((item) => (
                        <div className="paper-bar-row" key={`${item.source_root_model_id}-${item.model_id}-${item.profile_name}-success`}>
                          <div className="paper-bar-label">
                            <strong>{item.variant_label || item.model_id}</strong>
                            <span>{item.selection_label} / {formatBenchmarkSceneLabel(item.latest_scene)}</span>
                          </div>
                          <div className="paper-bar-track">
                            <div
                              className="paper-bar-fill paper-bar-success"
                              style={{ width: `${Math.max((Number(item.success_rate || 0) / optimizationChartMaxima.success) * 100, 6)}%` }}
                            />
                            <span className="paper-bar-value">{formatPercent(item.success_rate)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </article>
                </div>
              </>
            ) : null}
            <div className="metrics">
              <MetricCard
                title="优化源模型数"
                value={new Set(optimizationCompareRows.map((item) => item.source_root_model_id)).size}
                note="统计范围内至少同时包含原始版本和平台生成版本的源模型数"
              />
              <MetricCard
                title="最佳延迟收益"
                value={bestOptimizationLatencyRow ? formatSignedPercentDelta(bestOptimizationLatencyRow.latency_gain_pct, 1) : "-"}
                note={bestOptimizationLatencyRow ? `当前筛选中延迟收益最高：${bestOptimizationLatencyRow.model_id} · ${bestOptimizationLatencyRow.profile_name}` : "当前没有可计算收益的优化版本"}
              />
              <MetricCard
                title="最佳吞吐收益"
                value={bestOptimizationThroughputRow ? formatSignedPercentDelta(bestOptimizationThroughputRow.throughput_gain_pct, 1) : "-"}
                note={bestOptimizationThroughputRow ? `当前筛选中吞吐收益最高：${bestOptimizationThroughputRow.model_id} · ${bestOptimizationThroughputRow.profile_name}` : "当前没有可计算收益的优化版本"}
              />
              <MetricCard
                title="当前筛选"
                value={optimizationSourceModelId === "all" ? "全部源模型" : optimizationSourceModelId}
                note={optimizationSourceModelId === "all" ? "适合看全平台优化概览；若要截图建议先选定单个源模型" : "当前已切到单一源模型，可直接截图上方展示卡"}
              />
            </div>
            <div className="banner banner-muted">
              这张表看什么：这里专门看模型优化工作台生成版本和原模型之间的正式 benchmark 差异。每个源模型只保留一个“原始最优版本”作为对照基线；每种优化类型也只保留一个“该类型最优版本”，再相对于同一基线计算延迟收益和吞吐收益，更适合论文里的基线对比表达。
            </div>
            <div className="table-wrap compare-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>源模型</th>
                    <th>版本模型</th>
                    <th>版本类型</th>
                    <th>配置模板 / 设备</th>
                    <th>平均延迟</th>
                    <th>相对原始最好延迟</th>
                    <th>吞吐量</th>
                    <th>相对原始最好吞吐</th>
                    <th>成功率</th>
                    <th>最新记录</th>
                  </tr>
                </thead>
                <tbody>
                  {optimizationComparePagination.pagedItems.map((item) => (
                    <tr key={`${item.source_root_model_id}-${item.model_id}-${item.profile_name}`}>
                      <td>
                        <strong>{item.source_root_model_id}</strong>
                        <div className="table-note">
                          对照版本 {item.baseline_profile_name || "-"} / {item.baseline_device || "-"}
                        </div>
                        <div className="table-note">
                          对照延迟 {item.baseline_best_latency_ms !== null ? Number(item.baseline_best_latency_ms).toFixed(3) : "-"} ms · 对照吞吐 {item.baseline_best_throughput_qps !== null ? Number(item.baseline_best_throughput_qps).toFixed(3) : "-"} QPS
                        </div>
                      </td>
                      <td>
                        <strong>{item.model_id}</strong>
                        <div className="table-note">{item.providers || "-"}</div>
                      </td>
                      <td>
                        <span className="tag">{item.variant_label}</span>
                        <div className="table-note">{item.optimization_type || "baseline"}</div>
                      </td>
                      <td>
                        <div>{item.profile_name}</div>
                        <div className="table-note">{item.device || "-"}</div>
                      </td>
                      <td>{item.avg_latency_ms !== null ? item.avg_latency_ms.toFixed(3) : "-"}</td>
                      <td>
                        {item.is_generated_variant && item.latency_gain_pct !== null
                          ? formatSignedPercentDelta(item.latency_gain_pct, 1)
                          : <span className="table-note">基线</span>}
                      </td>
                      <td>{item.throughput_qps !== null ? item.throughput_qps.toFixed(3) : "-"}</td>
                      <td>
                        {item.is_generated_variant && item.throughput_gain_pct !== null
                          ? formatSignedPercentDelta(item.throughput_gain_pct, 1)
                          : <span className="table-note">基线</span>}
                      </td>
                      <td>{formatPercent(item.success_rate)}</td>
                      <td>
                        <strong>{item.latest_benchmark_id}</strong>
                        <div className="table-note">{item.latest_created_at}</div>
                        <div className="table-note">{item.selection_label} · {formatBenchmarkSceneLabel(item.latest_scene)}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <PaginationBar
                currentPage={optimizationComparePagination.page}
                totalPages={optimizationComparePagination.totalPages}
                pageSize={optimizationComparePagination.pageSize}
                totalItems={optimizationComparePagination.totalItems}
                onChange={optimizationComparePagination.setPage}
              />
            </div>
          </>
        ) : (
          <EmptyState
            icon="O"
            title="暂无可展示的优化前后对比"
            note="先对某个模型生成优化版本，并分别跑完原始模型与优化版本的 benchmark，这里才会出现正式对比结果。"
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
                {thesisRowPagination.pagedItems.map((item) => (
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
            <PaginationBar
              currentPage={thesisRowPagination.page}
              totalPages={thesisRowPagination.totalPages}
              pageSize={thesisRowPagination.pageSize}
              totalItems={thesisRowPagination.totalItems}
              onChange={thesisRowPagination.setPage}
            />
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

function ResourcesPage({ actionBusy, derived, onApplyGpuPoolUpdate }) {
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
  const gpuPoolState = gpuStatus.gpu_pool_state || {
    effective_gpu_ids: [],
    disabled_gpu_ids: [],
    visible_gpu_ids: [],
  };
  const visibleGpuIds = gpuPoolState.visible_gpu_ids || (gpuStatus.devices || []).map((item) => item.index).filter((item) => item !== null && item !== undefined);
  const currentPermissions = derived.currentAccessPermissions || [];
  const canManageGpuPool = currentPermissions.includes("strategies.manage");
  const activeGpuProcesses = (gpuStatus.devices || []).reduce(
    (sum, device) => sum + Number(device.active_processes?.length || 0),
    0,
  );
  const explicitGpuBindings = derived.deployments.filter((item) => Array.isArray(item.gpu_ids) && item.gpu_ids.length);
  const gpuDevicePagination = useTablePagination(gpuStatus.devices || [], 5);
  const [selectedGpuIds, setSelectedGpuIds] = useState(gpuPoolState.effective_gpu_ids || []);

  useEffect(() => {
    setSelectedGpuIds(gpuPoolState.effective_gpu_ids || []);
  }, [JSON.stringify(gpuPoolState.effective_gpu_ids || [])]);

  const pendingDisabledGpuIds = useMemo(
    () => visibleGpuIds.filter((item) => !selectedGpuIds.includes(item)),
    [selectedGpuIds, visibleGpuIds],
  );
  const pendingAffectedDeployments = useMemo(() => {
    if (!pendingDisabledGpuIds.length) {
      return [];
    }
    return (derived.deployments || []).filter(
      (item) =>
        item.use_gpu &&
        item.status !== "deleted" &&
        (item.gpu_ids || []).some((gpuId) => pendingDisabledGpuIds.includes(gpuId)),
    );
  }, [derived.deployments, pendingDisabledGpuIds]);

  async function handleSubmitGpuPool() {
    await onApplyGpuPoolUpdate(selectedGpuIds);
  }

  function handleToggleGpu(deviceIndex, checked) {
    setSelectedGpuIds((current) => {
      const next = new Set(current);
      if (checked) {
        next.add(deviceIndex);
      } else {
        next.delete(deviceIndex);
      }
      return visibleGpuIds.filter((item) => next.has(item));
    });
  }

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
              平台启用 GPU：{(gpuPoolState.effective_gpu_ids || []).length}
              <br />
              活跃 GPU 进程：{activeGpuProcesses}
            </div>
          </article>
          <article className="resource-card">
            <h4>执行后端</h4>
            <div className="resource-meta">
              {Array.from(providerSet).join(" / ") || "待检测"}
            </div>
          </article>
          <article className="resource-card">
            <h4>平台范围</h4>
            <div className="resource-meta">
              当前可用 GPU：{(gpuPoolState.effective_gpu_ids || []).length ? gpuPoolState.effective_gpu_ids.join(", ") : "仅 CPU"}
              <br />
              已禁用 GPU：{(gpuPoolState.disabled_gpu_ids || []).length ? gpuPoolState.disabled_gpu_ids.join(", ") : "无"}
              <br />
              显式绑定 deployment：{explicitGpuBindings.length}
            </div>
          </article>
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="GPU 管理"
          title="GPU 使用范围管理"
          note="在这里一次性选择平台允许使用的 GPU。提交后会统一迁移受影响 deployment，并仅触发一轮部署优化，适合批量启用或禁用多张卡。"
        />
        <div className="metrics">
          <MetricCard title="服务器可见 GPU" value={visibleGpuIds.length} note="实时从当前服务器读取，不写死 GPU 数量" />
          <MetricCard title="平台允许使用" value={(gpuPoolState.effective_gpu_ids || []).length} note={(gpuPoolState.effective_gpu_ids || []).length ? `GPU ${gpuPoolState.effective_gpu_ids.join(", ")}` : "当前仅允许 CPU"} />
          <MetricCard title="平台禁用" value={(gpuPoolState.disabled_gpu_ids || []).length} note={(gpuPoolState.disabled_gpu_ids || []).length ? `GPU ${gpuPoolState.disabled_gpu_ids.join(", ")}` : "暂无禁用"} />
          <MetricCard title="待迁移实例" value={pendingAffectedDeployments.length} note="按当前勾选结果预估会受影响的 deployment 数" />
        </div>
        {gpuStatus.available ? (
          <>
            <div className="gpu-scope-grid">
              {(gpuStatus.devices || []).map((device) => {
                const deviceIndex = Number(device.index);
                const checked = selectedGpuIds.includes(deviceIndex);
                return (
                  <label
                    className={`gpu-scope-card ${checked ? "is-selected" : ""}`}
                    key={device.uuid || device.index}
                  >
                    <div className="gpu-scope-card-top">
                      <strong>GPU {device.index ?? "-"}</strong>
                      <input
                        checked={checked}
                        disabled={!canManageGpuPool}
                        onChange={(event) => handleToggleGpu(deviceIndex, event.target.checked)}
                        type="checkbox"
                      />
                    </div>
                    <div className="table-note">{device.name || "-"}</div>
                    <div className="table-note">
                      显存 {formatFixed(device.memory_used_mb, 0)} / {formatFixed(device.memory_total_mb, 0)} MB
                    </div>
                    <div className="table-note">
                      利用率 {formatFixed(device.utilization_gpu_pct, 0)}% · 活跃进程 {device.active_processes?.length || 0}
                    </div>
                    <div className="table-note">
                      当前状态：{device.platform_state_label || (checked ? "平台可用" : "平台禁用")}
                    </div>
                  </label>
                );
              })}
            </div>
            <div className="detail-note">
              预览说明：若本次应用后禁用了 {pendingDisabledGpuIds.length || 0} 张 GPU，预计会影响 {pendingAffectedDeployments.length} 个 deployment。
              平台会优先把这些 deployment 迁移到仍可用的 GPU；若剩余 GPU 空间不足，则回退到 CPU，且不会重复触发多轮优化。
            </div>
            <div className="actions">
              <button
                disabled={!canManageGpuPool}
                onClick={() => setSelectedGpuIds(visibleGpuIds)}
                type="button"
              >
                全部启用
              </button>
              <button
                disabled={!canManageGpuPool}
                onClick={() => setSelectedGpuIds([])}
                type="button"
              >
                全部停用，仅 CPU
              </button>
              <button
                className="btn-primary"
                disabled={!canManageGpuPool || actionBusy["gpu-pool-update"]}
                onClick={() => void handleSubmitGpuPool()}
                type="button"
              >
                {actionBusy["gpu-pool-update"] ? "应用中" : "应用 GPU 变更"}
              </button>
            </div>
            {!canManageGpuPool ? (
              <div className="table-note">当前账号只有查看权限，GPU 使用范围管理仅对管理员或策略管理员开放。</div>
            ) : null}
          </>
        ) : (
          <EmptyState
            icon="G"
            title="暂时无法管理 GPU 范围"
            note={gpuStatus.detail || "当前 API 没有拿到 nvidia-smi 返回结果。"}
          />
        )}
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
                  <th>平台范围</th>
                  <th>显存</th>
                  <th>利用率</th>
                  <th>温度 / 功耗</th>
                  <th>绑定 Deployment</th>
                  <th>活跃进程</th>
                </tr>
              </thead>
              <tbody>
                {gpuDevicePagination.pagedItems.map((device) => (
                  <tr key={device.uuid || device.index}>
                    <td>
                      <strong>GPU {device.index ?? "-"}</strong>
                      <div>{device.name || "-"}</div>
                      <div className="mono">{device.uuid || "-"}</div>
                    </td>
                    <td>
                      <span className={`tag ${device.platform_enabled ? "tag-emphasis" : ""}`}>
                        {device.platform_state_label || (device.platform_enabled ? "平台可用" : "平台禁用")}
                      </span>
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
            <PaginationBar
              currentPage={gpuDevicePagination.page}
              totalPages={gpuDevicePagination.totalPages}
              pageSize={gpuDevicePagination.pageSize}
              totalItems={gpuDevicePagination.totalItems}
              onChange={gpuDevicePagination.setPage}
            />
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
                  后端偏好：{formatProviderPreferenceLabel(item.provider_preference)}
                  <br />
                  实例数：{execution.instance_count ?? 1}
                  <br />
                  批量大小：{execution.batch_size ?? 1}
                  <br />
                  算子内 / 算子间线程：{execution.intra_op_threads ?? 0} / {execution.inter_op_threads ?? 0}
                  <br />
                  实际激活：{providers.join(", ") || "待检测"}
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
