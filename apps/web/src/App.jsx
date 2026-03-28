import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

import {
  createBenchmark,
  createDeployment,
  deleteBenchmark,
  deleteDeployment,
  getDeploymentHealth,
  getDeploymentLogs,
  loadDashboardData,
  predictWithDeployment,
  restartDeployment,
  runCasrelDemo,
  stopDeployment,
} from "./api";
import { EmptyState, MetricCard, SectionHeader, StatusPill } from "./components";
import { navGroups } from "./nav";
import { escapeValue, safeJsonParse } from "./utils";

const initialData = {
  summary: null,
  models: [],
  modelDetails: {},
  deployments: [],
  benchmarks: [],
  report: null,
};

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
    note: "选择运行中的 deployment 做真实推理，查看三元组、健康状态和运行日志。",
  },
  "/archive": {
    title: "结果归档",
    note: "集中查看最近的 deployment 与 benchmark，便于截图和整理论文表格。",
  },
  "/models": {
    title: "模型库",
    note: "查看模型资产并从平台直接发起新的 deployment。",
  },
  "/artifacts": {
    title: "制品包",
    note: "查看 manifest、ONNX 制品、tokenizer 和 profile 解析结果。",
  },
  "/deployments": {
    title: "部署记录",
    note: "新建、停止、重启、删除 deployment，并查看健康与日志。",
  },
  "/benchmarks": {
    title: "基准剖析",
    note: "为运行中的 deployment 创建基准测试，并管理历史 benchmark。",
  },
  "/resources": {
    title: "运行资源",
    note: "汇总当前控制面中的可推理模型、运行中 deployment 和 provider 信息。",
  },
  "/logs": {
    title: "日志中心",
    note: "聚焦 LuckyModels 当前选中 deployment 的运行日志。",
  },
};

function buildInitialDeploymentForm(models, details) {
  const preferred = models.find((model) => model.predict_ready) || models[0];
  const modelId = preferred?.model_id || "";
  const profiles = getProfilesForModel(modelId, details);
  return {
    modelId,
    profileName: profiles.includes("default") ? "default" : profiles[0] || "",
    notes: "",
    launchRuntime: true,
  };
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
  return `workers=${workers} · batch=${batchSize} · intra=${intraThreads} · inter=${interThreads}`;
}

function formatProviderSummary(deployment) {
  const providers = getDeploymentProviders(deployment);
  return providers.length ? providers.join(", ") : "pending";
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
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
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

function buildInitialPredictForm(deployments, models) {
  const running = deployments.find((item) => item.status === "running") || deployments[0];
  const matchingModel = models.find((item) => item.model_id === running?.model_id);
  return {
    deploymentId: running?.deployment_id || "",
    text: "",
    scene: matchingModel?.dataset_name ? `${matchingModel.dataset_name}_relation_extraction` : "",
    optionsJson: "{\"max_length\": 100}",
  };
}

export default function App() {
  const [data, setData] = useState(initialData);
  const [loadStatus, setLoadStatus] = useState("正在加载");
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState({ type: "ok", text: "" });
  const [demoRunning, setDemoRunning] = useState(false);
  const [actionBusy, setActionBusy] = useState({});
  const [deploymentForm, setDeploymentForm] = useState(buildInitialDeploymentForm([], {}));
  const [benchmarkForm, setBenchmarkForm] = useState(buildInitialBenchmarkForm([]));
  const [predictForm, setPredictForm] = useState(buildInitialPredictForm([], []));
  const [predictResult, setPredictResult] = useState(null);
  const [selectedHealth, setSelectedHealth] = useState(null);
  const [selectedLogs, setSelectedLogs] = useState(null);

  useEffect(() => {
    void refreshAll();
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
      if (current.text) {
        return current;
      }
      return { ...current, text: sampleText };
    });

    setBenchmarkForm((current) => {
      if (current.text) {
        return current;
      }
      return { ...current, text: sampleText };
    });
  }, [data.report]);

  async function refreshAll({ silent = false } = {}) {
    if (!silent) {
      setLoadError("");
      setLoadStatus("正在加载");
    }
    try {
      const next = await loadDashboardData();
      startTransition(() => {
        setData(next);
        setLoadStatus("已加载");
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLoadError(message);
      setLoadStatus("加载失败");
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

  async function handleRunDemo() {
    setDemoRunning(true);
    setLoadStatus("正在生成演示报告");
    try {
      await runCasrelDemo();
      setNotice({ type: "ok", text: "CasRel 演示报告已重新生成。" });
      await refreshAll({ silent: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLoadError(message);
      setLoadStatus("演示失败");
    } finally {
      setDemoRunning(false);
    }
  }

  async function handleCreateDeployment(payloadOverride) {
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
      notes: payloadOverride?.notes ?? deploymentForm.notes ?? undefined,
    };
    await runAction("create-deployment", () => createDeployment(payload), `Deployment 已创建：${payload.model_id}`);
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
      "Benchmark 已创建。",
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

  const derived = useMemo(() => {
    const summary = data.summary || { model_count: 0, deployment_count: 0, benchmark_count: 0 };
    const models = data.models || [];
    const modelDetails = data.modelDetails || {};
    const deployments = data.deployments || [];
    const benchmarks = data.benchmarks || [];
    const report = data.report || {};
    const runningDeployments = deployments.filter((item) => item.status === "running");
    const predictableModels = models.filter((item) => item.predict_ready);
    const latestDeployment = deployments.length ? [...deployments].reverse()[0] : null;
    const latestBenchmark = benchmarks.length ? [...benchmarks].reverse()[0] : null;
    const deploymentProfiles = getProfilesForModel(deploymentForm.modelId, modelDetails);
    return {
      summary,
      report,
      models,
      modelDetails,
      deployments,
      benchmarks,
      runningDeployments,
      predictableModels,
      latestDeployment,
      latestBenchmark,
      deploymentProfiles,
    };
  }, [data, deploymentForm.modelId]);

  return (
    <Shell
      actionBusy={actionBusy}
      benchmarkForm={benchmarkForm}
      data={data}
      demoRunning={demoRunning}
      deploymentForm={deploymentForm}
      derived={derived}
      loadError={loadError}
      loadStatus={loadStatus}
      notice={notice}
      onBenchmarkFormChange={setBenchmarkForm}
      onCreateBenchmark={handleCreateBenchmark}
      onCreateDeployment={handleCreateDeployment}
      onDeleteBenchmark={handleDeleteBenchmark}
      onDeploymentAction={handleDeploymentAction}
      onDeploymentFormChange={setDeploymentForm}
      onInspectDeployment={handleInspectDeployment}
      onPredict={handlePredict}
      onPredictFormChange={setPredictForm}
      onRefresh={refreshAll}
      onRunDemo={handleRunDemo}
      predictForm={predictForm}
      predictResult={predictResult}
      selectedHealth={selectedHealth}
      selectedLogs={selectedLogs}
    />
  );
}

function Shell(props) {
  const location = useLocation();
  const currentMeta = pageMeta[location.pathname] || pageMeta["/"];
  const {
    actionBusy,
    benchmarkForm,
    data,
    demoRunning,
    deploymentForm,
    derived,
    loadError,
    loadStatus,
    notice,
    onBenchmarkFormChange,
    onCreateBenchmark,
    onCreateDeployment,
    onDeleteBenchmark,
    onDeploymentAction,
    onDeploymentFormChange,
    onInspectDeployment,
    onPredict,
    onPredictFormChange,
    onRefresh,
    onRunDemo,
    predictForm,
    predictResult,
    selectedHealth,
    selectedLogs,
  } = props;

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
            当前前端已切换为真实控制台模式，支持 deployment 创建、停止、删除、benchmark 创建以及在线推理。
          </div>
        </div>

        {navGroups.map((group) => (
          <div className="nav-group" key={group.title}>
            <div className="nav-title">{group.title}</div>
            {group.items.map((item) => (
              <NavLink
                className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
                end={item.path === "/"}
                key={item.id}
                to={item.path}
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}

        <div className="nav-group">
          <div className="nav-title">当前状态</div>
          <div className="footer-note">模型 {derived.summary.model_count}</div>
          <div className="footer-note">部署 {derived.summary.deployment_count}</div>
          <div className="footer-note">基准 {derived.summary.benchmark_count}</div>
          <div className="footer-note mono">127.0.0.1:38080</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{currentMeta.title}</h1>
            <div className="topbar-note">{currentMeta.note}</div>
          </div>
          <div className="topbar-meta">
            <span>{loadStatus}</span>
            <div className="top-icon">LM</div>
          </div>
        </header>

        <section className="content">
          {notice.text ? (
            <div className={`banner ${notice.type === "error" ? "banner-error" : "banner-ok"}`}>{notice.text}</div>
          ) : null}
          {loadError ? <div className="banner banner-error">{loadError}</div> : null}

          <Routes>
            <Route element={<OverviewPage derived={derived} />} path="/" />
            <Route element={<OverviewPage derived={derived} />} path="/dashboard" />
            <Route
              element={
                <ExperimentsPage
                  actionBusy={actionBusy}
                  demoRunning={demoRunning}
                  derived={derived}
                  onPredict={onPredict}
                  onPredictFormChange={onPredictFormChange}
                  onRefresh={onRefresh}
                  onRunDemo={onRunDemo}
                  predictForm={predictForm}
                  predictResult={predictResult}
                  selectedHealth={selectedHealth}
                  selectedLogs={selectedLogs}
                />
              }
              path="/experiments"
            />
            <Route element={<ArchivePage derived={derived} />} path="/archive" />
            <Route
              element={
                <ModelsPage
                  actionBusy={actionBusy}
                  data={data}
                  deploymentForm={deploymentForm}
                  derived={derived}
                  onCreateDeployment={onCreateDeployment}
                  onDeploymentFormChange={onDeploymentFormChange}
                />
              }
              path="/models"
            />
            <Route element={<ArtifactsPage data={data} />} path="/artifacts" />
            <Route
              element={
                <DeploymentsPage
                  actionBusy={actionBusy}
                  deploymentForm={deploymentForm}
                  derived={derived}
                  onCreateDeployment={onCreateDeployment}
                  onDeploymentAction={onDeploymentAction}
                  onDeploymentFormChange={onDeploymentFormChange}
                  onInspectDeployment={onInspectDeployment}
                  selectedHealth={selectedHealth}
                  selectedLogs={selectedLogs}
                />
              }
              path="/deployments"
            />
            <Route
              element={
                <BenchmarksPage
                  actionBusy={actionBusy}
                  benchmarkForm={benchmarkForm}
                  derived={derived}
                  onBenchmarkFormChange={onBenchmarkFormChange}
                  onCreateBenchmark={onCreateBenchmark}
                  onDeleteBenchmark={onDeleteBenchmark}
                />
              }
              path="/benchmarks"
            />
            <Route element={<ResourcesPage derived={derived} />} path="/resources" />
            <Route element={<LogsPage derived={derived} selectedLogs={selectedLogs} />} path="/logs" />
            <Route element={<Navigate replace to="/" />} path="*" />
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
          <div className="small">predict_ready = true</div>
        </div>
        <div className="summary-card">
          <div className="title">运行中部署</div>
          <div className="big">{derived.runningDeployments.length}</div>
          <div className="small">当前活跃 runtime</div>
        </div>
        <div className="summary-card">
          <div className="title">基准总数</div>
          <div className="big">{derived.summary.benchmark_count}</div>
          <div className="small">历史 benchmark 记录</div>
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="Recent Activity" title="最近状态" />
          <div className="metrics">
            <MetricCard title="最新 Deployment" value={derived.latestDeployment?.deployment_id || "-"} note={derived.latestDeployment?.model_id || "暂无"} />
            <MetricCard title="Deployment 状态" value={derived.latestDeployment?.status || "-"} note={derived.latestDeployment?.service_url || "-"} />
            <MetricCard title="最新 Benchmark" value={derived.latestBenchmark?.benchmark_id || "-"} note={derived.latestBenchmark?.model_id || "暂无"} />
            <MetricCard title="CasRel Report" value={derived.report.status || "empty"} note="/api/v1/demo/casrel/report" />
          </div>
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="Model Status" title="模型运行口径" />
          <div className="tag-list">
            {derived.models.map((model) => (
              <span className="tag" key={model.model_id}>
                {model.model_id} · {model.runtime_status}
              </span>
            ))}
          </div>
        </article>
      </section>
    </>
  );
}

function ExperimentsPage({
  actionBusy,
  demoRunning,
  derived,
  onPredict,
  onPredictFormChange,
  onRefresh,
  onRunDemo,
  predictForm,
  predictResult,
  selectedHealth,
  selectedLogs,
}) {
  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Live Predict"
          title="在线推理"
          note="从平台里选择运行中的 deployment，直接走控制面代理接口完成预测。"
          right={
            <div className="actions">
              <button onClick={() => void onRefresh()} type="button">刷新数据</button>
              <button className="btn-primary" disabled={demoRunning} onClick={() => void onRunDemo()} type="button">
                {demoRunning ? "运行中" : "刷新 CasRel 演示"}
              </button>
            </div>
          }
        />
        {derived.runningDeployments.length ? (
          <div className="form-grid">
            <label className="stack-field">
              <span>选择 Deployment</span>
              <select
                onChange={(event) => onPredictFormChange((current) => ({ ...current, deploymentId: event.target.value }))}
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
              <span>场景标识</span>
              <input
                onChange={(event) => onPredictFormChange((current) => ({ ...current, scene: event.target.value }))}
                type="text"
                value={predictForm.scene}
              />
            </label>
            <label className="stack-field stack-field-wide">
              <span>输入文本</span>
              <textarea
                onChange={(event) => onPredictFormChange((current) => ({ ...current, text: event.target.value }))}
                rows="4"
                value={predictForm.text}
              />
            </label>
            <label className="stack-field stack-field-wide">
              <span>Options JSON</span>
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
          <EmptyState icon="▶" title="暂无运行中的 Deployment" note="先去“部署记录”或“模型库”里创建并启动一个 deployment。" />
        )}
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="Predict Result" title="预测结果" />
          {predictResult ? (
            <>
              <div className="metrics">
                <MetricCard title="模型" value={predictResult.model_id} note={predictResult.model_name} />
                <MetricCard title="Latency" value={predictResult.latency_ms} note="ms" />
                <MetricCard title="Triples" value={predictResult.triple_count} note={predictResult.language} />
                <MetricCard title="Version" value={predictResult.version} note="runtime output" />
              </div>
              <div className="relation-list">
                {(predictResult.triples || []).map((item, index) => (
                  <div className="relation-item" key={`${item.subject}-${item.predicate}-${item.object}-${index + 1}`}>
                    {item.subject}
                    {" -> "}
                    {item.predicate}
                    {" -> "}
                    {item.object}
                    <br />
                    confidence: {item.confidence}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <EmptyState icon="T" title="还没有预测结果" note="从上方选择一个运行中的 deployment 并提交文本。" />
          )}
        </article>

        <article className="card detail-card">
          <SectionHeader kicker="Runtime Detail" title="运行快照" />
          <div className="logs">{selectedHealth ? JSON.stringify(selectedHealth, null, 2) : selectedLogs?.content || "先在部署页查看健康或日志。"} </div>
        </article>
      </section>
    </>
  );
}

function ArchivePage({ derived }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader kicker="Archive" title="结果归档" note="聚焦最近一次 deployment、benchmark 和 CasRel 演示报告。" />
      <div className="metrics">
        <MetricCard title="Latest Deployment" value={derived.latestDeployment?.deployment_id || "-"} note={derived.latestDeployment?.model_id || "-"} />
        <MetricCard title="Latest Benchmark" value={derived.latestBenchmark?.benchmark_id || "-"} note={derived.latestBenchmark?.model_id || "-"} />
        <MetricCard title="CasRel Report" value={derived.report.status || "empty"} note={derived.report.dataset?.name || "-"} />
        <MetricCard title="Eval Cases" value={derived.report.dataset?.prepared_eval_count || "-"} note={derived.report.dataset?.prepared_eval_file || "-"} />
      </div>
    </section>
  );
}

function ModelsPage({ actionBusy, data, deploymentForm, derived, onCreateDeployment, onDeploymentFormChange }) {
  const [search, setSearch] = useState("");
  const [runtimeFilter, setRuntimeFilter] = useState("all");
  const [availabilityFilter, setAvailabilityFilter] = useState("all");
  const deferredSearch = useDeferredValue(search);

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
      .filter((model) => {
        if (availabilityFilter === "all") {
          return true;
        }
        return availabilityFilter === "predict_ready" ? model.predict_ready : !model.predict_ready;
      })
      .sort(
        (left, right) =>
          Number(right.predict_ready) - Number(left.predict_ready) || left.model_name.localeCompare(right.model_name),
      );
  }, [availabilityFilter, deferredSearch, derived.models, runtimeFilter]);

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader kicker="Create Deployment" title="从模型库创建 Deployment" note="选择模型、profile 和备注，直接通过控制面创建 deployment。" />
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
            <span>Profile</span>
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
            <span>备注</span>
            <input
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, notes: event.target.value }))}
              type="text"
              value={deploymentForm.notes}
            />
          </label>
          <label className="stack-check">
            <input
              checked={deploymentForm.launchRuntime}
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, launchRuntime: event.target.checked }))}
              type="checkbox"
            />
            启动 runtime
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={actionBusy["create-deployment"] || !deploymentForm.modelId || !deploymentForm.profileName}
              onClick={() => void onCreateDeployment()}
              type="button"
            >
              新建 Deployment
            </button>
          </div>
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="Model Library" title="模型资产" note="模型卡片支持快速查看 runtime 状态，并为可推理模型直接创建 deployment。" />
        <div className="filter-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>搜索模型</span>
              <input
                onChange={(event) => setSearch(event.target.value)}
                placeholder="模型名 / model_id / dataset / adapter"
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
              <span>可用性</span>
              <select onChange={(event) => setAvailabilityFilter(event.target.value)} value={availabilityFilter}>
                <option value="all">全部模型</option>
                <option value="predict_ready">仅可推理</option>
                <option value="package_only">仅待适配</option>
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
                setAvailabilityFilter("all");
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
            return (
              <article className="asset-card" key={model.model_id}>
                <h4>{model.model_name}</h4>
                <div className="asset-meta">
                  编号：<span className="mono">{model.model_id}</span>
                  <br />
                  数据集：{model.dataset_name || "-"}
                  <br />
                  backbone：{model.backbone}
                  <br />
                  adapter：{model.runtime_adapter}
                  <br />
                  profile：{Object.keys(detail.resolved_profiles || {}).join(", ") || "-"}
                </div>
                <div className="tag-list">
                  <span className="tag">{model.runtime_status}</span>
                  <span className="tag">{manifest.framework || "-"}</span>
                  <span className="tag">{manifest.stage_count || "-" } stages</span>
                </div>
                <div className="actions inline-actions">
                  <button
                    className="btn-primary"
                    disabled={!model.predict_ready || actionBusy["create-deployment"]}
                    onClick={() =>
                      void onCreateDeployment({
                        model_id: model.model_id,
                        profile_name: resolveProfileName(model.model_id, data.modelDetails, "default"),
                        launch_runtime: true,
                        notes: "quick-create-from-models",
                      })
                    }
                    type="button"
                  >
                    快速部署
                  </button>
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

function ArtifactsPage({ data }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader kicker="Artifacts" title="制品包清单" note="展示 manifest 与解析后的制品路径，便于检查平台识别结果。" />
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
            <div className="list-card-body">
              package_root：<span className="mono">{detail.summary?.package_root || "-"}</span>
              <br />
              tokenizer：{detail.manifest?.tokenizer?.type || "-"}
              <br />
              relation_map：{detail.manifest?.labels?.relation_map || "-"}
            </div>
            <div className="tag-list">
              {Object.entries(detail.resolved_artifacts || {}).map(([name, value]) => (
                <span className="tag" key={`${detail.summary?.model_id}-${name}`}>
                  {name}: {value}
                </span>
              ))}
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
  onInspectDeployment,
  selectedHealth,
  selectedLogs,
}) {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [scopeFilter, setScopeFilter] = useState("recent");
  const deferredSearch = useDeferredValue(search);

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

    if (scopeFilter === "active") {
      next = next.filter((item) => ["running", "starting", "planned", "stopping"].includes(item.status));
    } else if (scopeFilter === "with_logs") {
      next = next.filter((item) => item.log_path);
    } else if (scopeFilter === "recent") {
      next = next.slice(0, 12);
    }

    return next;
  }, [deferredSearch, derived.deployments, scopeFilter, statusFilter]);

  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Deployment Control"
          title="部署控制"
          note="可以在这里创建、停止、重启和删除 deployment。threads / instance_count / batch_size 已接入 runtime。"
        />
        <div className="banner banner-muted">
          如果某个 deployment 是在 2026-03-28 这次 profile 更新前创建且一直没有重启，它仍可能运行旧参数。重启一次后，新的线程数、worker 数和 batch 配置才会真正生效。
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
            <span>Profile</span>
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
            <span>备注</span>
            <input
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, notes: event.target.value }))}
              type="text"
              value={deploymentForm.notes}
            />
          </label>
          <label className="stack-check">
            <input
              checked={deploymentForm.launchRuntime}
              onChange={(event) => onDeploymentFormChange((current) => ({ ...current, launchRuntime: event.target.checked }))}
              type="checkbox"
            />
            启动 runtime
          </label>
          <div className="actions">
            <button
              className="btn-primary"
              disabled={actionBusy["create-deployment"] || !deploymentForm.modelId || !deploymentForm.profileName}
              onClick={() => void onCreateDeployment()}
              type="button"
            >
              创建 Deployment
            </button>
          </div>
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader kicker="Deployment List" title="部署列表" />
        <div className="filter-panel">
          <div className="filter-grid">
            <label className="stack-field">
              <span>搜索 deployment</span>
              <input
                onChange={(event) => setSearch(event.target.value)}
                placeholder="deployment_id / model_id / notes / service_url"
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
              <span>显示范围</span>
              <select onChange={(event) => setScopeFilter(event.target.value)} value={scopeFilter}>
                <option value="recent">最近 12 条</option>
                <option value="active">仅活动 deployment</option>
                <option value="with_logs">仅有日志</option>
                <option value="all">全部记录</option>
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
                setScopeFilter("recent");
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
                  <th>Deployment</th>
                  <th>模型</th>
                  <th>Profile / 执行参数</th>
                  <th>状态</th>
                  <th>Service URL</th>
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
                      <div><strong>{item.profile_name}</strong></div>
                      <div className="table-note">{item.device || "-"} / {item.profile_runtime || item.runtime || "-"}</div>
                      <div className="table-note">{formatExecutionSummary(item)}</div>
                      <div className="table-note">provider: {formatProviderSummary(item)}</div>
                    </td>
                    <td><StatusPill status={item.status} /></td>
                    <td>
                      <div className="mono">{item.service_url || "-"}</div>
                      <div className="table-note">pid: {item.pid || "-"}</div>
                    </td>
                    <td>
                      <div className="actions row-actions">
                        <button onClick={() => void onInspectDeployment("health", item.deployment_id)} type="button">健康</button>
                        <button onClick={() => void onInspectDeployment("logs", item.deployment_id)} type="button">日志</button>
                        <button disabled={actionBusy[`stop-${item.deployment_id}`]} onClick={() => void onDeploymentAction("stop", item.deployment_id)} type="button">停止</button>
                        <button disabled={actionBusy[`restart-${item.deployment_id}`]} onClick={() => void onDeploymentAction("restart", item.deployment_id)} type="button">重启</button>
                        <button disabled={actionBusy[`delete-${item.deployment_id}`]} onClick={() => void onDeploymentAction("delete", item.deployment_id)} type="button">删除</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState
              icon="D"
              note={derived.deployments.length ? "当前筛选条件下没有匹配 deployment。" : "平台里还没有创建 deployment。"}
              title="暂无部署记录"
            />
          )}
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <SectionHeader kicker="Selected Health" title="健康状态" />
          <div className="logs">{selectedHealth ? JSON.stringify(selectedHealth, null, 2) : "点击 deployment 的“健康”按钮后在此显示。"} </div>
        </article>
        <article className="card detail-card">
          <SectionHeader kicker="Selected Logs" title="运行日志" />
          <div className="logs">{selectedLogs?.content || "点击 deployment 的“日志”按钮后在此显示。"}</div>
        </article>
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
      <section className="page-shell card section-shell">
        <SectionHeader kicker="Create Benchmark" title="新建基准测试" note="为运行中的 deployment 发起 benchmark，并将结果写回平台。" />
        {derived.runningDeployments.length ? (
          <div className="form-grid">
            <label className="stack-field">
              <span>Deployment</span>
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
              <span>Label</span>
              <input
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, label: event.target.value }))}
                type="text"
                value={benchmarkForm.label}
              />
            </label>
            <label className="stack-field">
              <span>Iterations</span>
              <input
                min="1"
                onChange={(event) => onBenchmarkFormChange((current) => ({ ...current, iterations: event.target.value }))}
                type="number"
                value={benchmarkForm.iterations}
              />
            </label>
            <label className="stack-field">
              <span>Warmup</span>
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
                新建 Benchmark
              </button>
            </div>
          </div>
        ) : (
          <EmptyState icon="B" title="暂无运行中的 Deployment" note="先创建并启动 deployment，才能在这里发起 benchmark。" />
        )}
      </section>

      <section className="page-shell card table-shell">
        <SectionHeader kicker="Benchmark Records" title="基准记录" />
        <div className="filter-panel filter-panel-table">
          <div className="filter-grid">
            <label className="stack-field">
              <span>搜索 benchmark</span>
              <input
                onChange={(event) => setSearch(event.target.value)}
                placeholder="benchmark_id / label / model_id / deployment_id"
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
                  <th>Benchmark</th>
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
                      <button
                        disabled={actionBusy[`delete-benchmark-${item.benchmark_id}`]}
                        onClick={() => void onDeleteBenchmark(item.benchmark_id)}
                        type="button"
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState
              icon="☰"
              note={derived.benchmarks.length ? "当前筛选条件下没有匹配 benchmark。" : "创建 benchmark 后会在这里出现。"}
              title="暂无基准记录"
            />
          )}
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Paper Snapshot"
          title="论文截图视图"
          note="选择单个模型后，一键生成更适合论文截图的 profile 对比条形图和结论卡片。导出文件仍会下载到你本地浏览器默认下载目录。"
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
                <div className="paper-hero-kicker">Paper Ready</div>
                <h3>{paperModelId}</h3>
                <p>
                  当前页面已经按 profile 聚合了该模型的已完成 benchmark，可直接用于论文截图、性能趋势说明和实验对比表整理。
                </p>
                <div className="tag-list">
                  <span className="tag">profiles: {selectedPaperModelRows.length}</span>
                  <span className="tag">runs: {selectedPaperModelRows.reduce((sum, item) => sum + item.run_count, 0)}</span>
                  <span className="tag">provider: {selectedPaperModelRows[0]?.providers || "pending"}</span>
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
                <div className="paper-chart-note">单位 ms，越低越好</div>
                <div className="paper-bar-list">
                  {selectedPaperModelRows.map((item) => (
                    <div className="paper-bar-row" key={`${item.model_id}-${item.profile_name}-latency`}>
                      <div className="paper-bar-label">
                        <strong>{item.profile_name}</strong>
                        <span>batch {formatFixed(item.avg_batch_size, 2)} / workers {item.instance_count ?? "-"}</span>
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
                <div className="paper-chart-note">单位 qps，越高越好</div>
                <div className="paper-bar-list">
                  {selectedPaperModelRows.map((item) => (
                    <div className="paper-bar-row" key={`${item.model_id}-${item.profile_name}-throughput`}>
                      <div className="paper-bar-label">
                        <strong>{item.profile_name}</strong>
                        <span>intra {item.intra_op_threads ?? "-"} / inter {item.inter_op_threads ?? "-"}</span>
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
                <div className="paper-chart-note">按 total_requests 统计</div>
                <div className="paper-bar-list">
                  {selectedPaperModelRows.map((item) => (
                    <div className="paper-bar-row" key={`${item.model_id}-${item.profile_name}-success`}>
                      <div className="paper-bar-label">
                        <strong>{item.profile_name}</strong>
                        <span>max_len {item.max_length ?? "-"} / device {item.device || "-"}</span>
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
            note="选择某个模型后点击“生成论文视图”，这里就会出现更适合截图的 profile 图表。"
          />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Profile Compare"
          title="Profile 对比视图"
          note="按模型聚合已完成 benchmark，直接比较不同 profile 的平均延迟、P95、吞吐和成功率。CSV 导出会由浏览器下载到你本地电脑。"
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
              当前聚合 {compareRows.length} 个 model/profile 组合，基于 {completedBenchmarks.length} 条已完成 benchmark。
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
                note="仅 completed benchmark 纳入聚合"
              />
            </div>
            <div className="table-wrap compare-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>模型</th>
                    <th>Profile</th>
                    <th>样本数</th>
                    <th>平均延迟</th>
                    <th>P95</th>
                    <th>吞吐量</th>
                    <th>成功率</th>
                    <th>平均 Batch</th>
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
            title="暂无可对比的 benchmark"
            note="先运行至少一条 completed benchmark，Profile 对比视图才会出现聚合结果。"
          />
        )}
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Paper Table"
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
                  <th>Profile</th>
                  <th>Batch</th>
                  <th>Workers</th>
                  <th>Intra</th>
                  <th>Inter</th>
                  <th>MaxLen</th>
                  <th>Avg Latency</th>
                  <th>P95</th>
                  <th>Throughput</th>
                  <th>Success</th>
                  <th>Runs</th>
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
            note="先运行 completed benchmark，或者在上方生成某个模型的论文视图。"
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
            <h4>资源使用</h4>
            <div className="resource-meta">
              历史 deployment：{derived.deployments.length}
              <br />
              历史 benchmark：{derived.benchmarks.length}
              <br />
              Provider：{Array.from(providerSet).join(" / ") || "pending"}
            </div>
          </article>
          <article className="resource-card">
            <h4>可用模型</h4>
            <div className="resource-meta">
              {derived.predictableModels.map((item) => item.model_name).join(" / ") || "暂无"}
            </div>
          </article>
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Runtime Snapshot"
          title="运行实例快照"
          note="这里展示运行中 deployment 的真实 profile 执行参数与 runtime 健康信息。"
        />
        <div className="banner banner-muted">
          当前机器还没有启用 CUDA Execution Provider，所以这里看到的 provider 大概率仍是 CPU。Profile 已先接入 CPU 侧线程数、worker 数和 batch 行为。
        </div>
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
                  profile：<strong>{item.profile_name}</strong>
                  <br />
                  device / runtime：{item.device || "-"} / {item.profile_runtime || item.runtime || "-"}
                  <br />
                  workers：{execution.instance_count ?? 1}
                  <br />
                  batch_size：{execution.batch_size ?? 1}
                  <br />
                  intra / inter threads：{execution.intra_op_threads ?? 0} / {execution.inter_op_threads ?? 0}
                  <br />
                  providers：{providers.join(", ") || "pending"}
                  <br />
                  pid_alive：{String(item.health?.pid_alive ?? false)}
                  <br />
                  sessions_loaded：{String(payload.sessions_loaded ?? false)}
                  <br />
                  service_url：<span className="mono">{item.service_url || "-"}</span>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            icon="R"
            title="暂无运行快照"
            note="先启动一个 deployment，平台才会自动采集 healthz 并展示真实执行参数。"
          />
        )}
      </section>
    </>
  );
}

function LogsPage({ derived, selectedLogs }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader kicker="Logs" title="日志中心" note="默认展示当前选中的 deployment 日志，没有选择时回退到最近 CasRel report 日志。" />
      <div className="logs">{selectedLogs?.content || derived.report.logs?.content || "先到部署页加载某个 deployment 的日志。"}</div>
    </section>
  );
}
