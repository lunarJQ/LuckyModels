import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { loadDashboardData, runCasrelDemo } from "./api";
import { EmptyState, MetricCard, SectionHeader, StatusPill } from "./components";
import { navGroups } from "./nav";
import { escapeValue, formatArray, safeJsonParse } from "./utils";

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
    title: "资产概览",
    note: "LuckyModels 的整体平台入口，适合用作论文中的系统首页截图。",
  },
  "/dashboard": {
    title: "资产概览",
    note: "LuckyModels 的整体平台入口，适合用作论文中的系统首页截图。",
  },
  "/experiments": {
    title: "推理试验",
    note: "聚焦 CasRel 运行演示、筛选条件和当前实验数据的操作页。",
  },
  "/archive": {
    title: "结果归档",
    note: "集中查看报告状态、导出文件和论文表格最常引用的数据来源。",
  },
  "/models": {
    title: "模型库",
    note: "展示标准模型包、家族信息、语言与 runtime adapter。",
  },
  "/artifacts": {
    title: "制品包",
    note: "展示 manifest、ONNX 制品、tokenizer 和 profile 配置。",
  },
  "/deployments": {
    title: "部署记录",
    note: "查看 deployment 状态、service URL 和日志路径。",
  },
  "/benchmarks": {
    title: "基准剖析",
    note: "查看吞吐量、延迟、P95 与 deployment 对应关系。",
  },
  "/resources": {
    title: "运行资源",
    note: "汇总节点、GPU provider 和 runtime backend 信息。",
  },
  "/logs": {
    title: "日志中心",
    note: "查看最新 CasRel runtime 的真实运行日志。",
  },
};

export default function App() {
  const [data, setData] = useState(initialData);
  const [filters, setFilters] = useState({
    query: "",
    model: "",
    profile: "",
    status: "",
  });
  const [loadStatus, setLoadStatus] = useState("正在加载");
  const [loadError, setLoadError] = useState("");
  const [demoRunning, setDemoRunning] = useState(false);
  const deferredQuery = useDeferredValue(filters.query);

  useEffect(() => {
    void refreshAll();
  }, []);

  async function refreshAll() {
    setLoadError("");
    setLoadStatus("正在加载");
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

  async function handleRunDemo() {
    setDemoRunning(true);
    setLoadStatus("正在生成演示报告");
    try {
      await runCasrelDemo();
      await refreshAll();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setLoadError(message);
      setLoadStatus("演示失败");
    } finally {
      setDemoRunning(false);
    }
  }

  const derived = useMemo(() => {
    const summary = data.summary || { model_count: 0, deployment_count: 0, benchmark_count: 0 };
    const report = data.report || {};
    const dataset = report.dataset || {};
    const benchmark = report.benchmark || {};
    const deployment = report.deployment || {};
    const stopped = report.deployment_stopped || {};
    const primaryCase = report.samples?.primary_case || {};
    const secondaryCase = report.samples?.secondary_case || {};
    const runtime = primaryCase.runtime || {};
    const warnings = runtime.warnings || [];
    const healthPayload = safeJsonParse(report.health?.payload || "{}");
    const models = data.models || [];
    const deployments = data.deployments || [];
    const benchmarks = data.benchmarks || [];
    const modelOptions = [...new Set(benchmarks.map((item) => item.model_id).filter(Boolean))];
    const profileOptions = [...new Set(benchmarks.map((item) => item.profile_name).filter(Boolean))];
    const filteredBenchmarks = benchmarks.filter((item) => {
      const blob = `${item.benchmark_id} ${item.label || ""} ${item.model_id || ""} ${item.profile_name || ""}`.toLowerCase();
      if (deferredQuery && !blob.includes(deferredQuery.trim().toLowerCase())) return false;
      if (filters.model && item.model_id !== filters.model) return false;
      if (filters.profile && item.profile_name !== filters.profile) return false;
      if (filters.status && item.status !== filters.status) return false;
      return true;
    });

    return {
      summary,
      report,
      dataset,
      benchmark,
      deployment,
      stopped,
      primaryCase,
      secondaryCase,
      runtime,
      warnings,
      healthPayload,
      models,
      deployments,
      benchmarks,
      modelOptions,
      profileOptions,
      filteredBenchmarks,
    };
  }, [data, deferredQuery, filters]);

  return (
    <Shell
      data={data}
      derived={derived}
      demoRunning={demoRunning}
      filters={filters}
      loadError={loadError}
      loadStatus={loadStatus}
      onChangeFilters={setFilters}
      onRefresh={refreshAll}
      onRunDemo={handleRunDemo}
    />
  );
}

function Shell({
  data,
  derived,
  demoRunning,
  filters,
  loadError,
  loadStatus,
  onChangeFilters,
  onRefresh,
  onRunDemo,
}) {
  const location = useLocation();
  const currentMeta = pageMeta[location.pathname] || pageMeta["/"];

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
            基于 React 路由拆分的 LuckyModels 前端控制台，围绕模型资产、部署验证、运行调试和基准剖析组织平台视图。
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
          <div className="footer-note">
            模型 {derived.summary.model_count} · 部署 {derived.summary.deployment_count} · 基准 {derived.summary.benchmark_count}
          </div>
          <div className="footer-note mono">/api/v1/demo/casrel/report</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{currentMeta.title}</h1>
            <div className="topbar-note">{currentMeta.note}</div>
          </div>
          <div className="topbar-meta">
            <span>react-router v0.3.0</span>
            <div className="top-icon">R</div>
            <div className="top-icon">UI</div>
          </div>
        </header>

        <section className="content">
          <Routes>
            <Route element={<OverviewPage derived={derived} />} path="/" />
            <Route element={<OverviewPage derived={derived} />} path="/dashboard" />
            <Route
              element={
                <ExperimentsPage
                  demoRunning={demoRunning}
                  derived={derived}
                  filters={filters}
                  loadStatus={loadStatus}
                  onChangeFilters={onChangeFilters}
                  onRefresh={onRefresh}
                  onRunDemo={onRunDemo}
                />
              }
              path="/experiments"
            />
            <Route element={<ArchivePage derived={derived} />} path="/archive" />
            <Route element={<ModelsPage data={data} derived={derived} />} path="/models" />
            <Route element={<ArtifactsPage data={data} />} path="/artifacts" />
            <Route element={<DeploymentsPage derived={derived} />} path="/deployments" />
            <Route
              element={<BenchmarksPage derived={derived} loadError={loadError} loadStatus={loadStatus} />}
              path="/benchmarks"
            />
            <Route element={<ResourcesPage derived={derived} />} path="/resources" />
            <Route element={<LogsPage loadError={loadError} report={derived.report} />} path="/logs" />
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
      <section className="hero page-shell">
        <div className="hero-card">
          <div className="hero-kicker">LuckyModels / Routed Control Plane</div>
          <h2>模型资产、部署验证与基准实验的多页面前端</h2>
          <div className="hero-desc">
            这一版把原来所有内容堆在一页里的方式拆成了真正的页面路由。后面继续加模型详情页、部署详情页、结果对比页和图表会清晰很多。
          </div>
          <div className="hero-points">
            <span className="hero-pill">多页面路由</span>
            <span className="hero-pill">统一 API 拉取</span>
            <span className="hero-pill">平台控制台风格</span>
            <span className="hero-pill">论文截图友好</span>
          </div>
        </div>
        <div className="hero-side">
          <div className="signal">控制平面状态已联通</div>
          <div>
            <h3 className="section-title">页面拆分</h3>
            <div className="section-note">
              现在左侧菜单不再只是滚动定位，而是跳转到不同 URL 页面。这样更像真正的平台系统，也更适合后面继续扩展。
            </div>
          </div>
          <div className="footer-note mono">Dashboard URL: http://127.0.0.1:38080/</div>
        </div>
      </section>

      <section className="summary-strip page-shell">
        <div className="summary-card">
          <div className="title">模型数量</div>
          <div className="big">{derived.summary.model_count}</div>
          <div className="small">当前已登记的标准模型包数量</div>
        </div>
        <div className="summary-card">
          <div className="title">部署数量</div>
          <div className="big">{derived.summary.deployment_count}</div>
          <div className="small">累计 deployment 记录，包含历史演示与调试</div>
        </div>
        <div className="summary-card">
          <div className="title">压测数量</div>
          <div className="big">{derived.summary.benchmark_count}</div>
          <div className="small">可直接用作论文中的延迟与吞吐统计</div>
        </div>
        <div className="summary-card">
          <div className="title">运行状态</div>
          <div className="big">{escapeValue(derived.stopped.status || "unknown")}</div>
          <div className="small">最新 CasRel 样例 deployment 的最终状态</div>
        </div>
      </section>

      <section className="detail-grid page-shell">
        <article className="card detail-card">
          <h3 className="detail-title">平台运行摘要</h3>
          <div className="detail-note">这一块面向论文中的系统实现说明，展示当前模型库、部署记录、压测记录和页面挂载信息。</div>
          <div className="metrics">
            <MetricCard note="控制平面当前扫描到的模型资产总数" title="模型包" value={derived.summary.model_count} />
            <MetricCard note="部署记录数量" title="Deployment" value={derived.summary.deployment_count} />
            <MetricCard note="性能压测记录数量" title="Benchmark" value={derived.summary.benchmark_count} />
            <MetricCard note="当前 LuckyModels 控制台挂载地址" title="页面地址" value="127.0.0.1:38080" />
          </div>
          <div className="footer-note mono">
            artifacts: {derived.summary.artifact_root || "-"}
            <br />
            logs: {derived.summary.logs_root || "-"}
          </div>
        </article>

        <article className="card detail-card">
          <h3 className="detail-title">CasRel 快照</h3>
          <div className="detail-note">首页保留一块简要快照，详细调试信息和日志已经拆到独立页面中。</div>
          <div className="metrics">
            <MetricCard note="当前演示所使用的数据集" title="Dataset" value={derived.dataset.name || "-"} />
            <MetricCard note="最新一次基准实验记录编号" title="Benchmark ID" value={derived.benchmark.benchmark_id || "-"} />
            <MetricCard note="主样例中 runtime 实际输出的三元组数" title="Primary Triples" value={derived.runtime.triple_count ?? "-"} />
            <MetricCard note="当前导出的测试样本数量" title="Eval Cases" value={derived.dataset.prepared_eval_count ?? "-"} />
          </div>
        </article>
      </section>
    </>
  );
}

function ExperimentsPage({ demoRunning, derived, filters, loadStatus, onChangeFilters, onRefresh, onRunDemo }) {
  return (
    <>
      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Experiment Controls"
          note="聚焦 CasRel 演示操作、筛选条件和当前实验状态。"
          title="推理试验"
        />
        <div className="toolbar toolbar-tight">
          <div className="filters">
            <label className="field">
              <span>🔎</span>
              <input
                onChange={(event) => onChangeFilters((current) => ({ ...current, query: event.target.value }))}
                placeholder="按基准名称查询"
                type="text"
                value={filters.query}
              />
            </label>
            <label className="field">
              <span>🧠</span>
              <select
                onChange={(event) => onChangeFilters((current) => ({ ...current, model: event.target.value }))}
                value={filters.model}
              >
                <option value="">按模型过滤</option>
                {derived.modelOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>📦</span>
              <select
                onChange={(event) => onChangeFilters((current) => ({ ...current, profile: event.target.value }))}
                value={filters.profile}
              >
                <option value="">按 Profile 过滤</option>
                {derived.profileOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>●</span>
              <select
                onChange={(event) => onChangeFilters((current) => ({ ...current, status: event.target.value }))}
                value={filters.status}
              >
                <option value="">按状态过滤</option>
                <option value="completed">completed</option>
                <option value="running">running</option>
                <option value="stopped">stopped</option>
                <option value="failed">failed</option>
              </select>
            </label>
          </div>
          <div className="actions">
            <button onClick={() => void onRefresh()} type="button">
              刷新数据
            </button>
            <button className="btn-primary" disabled={demoRunning} onClick={() => void onRunDemo()} type="button">
              {demoRunning ? "运行中" : "运行 CasRel 样例"}
            </button>
          </div>
        </div>
        <div className="metrics">
          <MetricCard note="当前演示 report 状态" title="Report Status" value={derived.report.status || "empty"} />
          <MetricCard note="主样例名称" title="Primary Case" value={derived.primaryCase.case_name || "-"} />
          <MetricCard note="当前主样例输出三元组数量" title="Triples" value={derived.runtime.triple_count ?? "-"} />
          <MetricCard note="顶部状态提示" title="Load Status" value={loadStatus} />
        </div>
      </section>

      <section className="page-shell card section-shell">
        <SectionHeader
          kicker="Current Cases"
          note="保留当前实验使用的两个真实 WebNLG 样例，便于快速比对。"
          title="当前样例"
        />
        <div className="relation-grid">
          <div className="relation-box">
            <h4>Primary Case</h4>
            <div className="relation-item">
              名称：{derived.primaryCase.case_name || "-"}
              <br />
              句子：{derived.primaryCase.sentText || "-"}
              <br />
              triples：{derived.runtime.triple_count ?? "-"}
            </div>
          </div>
          <div className="relation-box">
            <h4>Secondary Case</h4>
            <div className="relation-item">
              名称：{derived.secondaryCase.case_name || "-"}
              <br />
              句子：{derived.secondaryCase.sentText || "-"}
              <br />
              triples：{derived.secondaryCase.runtime?.triple_count ?? "-"}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

function ArchivePage({ derived }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader
        kicker="Archive / Demo Outputs"
        note="收拢论文里最常引用的结果文件、当前数据来源和最新样例状态，方便截图和写表格。"
        title="结果归档"
      />
      <div className="metrics">
        <MetricCard note="当前 CasRel 演示报告状态" title="Report Status" value={derived.report.status || "empty"} />
        <MetricCard note="当前演示所使用的数据集" title="Dataset" value={derived.dataset.name || "-"} />
        <MetricCard note="当前导出的测试样本数量" title="Eval Cases" value={derived.dataset.prepared_eval_count ?? "-"} />
        <MetricCard note="主样例中 runtime 实际输出的三元组数" title="Primary Triples" value={derived.runtime.triple_count ?? "-"} />
        <MetricCard note="最新一次基准实验记录编号" title="Benchmark ID" value={derived.benchmark.benchmark_id || "-"} />
        <MetricCard note="当前样例运行时地址" title="Runtime URL" value={derived.deployment.service_url || "-"} />
      </div>
      <div className="footer-note mono">
        report: /api/v1/demo/casrel/report
        <br />
        dataset: {derived.dataset.source || "-"}
        <br />
        eval file: {derived.dataset.prepared_eval_file || "-"}
      </div>
    </section>
  );
}

function ModelsPage({ data, derived }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader
        kicker="Model Library"
        note="展示当前控制平面已识别到的标准模型包，用于说明 luckyModels 的模型登记能力。"
        title="模型库"
      />
      <div className="asset-grid">
        {derived.models.length ? (
          derived.models.map((model) => {
            const detail = data.modelDetails[model.model_id] || {};
            const manifest = detail.manifest || {};
            return (
              <article className="asset-card" key={model.model_id}>
                <h4>{model.model_name || model.model_id}</h4>
                <div className="asset-meta">
                  模型编号：<span className="mono">{model.model_id}</span>
                  <br />
                  家族：{model.model_family}
                  <br />
                  语言：{model.language}
                  <br />
                  版本：{model.version}
                  <br />
                  Runtime Adapter：{model.runtime_adapter}
                  <br />
                  任务类型：{manifest.task_type || "-"}
                </div>
                <div className="tag-list">
                  <span className="tag">{manifest.framework || "-"}</span>
                  <span className="tag">{manifest.backbone || "-"}</span>
                  <span className="tag">{model.package_root || "-"}</span>
                </div>
              </article>
            );
          })
        ) : (
          <EmptyState icon="M" title="暂无模型资产" wide />
        )}
      </div>
    </section>
  );
}

function ArtifactsPage({ data }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader
        kicker="Packages / Manifests"
        note="把 manifest 里的关键字段、ONNX 制品、tokenizer 和 profile 信息做成可读化展示。"
        title="制品包"
      />
      <div className="list-grid">
        {Object.values(data.modelDetails).length ? (
          Object.values(data.modelDetails).map((detail) => {
            const manifest = detail.manifest || {};
            return (
              <article className="list-card" key={detail.summary?.model_id || manifest.model_id}>
                <div className="list-card-header">
                  <div>
                    <h4>{manifest.model_name || detail.summary?.model_id || "-"}</h4>
                    <div className="mono">{manifest.model_id || detail.summary?.model_id || "-"}</div>
                  </div>
                  <StatusPill status={manifest.framework || "manifest"} />
                </div>
                <div className="list-card-body">
                  package_root：<span className="mono">{detail.summary?.package_root || "-"}</span>
                  <br />
                  tokenizer：{manifest.tokenizer?.type || "-"}
                  <br />
                  labels：{manifest.labels?.relation_map || "-"}
                  <br />
                  max_length：{manifest.inference?.max_length ?? "-"}
                  <br />
                  batch_size：{manifest.inference?.batch_size ?? "-"}
                </div>
                <div className="tag-list">
                  {Object.entries(detail.resolved_artifacts || {}).map(([name, path]) => (
                    <span className="tag" key={`${detail.summary?.model_id}-${name}`}>
                      {name}: {path}
                    </span>
                  ))}
                  {Object.keys(detail.resolved_profiles || {}).map((name) => (
                    <span className="tag" key={`${detail.summary?.model_id}-${name}-profile`}>
                      profile: {name}
                    </span>
                  ))}
                </div>
              </article>
            );
          })
        ) : (
          <EmptyState icon="P" note="当前还没有可展示的 manifest 详情。" title="暂无制品包信息" />
        )}
      </div>
    </section>
  );
}

function DeploymentsPage({ derived }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader
        kicker="Deployments"
        note="展示 runtime 启动记录、服务地址、日志路径和 deployment 状态，便于对部署过程做工程化说明。"
        title="部署记录"
      />
      <div className="table-wrap">
        {derived.deployments.length ? (
          <table>
            <thead>
              <tr>
                <th>Deployment</th>
                <th>模型</th>
                <th>Profile</th>
                <th>节点</th>
                <th>状态</th>
                <th>Service URL</th>
                <th>日志</th>
              </tr>
            </thead>
            <tbody>
              {[...derived.deployments].reverse().map((item) => (
                <tr key={item.deployment_id}>
                  <td>
                    <strong>{item.deployment_id}</strong>
                    <div className="mono">{item.notes || "-"}</div>
                  </td>
                  <td>{item.model_id}</td>
                  <td>{item.profile_name}</td>
                  <td>{item.node_name || "-"}</td>
                  <td><StatusPill status={item.status} /></td>
                  <td><div className="mono">{item.service_url || "-"}</div></td>
                  <td><div className="mono">{item.log_path || "-"}</div></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <EmptyState icon="D" title="暂无部署记录" />
        )}
      </div>
    </section>
  );
}

function BenchmarksPage({ derived, loadError, loadStatus }) {
  return (
    <section className="page-shell card table-shell">
      <div className="table-head">
        <div>
          <h2 className="table-title">部署与基准总览</h2>
          <div className="table-note">统一展示 LuckyModels 控制平面中生成的 benchmark 记录、吞吐量、延迟指标和 deployment 关联信息。</div>
        </div>
        <div className={`status ${loadStatus === "已加载" ? "ok" : loadStatus === "加载失败" ? "failed" : "partial_success"}`}>{loadStatus}</div>
      </div>
      <div className="table-wrap">
        {derived.filteredBenchmarks.length ? (
          <table>
            <thead>
              <tr>
                <th>名称</th>
                <th>模型名称</th>
                <th>Profile</th>
                <th>GPU</th>
                <th>状态</th>
                <th>总吞吐量</th>
                <th>平均延迟</th>
                <th>P95</th>
                <th>操作信息</th>
              </tr>
            </thead>
            <tbody>
              {[...derived.filteredBenchmarks].reverse().map((item) => {
                const deploymentRecord = derived.deployments.find((dep) => dep.deployment_id === item.deployment_id) || {};
                return (
                  <tr key={item.benchmark_id}>
                    <td>
                      <strong>{item.label || item.benchmark_id}</strong>
                      <div className="mono">{item.benchmark_id}</div>
                    </td>
                    <td>{item.model_id}</td>
                    <td>{item.profile_name}</td>
                    <td>{formatArray(deploymentRecord.gpu_ids)}</td>
                    <td><StatusPill status={item.status} /></td>
                    <td>{escapeValue(item.throughput_qps)}</td>
                    <td>{escapeValue(item.avg_latency_ms)}</td>
                    <td>{escapeValue(item.p95_latency_ms)}</td>
                    <td><div className="mono">{item.deployment_id}</div></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <EmptyState
            icon="☰"
            note={loadError || "还没有生成 benchmark 记录。点击“推理试验”页的按钮可自动运行一轮 CasRel 样例并生成数据。"}
            title="暂无基准实验"
          />
        )}
      </div>
    </section>
  );
}

function ResourcesPage({ derived }) {
  return (
    <section className="page-shell">
      <div className="resource-grid">
        <article className="resource-card">
          <h4>计算节点</h4>
          <div className="resource-meta">
            节点名称：{derived.deployment.node_name || "local-node"}
            <br />
            运行模式：本机控制平面 + 本地 runtime 进程
            <br />
            服务地址：<span className="mono">{derived.deployment.service_url || "-"}</span>
          </div>
        </article>
        <article className="resource-card">
          <h4>GPU 资源</h4>
          <div className="resource-meta">
            当前演示 deployment 使用 GPU 数量：{(derived.deployment.gpu_ids || []).length || 0}
            <br />
            当前 provider：{formatArray(derived.healthPayload.providers)}
            <br />
            备注：这一版 CasRel 演示主要用于验证 ONNX runtime 接入链路。
          </div>
        </article>
        <article className="resource-card">
          <h4>运行时后端</h4>
          <div className="resource-meta">
            adapter：{derived.healthPayload.adapter || "-"}
            <br />
            backend：{derived.healthPayload.session_backend || "-"}
            <br />
            sessions_loaded：{escapeValue(derived.healthPayload.sessions_loaded)}
            <br />
            model_id：{derived.healthPayload.model_id || "-"}
          </div>
        </article>
      </div>
    </section>
  );
}

function LogsPage({ loadError, report }) {
  return (
    <section className="page-shell card section-shell">
      <SectionHeader
        kicker="Runtime Logs"
        note="保留最新 CasRel runtime 的启动日志和推理访问记录，方便你展示“平台已真正跑通”。"
        title="日志中心"
      />
      <div className="logs">{report.logs?.content || loadError || "等待演示报告。"}</div>
    </section>
  );
}
