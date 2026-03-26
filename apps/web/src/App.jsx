import { startTransition, useDeferredValue, useEffect, useRef, useState } from "react";

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
  const [activeSection, setActiveSection] = useState("overview-section");
  const deferredQuery = useDeferredValue(filters.query);
  const sectionRefs = useRef(new Map());

  useEffect(() => {
    void refreshAll();
  }, []);

  useEffect(() => {
    const sections = [...sectionRefs.current.values()].filter(Boolean);
    if (!sections.length || !("IntersectionObserver" in window)) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        if (visible?.target?.id) {
          setActiveSection(visible.target.id);
        }
      },
      {
        rootMargin: "-20% 0px -65% 0px",
        threshold: [0.15, 0.35, 0.6],
      },
    );

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [data.summary, data.models.length, data.benchmarks.length]);

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

  function setSectionRef(id) {
    return (node) => {
      if (node) {
        sectionRefs.current.set(id, node);
      } else {
        sectionRefs.current.delete(id);
      }
    };
  }

  function scrollToSection(id) {
    const target = sectionRefs.current.get(id) || document.getElementById(id);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveSection(id);
  }

  const summary = data.summary || { model_count: 0, deployment_count: 0, benchmark_count: 0 };
  const report = data.report || {};
  const dataset = report.dataset || {};
  const benchmark = report.benchmark || {};
  const deployment = report.deployment || {};
  const stopped = report.deployment_stopped || {};
  const primaryCase = report.samples?.primary_case || {};
  const secondaryCase = report.samples?.secondary_case || {};
  const runtime = primaryCase.runtime || {};
  const debug = runtime.debug || {};
  const warnings = runtime.warnings || [];
  const healthPayload = safeJsonParse(report.health?.payload || "{}");
  const models = data.models || [];
  const deployments = data.deployments || [];
  const benchmarks = data.benchmarks || [];

  const filteredBenchmarks = benchmarks.filter((item) => {
    const blob = `${item.benchmark_id} ${item.label || ""} ${item.model_id || ""} ${item.profile_name || ""}`.toLowerCase();
    if (deferredQuery && !blob.includes(deferredQuery.trim().toLowerCase())) return false;
    if (filters.model && item.model_id !== filters.model) return false;
    if (filters.profile && item.profile_name !== filters.profile) return false;
    if (filters.status && item.status !== filters.status) return false;
    return true;
  });

  const modelOptions = [...new Set(benchmarks.map((item) => item.model_id).filter(Boolean))];
  const profileOptions = [...new Set(benchmarks.map((item) => item.profile_name).filter(Boolean))];

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
            基于 React 重构的 LuckyModels 前端控制台，围绕模型资产、部署验证、运行调试和基准剖析组织平台视图。
          </div>
        </div>

        {navGroups.map((group) => (
          <div className="nav-group" key={group.title}>
            <div className="nav-title">{group.title}</div>
            {group.items.map((item) => (
              <button
                className={`nav-item ${activeSection === item.id ? "active" : ""}`}
                key={item.id}
                onClick={() => scrollToSection(item.id)}
                type="button"
              >
                {item.label}
              </button>
            ))}
          </div>
        ))}

        <div className="nav-group">
          <div className="nav-title">当前状态</div>
          <div className="footer-note">
            模型 {summary.model_count} · 部署 {summary.deployment_count} · 基准 {summary.benchmark_count}
          </div>
          <div className="footer-note mono">/api/v1/demo/casrel/report</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>LuckyModels React 控制台</h1>
            <div className="topbar-note">
              面向知识抽取模型库落地的组件化前端，展示 CasRel 演示、部署记录、制品信息和性能数据。
            </div>
          </div>
          <div className="topbar-meta">
            <span>react-dashboard v0.2.0</span>
            <div className="top-icon">R</div>
            <div className="top-icon">UI</div>
          </div>
        </header>

        <section className="content">
          <section className="hero content-section" id="overview-section" ref={setSectionRef("overview-section")}>
            <div className="hero-card">
              <div className="hero-kicker">LuckyModels / React Control Plane</div>
              <h2>模型资产、部署验证与基准实验的统一前端</h2>
              <div className="hero-desc">
                这版页面不再是单个内联 HTML 演示，而是一个组件化 React 前端。后面继续加模型详情页、部署详情页、结果对比页和图表会顺手很多。
              </div>
              <div className="hero-points">
                <span className="hero-pill">React 组件化</span>
                <span className="hero-pill">统一 API 拉取</span>
                <span className="hero-pill">可点击侧边导航</span>
                <span className="hero-pill">论文截图友好</span>
              </div>
            </div>
            <div className="hero-side">
              <div className="signal">控制平面状态已联通</div>
              <div>
                <h3 className="section-title">演示用途</h3>
                <div className="section-note">
                  你可以直接用这一页做平台截图，也可以点击按钮重新生成 CasRel 演示报告，刷新最新 benchmark 和 debug 数据。
                </div>
              </div>
              <div className="footer-note mono">Dashboard URL: http://127.0.0.1:38080/</div>
            </div>
          </section>

          <section className="content-section" id="experiment-section" ref={setSectionRef("experiment-section")}>
            <div className="toolbar">
              <div className="filters">
                <label className="field">
                  <span>🔎</span>
                  <input
                    onChange={(event) => setFilters((current) => ({ ...current, query: event.target.value }))}
                    placeholder="按基准名称查询"
                    type="text"
                    value={filters.query}
                  />
                </label>
                <label className="field">
                  <span>🧠</span>
                  <select
                    onChange={(event) => setFilters((current) => ({ ...current, model: event.target.value }))}
                    value={filters.model}
                  >
                    <option value="">按模型过滤</option>
                    {modelOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>📦</span>
                  <select
                    onChange={(event) => setFilters((current) => ({ ...current, profile: event.target.value }))}
                    value={filters.profile}
                  >
                    <option value="">按 Profile 过滤</option>
                    {profileOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>●</span>
                  <select
                    onChange={(event) => setFilters((current) => ({ ...current, status: event.target.value }))}
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
                <button onClick={() => void refreshAll()} type="button">
                  刷新数据
                </button>
                <button className="btn-primary" disabled={demoRunning} onClick={() => void handleRunDemo()} type="button">
                  {demoRunning ? "运行中" : "运行 CasRel 样例"}
                </button>
              </div>
            </div>

            <section className="summary-strip">
              <div className="summary-card">
                <div className="title">模型数量</div>
                <div className="big">{summary.model_count}</div>
                <div className="small">当前已登记的标准模型包数量</div>
              </div>
              <div className="summary-card">
                <div className="title">部署数量</div>
                <div className="big">{summary.deployment_count}</div>
                <div className="small">累计 deployment 记录，包含历史演示与调试</div>
              </div>
              <div className="summary-card">
                <div className="title">压测数量</div>
                <div className="big">{summary.benchmark_count}</div>
                <div className="small">可直接用作论文中的延迟与吞吐统计</div>
              </div>
              <div className="summary-card">
                <div className="title">运行状态</div>
                <div className="big">{escapeValue(stopped.status || "unknown")}</div>
                <div className="small">最新 CasRel 样例 deployment 的最终状态</div>
              </div>
            </section>
          </section>

          <section className="card section-shell content-section" id="archive-section" ref={setSectionRef("archive-section")}>
            <SectionHeader
              kicker="Archive / Demo Outputs"
              note="收拢论文里最常引用的结果文件、当前数据来源和最新样例状态，方便截图和写表格。"
              title="结果归档"
            />
            <div className="metrics">
              <MetricCard note="当前 CasRel 演示报告状态" title="Report Status" value={report.status || "empty"} />
              <MetricCard note="当前演示所使用的数据集" title="Dataset" value={dataset.name || "-"} />
              <MetricCard note="当前导出的测试样本数量" title="Eval Cases" value={dataset.prepared_eval_count ?? "-"} />
              <MetricCard note="主样例中 runtime 实际输出的三元组数" title="Primary Triples" value={runtime.triple_count ?? "-"} />
              <MetricCard note="最新一次基准实验记录编号" title="Benchmark ID" value={benchmark.benchmark_id || "-"} />
              <MetricCard note="当前主展示样例名称" title="Sample Sentence" value={primaryCase.case_name || "-"} />
            </div>
            <div className="footer-note mono">
              report: /api/v1/demo/casrel/report
              <br />
              dataset: {dataset.source || "-"}
              <br />
              eval file: {dataset.prepared_eval_file || "-"}
            </div>
          </section>

          <section className="card section-shell content-section" id="models-section" ref={setSectionRef("models-section")}>
            <SectionHeader
              kicker="Model Library"
              note="展示当前控制平面已识别到的标准模型包，用于说明 luckyModels 的模型登记能力。"
              title="模型库"
            />
            <div className="asset-grid">
              {models.length ? (
                models.map((model) => {
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

          <section className="card section-shell content-section" id="artifacts-section" ref={setSectionRef("artifacts-section")}>
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

          <section className="card section-shell content-section" id="deployments-section" ref={setSectionRef("deployments-section")}>
            <SectionHeader
              kicker="Deployments"
              note="展示 runtime 启动记录、服务地址、日志路径和 deployment 状态，便于对部署过程做工程化说明。"
              title="部署记录"
            />
            <div className="table-wrap">
              {deployments.length ? (
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
                    {[...deployments].reverse().map((item) => (
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

          <section className="card table-shell content-section" id="benchmarks-section" ref={setSectionRef("benchmarks-section")}>
            <div className="table-head">
              <div>
                <h2 className="table-title">部署与基准总览</h2>
                <div className="table-note">统一展示 LuckyModels 控制平面中生成的 benchmark 记录、吞吐量、延迟指标和 deployment 关联信息。</div>
              </div>
              <div className={`status ${loadStatus === "已加载" ? "ok" : loadStatus === "加载失败" ? "failed" : "partial_success"}`}>{loadStatus}</div>
            </div>
            <div className="table-wrap">
              {filteredBenchmarks.length ? (
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
                    {[...filteredBenchmarks].reverse().map((item) => {
                      const deploymentRecord = deployments.find((dep) => dep.deployment_id === item.deployment_id) || {};
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
                  note="还没有生成 benchmark 记录。点击上方按钮可自动运行一轮 CasRel 样例并生成数据。"
                  title="暂无基准实验"
                />
              )}
            </div>
          </section>

          <section className="detail-grid content-section" id="nodes-section" ref={setSectionRef("nodes-section")}>
            <article className="card detail-card">
              <h3 className="detail-title">平台运行摘要</h3>
              <div className="detail-note">这一块面向论文中的系统实现说明，展示当前模型库、部署记录、压测记录和页面挂载信息。</div>
              <div className="metrics">
                <MetricCard note="控制平面当前扫描到的模型资产总数" title="模型包" value={summary.model_count} />
                <MetricCard note="部署记录数量" title="Deployment" value={summary.deployment_count} />
                <MetricCard note="性能压测记录数量" title="Benchmark" value={summary.benchmark_count} />
                <MetricCard note="当前 LuckyModels 控制台挂载地址" title="页面地址" value="127.0.0.1:38080" />
              </div>
              <div className="footer-note mono">
                artifacts: {summary.artifact_root || "-"}
                <br />
                logs: {summary.logs_root || "-"}
              </div>
            </article>

            <article className="card detail-card" id="runtime-section" ref={setSectionRef("runtime-section")}>
              <h3 className="detail-title">CasRel 调试报告</h3>
              <div className="detail-note">演示接口会自动创建 deployment、运行 benchmark，并返回当前制品的 token 化与分数分布，用于解释抽取效果现状。</div>
              <div className="metrics">
                <MetricCard note="CasRel 样例自动创建的 deployment" title="Deployment" value={deployment.deployment_id || "-"} />
                <MetricCard note="平均延迟 / ms" title="Avg Latency" value={benchmark.avg_latency_ms ?? "-"} />
                <MetricCard note="P95 延迟 / ms" title="P95" value={benchmark.p95_latency_ms ?? "-"} />
                <MetricCard note="总吞吐量 / qps" title="QPS" value={benchmark.throughput_qps ?? "-"} />
                <MetricCard note="已生成的 WebNLG 测试样本数量" title="Eval Cases" value={dataset.prepared_eval_count ?? "-"} />
                <MetricCard note="样例运行时地址" title="Runtime URL" value={deployment.service_url || "-"} />
              </div>
              <div className="warn-list">
                {warnings.map((item) => (
                  <span className="pill warn" key={item}>
                    {item}
                  </span>
                ))}
              </div>
              <div className="score-block">
                <div className="detail-note">{primaryCase.sentText || runtime.text || "-"}</div>
                <div className="detail-note">
                  数据集来源：{dataset.source || "-"}；测试样本导出文件：{dataset.prepared_eval_file || "-"}；选中的演示样例：
                  {" "}
                  {(dataset.selected_cases || []).join(", ") || "-"}。
                </div>
                <div className="relation-grid">
                  <div className="relation-box">
                    <h4>Gold Relations</h4>
                    <div className="relation-list">
                      {(primaryCase.gold_relations || []).length ? (
                        primaryCase.gold_relations.map((item, index) => (
                          <div className="relation-item" key={`gold-${index + 1}`}>
                            <strong>{item.em1Text}</strong>
                            <span> -- {item.label} -- </span>
                            <strong>{item.em2Text}</strong>
                          </div>
                        ))
                      ) : (
                        <div className="relation-item">暂无 gold relations</div>
                      )}
                    </div>
                  </div>
                  <div className="relation-box">
                    <h4>Runtime Triples</h4>
                    <div className="relation-list">
                      {(runtime.triples || []).length ? (
                        runtime.triples.map((item, index) => (
                          <div className="relation-item" key={`triple-${index + 1}`}>
                            <strong>{item.subject}</strong>
                            <span> -- {item.predicate} -- </span>
                            <strong>{item.object}</strong>
                            <br />
                            <span className="mono">confidence={item.confidence}</span>
                          </div>
                        ))
                      ) : (
                        <div className="relation-item">当前样例未输出三元组</div>
                      )}
                    </div>
                  </div>
                </div>
                <div className="relation-grid">
                  <div className="relation-box">
                    <h4>Primary Case</h4>
                    <div className="relation-item">
                      名称：{primaryCase.case_name || "-"}
                      <br />
                      triples：{runtime.triple_count ?? "-"}
                    </div>
                  </div>
                  <div className="relation-box">
                    <h4>Secondary Case</h4>
                    <div className="relation-item">
                      名称：{secondaryCase.case_name || "-"}
                      <br />
                      triples：{secondaryCase.runtime?.triple_count ?? "-"}
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </section>

          <section className="content-section" id="gpu-section" ref={setSectionRef("gpu-section")}>
            <div className="resource-grid">
              <article className="resource-card">
                <h4>计算节点</h4>
                <div className="resource-meta">
                  节点名称：{deployment.node_name || "local-node"}
                  <br />
                  运行模式：本机控制平面 + 本地 runtime 进程
                  <br />
                  服务地址：<span className="mono">{deployment.service_url || "-"}</span>
                </div>
              </article>
              <article className="resource-card">
                <h4>GPU 资源</h4>
                <div className="resource-meta">
                  当前演示 deployment 使用 GPU 数量：{(deployment.gpu_ids || []).length || 0}
                  <br />
                  当前 provider：{formatArray(healthPayload.providers)}
                  <br />
                  备注：这一版 CasRel 演示主要用于验证 ONNX runtime 接入链路。
                </div>
              </article>
              <article className="resource-card">
                <h4>运行时后端</h4>
                <div className="resource-meta">
                  adapter：{healthPayload.adapter || "-"}
                  <br />
                  backend：{healthPayload.session_backend || "-"}
                  <br />
                  sessions_loaded：{escapeValue(healthPayload.sessions_loaded)}
                  <br />
                  model_id：{healthPayload.model_id || "-"}
                </div>
              </article>
            </div>
          </section>

          <section className="card section-shell content-section" id="logs-section" ref={setSectionRef("logs-section")}>
            <SectionHeader
              kicker="Runtime Logs"
              note="保留最新 CasRel runtime 的启动日志和推理访问记录，方便你展示“平台已真正跑通”。"
              title="日志中心"
            />
            <div className="logs">{report.logs?.content || loadError || "等待演示报告。"}</div>
          </section>
        </section>
      </main>
    </div>
  );
}
