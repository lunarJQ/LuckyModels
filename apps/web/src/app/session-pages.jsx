function hasCurrentPermission(currentPermissions, permissionId) {
  return new Set(currentPermissions || []).has(permissionId);
}

function buildBootstrapStageItems(currentPermissions, currentUser, currentPhase) {
  const canViewModels = hasCurrentPermission(currentPermissions, "models.view");
  const canViewDeployments = hasCurrentPermission(currentPermissions, "deployments.view");
  const canViewBenchmarks = hasCurrentPermission(currentPermissions, "benchmarks.view");
  const canViewPredictions = hasCurrentPermission(currentPermissions, "predictions.view");
  const canRunTaskRouting = hasCurrentPermission(currentPermissions, "task_routing.run");
  const canViewResources = hasCurrentPermission(currentPermissions, "resources.view");
  const canViewUsers = hasCurrentPermission(currentPermissions, "users.view");
  const isAdmin = Boolean(currentUser?.is_admin);

  return [
    {
      key: "base",
      title: "基础清单",
      note: isAdmin
        ? "平台概览、模型、部署、基准、结果反馈与治理入口"
        : [
          "平台概览",
          canViewModels ? "模型资产" : "",
          canViewDeployments ? "部署实例" : "",
          canViewBenchmarks ? "基准记录" : "",
          canViewPredictions ? "结果反馈" : "",
          canRunTaskRouting ? "任务调度" : "",
        ].filter(Boolean).join("、") || "当前身份可见基础数据",
      active: ["bootstrap", "base", "base-done", "deployment-health", "model-details", "ready"].includes(currentPhase),
      done: ["base-done", "deployment-health", "model-details", "ready"].includes(currentPhase),
    },
    {
      key: "deployment-health",
      title: "运行状态",
      note: canViewDeployments
        ? canViewResources
          ? "部署健康、服务可达性与资源相关状态补充"
          : "部署健康与服务可达性补充"
        : "当前身份无需补充部署健康详情",
      active: ["deployment-health", "model-details", "ready"].includes(currentPhase),
      done: ["model-details", "ready"].includes(currentPhase),
    },
    {
      key: "model-details",
      title: "模型详情",
      note: canViewModels
        ? canViewUsers
          ? "模型能力、配置模板、限制信息与权限收口整理"
          : "模型能力、配置模板与限制信息整理"
        : "当前身份无需读取模型详情，直接进入主页面整理",
      active: ["model-details", "ready"].includes(currentPhase),
      done: ["ready"].includes(currentPhase),
    },
  ];
}

export function LoginPage({ actionBusy, authChecking, loadError, loginForm, notice, onLogin, onLoginFormChange }) {
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

export function BootstrapLoadingPage({
  currentPermissions = [],
  currentUser,
  loadStatus,
  loadProgress,
  loadError = "",
  onRetry = null,
}) {
  const isFailed = Boolean(loadError);
  const progressPercent = Math.max(0, Math.min(100, Number(loadProgress?.percent) || 0));
  const stageItems = buildBootstrapStageItems(currentPermissions, currentUser, loadProgress?.phase);
  const userFacingSummary = currentUser?.is_admin
    ? "管理员视角会同步更多治理与资源信息，因此首轮初始化通常更完整。"
    : "当前按普通用户可见范围初始化页面，不会额外加载你无权访问的治理与资源模块。";

  return (
    <main className="bootstrap-shell">
      <section className="bootstrap-stage">
        <article className="bootstrap-hero card">
          <div className="bootstrap-kicker">LuckyModels Control Center</div>
          <h1>{isFailed ? "控制台初始化失败" : "正在进入控制台"}</h1>
          <p className="bootstrap-desc">
            {isFailed
              ? "平台登录已经完成，但首轮数据还没有成功加载。可以直接重试，不会影响当前登录状态。"
              : "正在同步模型、部署、基准、结果反馈与运行资源信息。首轮数据尚未就绪前，页面不会再先显示一组误导性的 0 值。"}
          </p>
          <div className="bootstrap-pill-row">
            <span className="hero-pill">当前用户：{currentUser?.display_name || currentUser?.username || "已登录"}</span>
            <span className="hero-pill">状态：{loadStatus || (isFailed ? "加载失败" : "正在加载")}</span>
            <span className="hero-pill">{isFailed ? "等待重试" : `进度：${progressPercent}%`}</span>
          </div>
          <div className="bootstrap-progress">
            <div className="bootstrap-progress-track">
              <div className="bootstrap-progress-fill" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="bootstrap-progress-meta">
              <strong>{loadProgress?.label || (isFailed ? "控制台初始化失败" : "正在同步真实数据")}</strong>
              <span>{loadProgress?.detail || (isFailed ? "请重试后重新进入主页面。" : "请稍候，平台正在拉取首轮真实数据。")}</span>
              <span>{userFacingSummary}</span>
            </div>
          </div>
          {loadError ? <div className="banner banner-error">{loadError}</div> : null}
          <div className="bootstrap-actions">
            {onRetry ? (
              <button className="btn-primary" onClick={onRetry} type="button">
                重新加载
              </button>
            ) : (
              <div className="bootstrap-inline-note">首次加载通常会稍慢一些，等所有核心接口返回后再进入主页面。</div>
            )}
          </div>
        </article>

        <section className="bootstrap-stage-grid">
          {stageItems.map((item) => (
            <article className={`bootstrap-stage-card card ${item.active ? "is-active" : ""} ${item.done ? "is-done" : ""}`} key={item.key}>
              <div className="bootstrap-stage-badge">{item.done ? "已完成" : item.active ? "进行中" : "等待中"}</div>
              <strong>{item.title}</strong>
              <span>{item.note}</span>
            </article>
          ))}
        </section>

        <section className="bootstrap-summary-grid">
          {Array.from({ length: 4 }).map((_, index) => (
            <article className="bootstrap-summary-card card" key={`bootstrap-summary-${index}`}>
              <div className="bootstrap-skeleton bootstrap-skeleton-title" />
              <div className="bootstrap-skeleton bootstrap-skeleton-value" />
              <div className="bootstrap-skeleton bootstrap-skeleton-note" />
            </article>
          ))}
        </section>

        <section className="bootstrap-panel-grid">
          <article className="bootstrap-panel card">
            <div className="bootstrap-panel-head">
              <div className="bootstrap-skeleton bootstrap-skeleton-title wide" />
              <div className="bootstrap-skeleton bootstrap-skeleton-tag" />
            </div>
            <div className="bootstrap-row-list">
              {Array.from({ length: 5 }).map((_, index) => (
                <div className="bootstrap-row" key={`bootstrap-row-left-${index}`}>
                  <div className="bootstrap-skeleton bootstrap-skeleton-line" />
                  <div className="bootstrap-skeleton bootstrap-skeleton-line short" />
                </div>
              ))}
            </div>
          </article>

          <article className="bootstrap-panel card">
            <div className="bootstrap-panel-head">
              <div className="bootstrap-skeleton bootstrap-skeleton-title wide" />
              <div className="bootstrap-skeleton bootstrap-skeleton-tag" />
            </div>
            <div className="bootstrap-chart-area">
              <div className="bootstrap-skeleton bootstrap-skeleton-chart" />
            </div>
            <div className="bootstrap-row-list compact">
              {Array.from({ length: 3 }).map((_, index) => (
                <div className="bootstrap-row" key={`bootstrap-row-right-${index}`}>
                  <div className="bootstrap-skeleton bootstrap-skeleton-line" />
                  <div className="bootstrap-skeleton bootstrap-skeleton-line short" />
                </div>
              ))}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}

export function GuardedPage({ allowed, children }) {
  return allowed ? children : null;
}
