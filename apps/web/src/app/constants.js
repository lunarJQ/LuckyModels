export const initialData = {
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

export const initialLoadProgress = {
  phase: "idle",
  percent: 0,
  label: "等待加载",
  detail: "",
};

export const DEFAULT_PREDICT_TEXT = "LuckyModels provides a stable fallback sample for relation extraction.";

export const DEFAULT_PREDICT_DATASETS = [
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

export const BENCHMARK_SCENE_OPTIONS = [
  { value: "en_general", label: "英文通用场景" },
  { value: "zh_general", label: "中文通用场景" },
  { value: "zh_industrial", label: "中文工业设备场景" },
];

export const pageMeta = {
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
