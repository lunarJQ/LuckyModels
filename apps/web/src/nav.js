export const navGroups = [
  {
    title: "工作台",
    items: [
      { id: "overview", path: "/", label: "资产概览", icon: "◆", description: "平台总览与系统态势" },
      { id: "experiment", path: "/experiments", label: "推理试验", icon: "◉", description: "在线推理与结果归档", requiredPermissions: ["predictions.view"] },
      { id: "task-routing", path: "/task-routing", label: "任务调度", icon: "⇄", description: "多模型策略执行", requiredPermissions: ["strategies.manage"] },
      { id: "deployment-policy", path: "/deployment-policy", label: "部署策略", icon: "△", description: "资源建议与部署方案", requiredPermissions: ["strategies.manage"] },
      { id: "archive", path: "/archive", label: "结果归档", icon: "▣", description: "预测与实验沉淀", requiredPermissions: ["predictions.view"] },
    ],
  },
  {
    title: "模型资产",
    items: [
      { id: "models", path: "/models", label: "模型库", icon: "□", description: "模型信息与部署入口", requiredPermissions: ["models.view"] },
      { id: "capability-matrix", path: "/capability-matrix", label: "能力矩阵", icon: "≡", description: "模型能力总表", requiredPermissions: ["models.view"] },
      { id: "artifacts", path: "/artifacts", label: "制品包", icon: "◫", description: "Manifest 与制品清单", requiredPermissions: ["models.view"] },
      { id: "deployments", path: "/deployments", label: "部署记录", icon: "▶", description: "服务实例与运行状态", requiredPermissions: ["deployments.view"] },
      { id: "benchmarks", path: "/benchmarks", label: "基准剖析", icon: "∥", description: "性能结果与对比", requiredPermissions: ["benchmarks.view"] },
    ],
  },
  {
    title: "运行资源",
    items: [
      { id: "resources", path: "/resources", label: "运行资源", icon: "⌁", description: "GPU 与运行快照", requiredPermissions: ["resources.view"] },
      { id: "logs", path: "/logs", label: "日志中心", icon: "⋯", description: "部署日志与排障", requiredPermissions: ["resources.view"] },
    ],
  },
  {
    title: "系统管理",
    items: [
      { id: "access-control", path: "/access-control", label: "用户权限", icon: "☰", description: "角色与权限总览", requiredPermissions: ["users.view"] },
      { id: "admin-console", path: "/admin-console", label: "管理员面板", icon: "★", description: "账号治理与密码重置", requiredPermissions: ["users.manage"] },
    ],
  },
];
