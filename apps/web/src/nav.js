export const navGroups = [
  {
    title: "工作台",
    items: [
      { id: "overview", path: "/", label: "资产概览" },
      { id: "experiment", path: "/experiments", label: "推理试验" },
      { id: "task-routing", path: "/task-routing", label: "任务调度" },
      { id: "deployment-policy", path: "/deployment-policy", label: "部署策略" },
      { id: "archive", path: "/archive", label: "结果归档" },
    ],
  },
  {
    title: "模型资产",
    items: [
      { id: "models", path: "/models", label: "模型库" },
      { id: "capability-matrix", path: "/capability-matrix", label: "能力矩阵" },
      { id: "artifacts", path: "/artifacts", label: "制品包" },
      { id: "deployments", path: "/deployments", label: "部署记录" },
      { id: "benchmarks", path: "/benchmarks", label: "基准剖析" },
    ],
  },
  {
    title: "运行资源",
    items: [
      { id: "resources", path: "/resources", label: "运行资源" },
      { id: "logs", path: "/logs", label: "日志中心" },
    ],
  },
  {
    title: "系统管理",
    items: [
      { id: "access-control", path: "/access-control", label: "用户权限" },
      { id: "admin-console", path: "/admin-console", label: "管理员面板" },
    ],
  },
];
