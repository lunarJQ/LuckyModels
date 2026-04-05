import { navGroups } from "../nav";

export function formatRoleLabel(roleId) {
  const mapping = {
    admin: "平台管理员",
    user: "普通用户",
  };
  return mapping[roleId] || roleId || "-";
}

export function formatPermissionLabel(permissionId) {
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

export function getRoleScopeSummary(roleId) {
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

export function hasRequiredPermissions(currentPermissions, requiredPermissions = []) {
  if (!requiredPermissions.length) {
    return true;
  }
  const permissionSet = new Set(currentPermissions || []);
  return requiredPermissions.every((permissionId) => permissionSet.has(permissionId));
}

export function getRequiredPermissionsForPath(pathname) {
  if (pathname === "/dashboard") {
    return pageConfigMap["/"]?.requiredPermissions || [];
  }
  return pageConfigMap[pathname]?.requiredPermissions || [];
}

export function isAdminOnlyPath(pathname) {
  if (pathname === "/dashboard") {
    return Boolean(pageConfigMap["/"]?.adminOnly);
  }
  return Boolean(pageConfigMap[pathname]?.adminOnly);
}

export function canAccessPath(pathname, currentPermissions, currentUser) {
  const hasPermissions = hasRequiredPermissions(currentPermissions, getRequiredPermissionsForPath(pathname));
  if (!hasPermissions) {
    return false;
  }
  if (isAdminOnlyPath(pathname) && !currentUser?.is_admin) {
    return false;
  }
  return true;
}

export function getVisibleNavGroups(currentPermissions, currentUser) {
  return navGroups
    .map((group) => ({
      ...group,
      items: group.items.filter((item) => canAccessPath(item.path, currentPermissions, currentUser)),
    }))
    .filter((group) => group.items.length);
}

export function getFallbackPath(currentPermissions, currentUser) {
  if (canAccessPath("/", currentPermissions, currentUser)) {
    return "/";
  }
  const visibleGroups = getVisibleNavGroups(currentPermissions, currentUser);
  return visibleGroups[0]?.items[0]?.path || "/";
}
