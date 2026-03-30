from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .schemas import (
    AccessCurrentUserSwitchRequest,
    AccessOverview,
    AccessPermission,
    AccessRole,
    AccessUser,
    AccessUserCreateRequest,
    AccessUserPasswordResetResponse,
    AccessUserPasswordResetRequest,
    AccessUserToggleRequest,
    AccessUserUpdateRequest,
    AuthLoginRequest,
    AuthSessionResponse,
    utc_now,
)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


AUTH_VERSION = 2

BUILTIN_PERMISSIONS = [
    AccessPermission(
        permission_id="models.view",
        permission_name="查看模型资产",
        category="模型资产",
        description="查看模型库、能力矩阵、制品包与模型详情。",
    ),
    AccessPermission(
        permission_id="deployments.view",
        permission_name="查看部署记录",
        category="部署运行",
        description="查看部署列表、运行状态、健康检查和日志。",
    ),
    AccessPermission(
        permission_id="deployments.manage",
        permission_name="管理部署实例",
        category="部署运行",
        description="创建、重启、停止和删除部署实例。",
    ),
    AccessPermission(
        permission_id="benchmarks.view",
        permission_name="查看基准结果",
        category="实验评测",
        description="查看 benchmark 记录、聚合结果和论文视图。",
    ),
    AccessPermission(
        permission_id="benchmarks.manage",
        permission_name="运行基准测试",
        category="实验评测",
        description="创建和删除 benchmark，调整实验参数。",
    ),
    AccessPermission(
        permission_id="predictions.view",
        permission_name="查看预测归档",
        category="预测归档",
        description="查看在线预测结果和历史归档。",
    ),
    AccessPermission(
        permission_id="strategies.manage",
        permission_name="运行策略分析",
        category="策略能力",
        description="运行任务调度与部署策略分析能力。",
    ),
    AccessPermission(
        permission_id="resources.view",
        permission_name="查看运行资源",
        category="系统资源",
        description="查看 GPU、日志和运行资源快照。",
    ),
    AccessPermission(
        permission_id="archive.export",
        permission_name="导出归档结果",
        category="预测归档",
        description="导出预测、benchmark 和论文视图数据。",
    ),
    AccessPermission(
        permission_id="users.view",
        permission_name="查看用户权限",
        category="系统管理",
        description="查看用户、角色和权限总览。",
    ),
    AccessPermission(
        permission_id="users.manage",
        permission_name="管理用户权限",
        category="系统管理",
        description="创建、更新、停用用户并调整角色绑定。",
    ),
]

BUILTIN_ROLES = [
    AccessRole(
        role_id="admin",
        role_name="平台管理员",
        description="拥有全部权限，负责用户、部署、实验与系统治理。",
        permissions=[item.permission_id for item in BUILTIN_PERMISSIONS],
    ),
    AccessRole(
        role_id="user",
        role_name="普通用户",
        description="拥有日常查看、在线推理和结果导出能力，但不允许管理员操作。",
        permissions=[
            "models.view",
            "deployments.view",
            "benchmarks.view",
            "predictions.view",
            "archive.export",
        ],
    ),
]

DEFAULT_USERS = [
    AccessUser(
        user_id="user-admin",
        username="admin",
        display_name="平台管理员",
        email="admin@luckymodels.local",
        role_ids=["admin"],
        is_active=True,
        is_admin=True,
        notes="默认管理员账号，用于控制台初始化和权限配置。",
        password_hash=_hash_password("admin"),
    ),
    AccessUser(
        user_id="user-default",
        username="user",
        display_name="普通用户",
        email="user@luckymodels.local",
        role_ids=["user"],
        is_active=True,
        is_admin=False,
        notes="默认普通用户账号，用于登录与权限分层演示。",
        password_hash=_hash_password("user"),
    ),
]


class AccessStore:
    def __init__(self, access_control_file: Path) -> None:
        self.access_control_file = access_control_file
        self.access_control_file.parent.mkdir(parents=True, exist_ok=True)

    def list_users(self) -> list[AccessUser]:
        return self._load_state()["users"]

    def list_roles(self) -> list[AccessRole]:
        return self._load_state()["roles"]

    def list_permissions(self) -> list[AccessPermission]:
        return self._load_state()["permissions"]

    def get_current_user(self) -> AccessUser | None:
        state = self._load_state()
        current_user_id = state["current_user_id"]
        if not current_user_id:
            return None
        return next((item for item in state["users"] if item.user_id == current_user_id), None)

    def get_current_permissions(self) -> list[str]:
        current_user = self.get_current_user()
        if current_user is None or not current_user.is_active:
            return []
        return self.resolve_permissions(current_user)

    def get_overview(self) -> AccessOverview:
        state = self._load_state()
        current_user = next((item for item in state["users"] if item.user_id == state["current_user_id"]), None)
        return AccessOverview(
            current_user=current_user,
            current_permissions=self.resolve_permissions(current_user) if current_user else [],
            users=state["users"],
            roles=state["roles"],
            permissions=state["permissions"],
        )

    def create_user(self, request: AccessUserCreateRequest) -> AccessUser:
        state = self._load_state()
        users = state["users"]
        if any(item.username == request.username for item in users):
            raise ValueError(f"Username already exists: {request.username}")
        self._validate_role_ids(request.role_ids, state["roles"])

        user = AccessUser(
            user_id=f"user-{uuid4().hex[:10]}",
            username=request.username.strip(),
            display_name=request.display_name.strip(),
            email=request.email.strip(),
            role_ids=request.role_ids,
            is_active=request.is_active,
            is_admin="admin" in request.role_ids,
            notes=(request.notes or "").strip() or None,
            password_hash=_hash_password((request.password or request.username).strip()),
        )
        users.append(user)
        self._write_state(state["current_user_id"], users, state["roles"])
        return user

    def update_user(self, user_id: str, request: AccessUserUpdateRequest) -> AccessUser:
        state = self._load_state()
        users = state["users"]
        updated_user: AccessUser | None = None
        next_users: list[AccessUser] = []
        for item in users:
            if item.user_id != user_id:
                next_users.append(item)
                continue

            role_ids = item.role_ids if request.role_ids is None else request.role_ids
            self._validate_role_ids(role_ids, state["roles"])
            updated_user = item.model_copy(
                update={
                    "display_name": item.display_name if request.display_name is None else request.display_name.strip(),
                    "email": item.email if request.email is None else request.email.strip(),
                    "role_ids": role_ids,
                    "is_active": item.is_active if request.is_active is None else request.is_active,
                    "is_admin": "admin" in role_ids,
                    "notes": item.notes if request.notes is None else ((request.notes or "").strip() or None),
                    "updated_at": utc_now(),
                }
            )
            next_users.append(updated_user)

        if updated_user is None:
            raise KeyError(f"User not found: {user_id}")

        self._write_state(state["current_user_id"], next_users, state["roles"])
        return updated_user

    def set_user_active(self, user_id: str, request: AccessUserToggleRequest) -> AccessUser:
        updated = self.update_user(
            user_id,
            AccessUserUpdateRequest(is_active=request.is_active),
        )
        state = self._load_state()
        if not request.is_active and state["current_user_id"] == user_id:
            fallback = next((item for item in state["users"] if item.user_id != user_id and item.is_active), None)
            if fallback is not None:
                self.switch_current_user(AccessCurrentUserSwitchRequest(user_id=fallback.user_id))
                state = self._load_state()
        return next((item for item in state["users"] if item.user_id == user_id), updated)

    def switch_current_user(self, request: AccessCurrentUserSwitchRequest) -> AccessUser:
        state = self._load_state()
        target = next((item for item in state["users"] if item.user_id == request.user_id), None)
        if target is None:
            raise KeyError(f"User not found: {request.user_id}")
        if not target.is_active:
            raise ValueError(f"User is inactive: {request.user_id}")
        self._write_state(target.user_id, state["users"], state["roles"])
        return target

    def reset_user_password(
        self,
        user_id: str,
        request: AccessUserPasswordResetRequest,
    ) -> AccessUserPasswordResetResponse:
        password = request.password.strip()
        if len(password) < 4:
            raise ValueError("新密码至少需要 4 个字符")

        state = self._load_state()
        updated_user: AccessUser | None = None
        next_users: list[AccessUser] = []
        for item in state["users"]:
            if item.user_id != user_id:
                next_users.append(item)
                continue

            updated_user = item.model_copy(
                update={
                    "password_hash": _hash_password(password),
                    "updated_at": utc_now(),
                }
            )
            next_users.append(updated_user)

        if updated_user is None:
            raise KeyError(f"User not found: {user_id}")

        self._write_state(state["current_user_id"], next_users, state["roles"])
        return AccessUserPasswordResetResponse(
            user_id=updated_user.user_id,
            username=updated_user.username,
        )

    def get_session(self) -> AuthSessionResponse:
        current_user = self.get_current_user()
        if current_user is None or not current_user.is_active:
            return AuthSessionResponse(authenticated=False, current_user=None, current_permissions=[])
        return AuthSessionResponse(
            authenticated=True,
            current_user=current_user,
            current_permissions=self.resolve_permissions(current_user),
        )

    def login(self, request: AuthLoginRequest) -> AuthSessionResponse:
        state = self._load_state()
        username = request.username.strip()
        password_hash = _hash_password(request.password)
        target = next((item for item in state["users"] if item.username == username), None)
        if target is None:
            raise ValueError("用户名或密码错误")
        if not target.is_active:
            raise ValueError("该账号已停用")
        if (target.password_hash or "") != password_hash:
            raise ValueError("用户名或密码错误")
        self._write_state(target.user_id, state["users"], state["roles"])
        return AuthSessionResponse(
            authenticated=True,
            current_user=target,
            current_permissions=self.resolve_permissions(target),
        )

    def logout(self) -> AuthSessionResponse:
        state = self._load_state()
        self._write_state(None, state["users"], state["roles"])
        return AuthSessionResponse(authenticated=False, current_user=None, current_permissions=[])

    def resolve_permissions(self, user: AccessUser | None) -> list[str]:
        if user is None or not user.is_active:
            return []
        role_map = {item.role_id: item for item in self.list_roles()}
        permissions: list[str] = []
        for role_id in user.role_ids:
            role = role_map.get(role_id)
            if role is None:
                continue
            for permission in role.permissions:
                if permission not in permissions:
                    permissions.append(permission)
        return permissions

    def has_permission(self, user: AccessUser | None, permission_id: str) -> bool:
        permissions = self.resolve_permissions(user)
        return permission_id in permissions

    def _load_state(self) -> dict[str, object]:
        if not self.access_control_file.exists():
            state = {
                "auth_version": AUTH_VERSION,
                "current_user_id": None,
                "users": DEFAULT_USERS,
                "roles": BUILTIN_ROLES,
                "permissions": BUILTIN_PERMISSIONS,
            }
            self._write_state(
                state["current_user_id"],
                state["users"],
                state["roles"],
            )
            return state

        with self.access_control_file.open("r", encoding="utf-8") as handle:
            raw = json.load(handle) or {}

        roles = [AccessRole.model_validate(item) for item in raw.get("roles") or [role.model_dump() for role in BUILTIN_ROLES]]
        roles = self._normalize_builtin_roles(roles)
        users = [AccessUser.model_validate(item) for item in raw.get("users") or [user.model_dump() for user in DEFAULT_USERS]]
        users = self._merge_default_users(users)
        current_user_id = raw.get("current_user_id")
        permissions = BUILTIN_PERMISSIONS
        auth_version = int(raw.get("auth_version") or 0)
        users, migrated_users = self._normalize_users_for_builtin_roles(users)
        needs_migration = auth_version < AUTH_VERSION or migrated_users

        if current_user_id is not None and not any(item.user_id == current_user_id and item.is_active for item in users):
            current_user_id = None
            needs_migration = True
        if needs_migration:
            current_user_id = None
            self._write_state(current_user_id, users, roles)

        return {
            "current_user_id": current_user_id,
            "users": users,
            "roles": roles,
            "permissions": permissions,
        }

    def _write_state(
        self,
        current_user_id: str | None,
        users: list[AccessUser],
        roles: list[AccessRole],
    ) -> None:
        with self.access_control_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "auth_version": AUTH_VERSION,
                    "current_user_id": current_user_id,
                    "users": [
                        {
                            **item.model_dump(),
                            "password_hash": item.password_hash,
                        }
                        for item in users
                    ],
                    "roles": [item.model_dump() for item in roles],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _validate_role_ids(role_ids: list[str], roles: list[AccessRole]) -> None:
        role_map = {item.role_id for item in roles}
        missing = [item for item in role_ids if item not in role_map]
        if missing:
            raise ValueError(f"Unknown role ids: {', '.join(missing)}")

    @staticmethod
    def _merge_default_users(users: list[AccessUser]) -> list[AccessUser]:
        user_map = {item.username: item for item in users}
        for default_user in DEFAULT_USERS:
            existing = user_map.get(default_user.username)
            if existing is None:
                users.append(default_user)
                continue
            if not existing.password_hash:
                existing.password_hash = default_user.password_hash
            if default_user.username == "admin":
                existing.role_ids = ["admin"]
                existing.is_admin = True
            if default_user.username == "user" and not existing.role_ids:
                existing.role_ids = ["user"]
        return users

    @staticmethod
    def _normalize_builtin_roles(roles: list[AccessRole]) -> list[AccessRole]:
        role_map = {item.role_id: item for item in roles}
        normalized_roles: list[AccessRole] = []
        for builtin_role in BUILTIN_ROLES:
            existing = role_map.get(builtin_role.role_id)
            if existing is None:
                normalized_roles.append(builtin_role)
                continue
            normalized_roles.append(
                builtin_role.model_copy(
                    update={
                        "builtin": True,
                    }
                )
            )
        return normalized_roles

    @staticmethod
    def _normalize_users_for_builtin_roles(users: list[AccessUser]) -> tuple[list[AccessUser], bool]:
        normalized_users: list[AccessUser] = []
        changed = False
        for user in users:
            normalized_role_ids = ["admin"] if "admin" in (user.role_ids or []) or user.is_admin else ["user"]
            normalized_is_admin = normalized_role_ids == ["admin"]
            if user.role_ids != normalized_role_ids or user.is_admin != normalized_is_admin:
                changed = True
                normalized_users.append(
                    user.model_copy(
                        update={
                            "role_ids": normalized_role_ids,
                            "is_admin": normalized_is_admin,
                            "updated_at": utc_now(),
                        }
                    )
                )
                continue
            normalized_users.append(user)
        return normalized_users, changed
