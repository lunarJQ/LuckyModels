from __future__ import annotations

from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import (
    AccessCurrentUserSwitchRequest,
    AccessOverview,
    AccessRole,
    AccessPermission,
    AccessUser,
    AccessUserCreateRequest,
    AccessUserPasswordResetRequest,
    AccessUserPasswordResetResponse,
    AccessUserToggleRequest,
    AccessUserUpdateRequest,
)

from ..dependencies import get_access_store
from ..permissions import get_current_user, require_permission

router = APIRouter(tags=["access-control"])


@router.get("/access-control/overview", response_model=AccessOverview)
def get_access_control_overview() -> AccessOverview:
    _ensure_read_access()
    return get_access_store().get_overview()


@router.get("/access-control/users", response_model=list[AccessUser])
def list_access_users() -> list[AccessUser]:
    _ensure_read_access()
    return get_access_store().list_users()


@router.get("/access-control/roles", response_model=list[AccessRole])
def list_access_roles() -> list[AccessRole]:
    _ensure_read_access()
    return get_access_store().list_roles()


@router.get("/access-control/permissions", response_model=list[AccessPermission])
def list_access_permissions() -> list[AccessPermission]:
    _ensure_read_access()
    return get_access_store().list_permissions()


@router.post("/access-control/users", response_model=AccessUser)
def create_access_user(request: AccessUserCreateRequest) -> AccessUser:
    _ensure_manage_access()
    try:
        return get_access_store().create_user(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/access-control/users/{user_id}", response_model=AccessUser)
def update_access_user(user_id: str, request: AccessUserUpdateRequest) -> AccessUser:
    _ensure_manage_access()
    try:
        return get_access_store().update_user(user_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/access-control/users/{user_id}/toggle", response_model=AccessUser)
def toggle_access_user(user_id: str, request: AccessUserToggleRequest) -> AccessUser:
    _ensure_manage_access()
    try:
        return get_access_store().set_user_active(user_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/access-control/current-user", response_model=AccessUser)
def switch_current_access_user(request: AccessCurrentUserSwitchRequest) -> AccessUser:
    _ensure_manage_access()
    try:
        return get_access_store().switch_current_user(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/access-control/users/{user_id}/reset-password", response_model=AccessUserPasswordResetResponse)
def reset_access_user_password(
    user_id: str,
    request: AccessUserPasswordResetRequest,
) -> AccessUserPasswordResetResponse:
    _ensure_manage_access()
    try:
        return get_access_store().reset_user_password(user_id, request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _ensure_read_access() -> None:
    require_permission("users.view")


def _ensure_manage_access() -> None:
    current_user = get_current_user()
    if get_access_store().has_permission(current_user, "users.manage"):
        return
    raise HTTPException(status_code=403, detail="Current user lacks access-control manage permission")
