from __future__ import annotations

from fastapi import HTTPException

from .dependencies import get_access_store


def get_current_user():
    store = get_access_store()
    current_user = store.get_current_user()
    if current_user is None:
        raise HTTPException(status_code=401, detail="No active current user is configured")
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Current user is inactive")
    return current_user


def require_permission(permission_id: str):
    store = get_access_store()
    current_user = get_current_user()
    if store.has_permission(current_user, permission_id):
        return current_user
    raise HTTPException(status_code=403, detail=f"Current user lacks permission: {permission_id}")


def require_admin():
    current_user = get_current_user()
    if current_user.is_admin:
        return current_user
    raise HTTPException(status_code=403, detail="Current user must be an administrator")
