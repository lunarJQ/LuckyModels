from __future__ import annotations

from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import AuthLoginRequest, AuthSessionResponse

from ..dependencies import get_access_store

router = APIRouter(tags=["auth"])


@router.get("/auth/session", response_model=AuthSessionResponse)
def get_auth_session() -> AuthSessionResponse:
    return get_access_store().get_session()


@router.post("/auth/login", response_model=AuthSessionResponse)
def login(request: AuthLoginRequest) -> AuthSessionResponse:
    try:
        return get_access_store().login(request)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/auth/logout", response_model=AuthSessionResponse)
def logout() -> AuthSessionResponse:
    return get_access_store().logout()
