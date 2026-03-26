from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    html_path = Path(__file__).resolve().parents[1] / "ui" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_alias() -> HTMLResponse:
    return dashboard()
