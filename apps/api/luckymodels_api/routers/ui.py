from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    ui_root = Path(__file__).resolve().parents[1] / "ui"
    dist_index = ui_root / "dist" / "index.html"
    html_path = dist_index if dist_index.exists() else ui_root / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_alias() -> HTMLResponse:
    return dashboard()
