from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

FRONTEND_PATHS = [
    "/",
    "/login",
    "/dashboard",
    "/experiments",
    "/access-control",
    "/admin-console",
    "/archive",
    "/models",
    "/task-routing",
    "/deployment-policy",
    "/capability-matrix",
    "/artifacts",
    "/deployments",
    "/benchmarks",
    "/resources",
    "/logs",
]

def _render_dashboard() -> HTMLResponse:
    ui_root = Path(__file__).resolve().parents[1] / "ui"
    dist_index = ui_root / "dist" / "index.html"
    html_path = dist_index if dist_index.exists() else ui_root / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


for frontend_path in FRONTEND_PATHS:
    router.add_api_route(
        frontend_path,
        _render_dashboard,
        methods=["GET"],
        response_class=HTMLResponse,
    )
