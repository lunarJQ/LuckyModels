from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from luckymodels_api.dependencies import get_access_store
from luckymodels_api.routers import access_control, auth, benchmarks, demo, deployments, health, models, predictions, strategies, system, ui

app = FastAPI(
    title="LuckyModels Control Plane",
    version="0.1.0",
    description="Control-plane foundation for LuckyModels 1.0",
)

ui_dist = Path(__file__).resolve().parent / "luckymodels_api" / "ui" / "dist"

app.mount("/static", StaticFiles(directory=str(ui_dist), check_dir=False), name="static")


@app.middleware("http")
async def require_authentication(request: Request, call_next):
    path = request.url.path
    open_api_paths = {
        "/api/v1/auth/login",
        "/api/v1/auth/session",
    }
    if path.startswith("/api/v1") and path not in open_api_paths:
        current_user = get_access_store().get_current_user()
        if current_user is None or not current_user.is_active:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )
    return await call_next(request)

app.include_router(ui.router)
app.include_router(health.router)
app.include_router(auth.router, prefix="/api/v1")
app.include_router(system.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(access_control.router, prefix="/api/v1")
app.include_router(deployments.router, prefix="/api/v1")
app.include_router(benchmarks.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(strategies.router, prefix="/api/v1")
app.include_router(demo.router, prefix="/api/v1")
