from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from luckymodels_api.routers import benchmarks, demo, deployments, health, models, predictions, system, ui

app = FastAPI(
    title="LuckyModels Control Plane",
    version="0.1.0",
    description="Control-plane foundation for LuckyModels 1.0",
)

ui_dist = Path(__file__).resolve().parent / "luckymodels_api" / "ui" / "dist"

app.mount("/static", StaticFiles(directory=str(ui_dist), check_dir=False), name="static")

app.include_router(ui.router)
app.include_router(health.router)
app.include_router(system.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(deployments.router, prefix="/api/v1")
app.include_router(benchmarks.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(demo.router, prefix="/api/v1")
