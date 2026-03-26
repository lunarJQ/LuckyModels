from fastapi import FastAPI

from luckymodels_api.routers import deployments, health, models, system

app = FastAPI(
    title="LuckyModels Control Plane",
    version="0.1.0",
    description="Control-plane foundation for LuckyModels 1.0",
)

app.include_router(health.router)
app.include_router(system.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(deployments.router, prefix="/api/v1")
