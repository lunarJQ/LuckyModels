from fastapi import APIRouter

from luckymodels_core.schemas import SystemSummary
from luckymodels_core.settings import get_settings

from ..dependencies import get_benchmark_store, get_deployment_store, get_registry

router = APIRouter(tags=["system"])


@router.get("/system/summary", response_model=SystemSummary)
def get_system_summary() -> SystemSummary:
    registry = get_registry()
    deployment_store = get_deployment_store()
    benchmark_store = get_benchmark_store()
    settings = get_settings()
    return SystemSummary(
        model_count=len(registry.list_models()),
        deployment_count=len(deployment_store.list()),
        benchmark_count=len(benchmark_store.list()),
        artifact_root=str(settings.artifacts_root),
        logs_root=str(settings.logs_root),
    )
