from fastapi import APIRouter

from luckymodels_core.schemas import DeploymentPolicyResponse, TaskRoutingRequest, TaskRoutingResponse
from luckymodels_core.strategy_engine import build_deployment_policy, run_task_routing_strategy

from ..dependencies import get_benchmark_store, get_deployment_store, get_prediction_store, get_registry
from .system import get_gpu_status

router = APIRouter(tags=["strategies"])


@router.post("/strategies/task-routing/run", response_model=TaskRoutingResponse)
def run_task_strategy(request: TaskRoutingRequest) -> TaskRoutingResponse:
    registry = get_registry()
    deployment_store = get_deployment_store()
    benchmark_store = get_benchmark_store()
    prediction_store = get_prediction_store()
    return run_task_routing_strategy(
        request,
        models=registry.list_models(),
        deployments=deployment_store.list(),
        benchmarks=benchmark_store.list(),
        prediction_archives=prediction_store.list(),
    )


@router.get("/strategies/deployment-policy", response_model=DeploymentPolicyResponse)
def get_deployment_policy() -> DeploymentPolicyResponse:
    registry = get_registry()
    deployment_store = get_deployment_store()
    benchmark_store = get_benchmark_store()
    prediction_store = get_prediction_store()
    gpu_snapshot = get_gpu_status()
    return build_deployment_policy(
        models=registry.list_models(),
        deployments=deployment_store.list(),
        benchmarks=benchmark_store.list(),
        prediction_archives=prediction_store.list(),
        gpu_snapshot=gpu_snapshot,
    )
