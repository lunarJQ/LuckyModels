from fastapi import APIRouter

from luckymodels_core.deployment_policy_scheduler import DeploymentPolicyScheduler
from luckymodels_core.schemas import (
    DeploymentPolicyAutomationState,
    DeploymentPolicyAutomationUpdateRequest,
    DeploymentPolicyResponse,
    TaskRoutingRequest,
    TaskRoutingResponse,
)
from luckymodels_core.settings import get_settings
from luckymodels_core.strategy_engine import build_deployment_policy, run_task_routing_strategy

from ..dependencies import (
    get_benchmark_store,
    get_deployment_policy_store,
    get_deployment_store,
    get_inference_activity_tracker,
    get_prediction_store,
    get_registry,
)
from .system import get_gpu_status

router = APIRouter(tags=["strategies"])
_scheduler: DeploymentPolicyScheduler | None = None


def start_strategy_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.start()
        return
    _scheduler = DeploymentPolicyScheduler(
        settings=get_settings(),
        get_registry=get_registry,
        get_deployment_store=get_deployment_store,
        get_benchmark_store=get_benchmark_store,
        get_prediction_store=get_prediction_store,
        get_policy_store=get_deployment_policy_store,
        get_gpu_snapshot=get_gpu_status,
        get_inference_activity_tracker=get_inference_activity_tracker,
    )
    _scheduler.start()


def stop_strategy_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    _scheduler.stop()


def _get_scheduler() -> DeploymentPolicyScheduler:
    start_strategy_scheduler()
    assert _scheduler is not None
    return _scheduler


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
        activity_tracker=get_inference_activity_tracker(),
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


@router.get("/strategies/deployment-policy/automation", response_model=DeploymentPolicyAutomationState)
def get_deployment_policy_automation() -> DeploymentPolicyAutomationState:
    return _get_scheduler().get_state()


@router.post("/strategies/deployment-policy/automation", response_model=DeploymentPolicyAutomationState)
def update_deployment_policy_automation(
    request: DeploymentPolicyAutomationUpdateRequest,
) -> DeploymentPolicyAutomationState:
    return _get_scheduler().update_state(
        enabled=request.enabled,
        interval_seconds=request.interval_seconds,
        protect_active_window_seconds=request.protect_active_window_seconds,
    )


@router.post("/strategies/deployment-policy/automation/run", response_model=DeploymentPolicyAutomationState)
def run_deployment_policy_automation_now() -> DeploymentPolicyAutomationState:
    return _get_scheduler().run_once(force=True)
