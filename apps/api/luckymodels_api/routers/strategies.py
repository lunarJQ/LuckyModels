from uuid import uuid4

from fastapi import APIRouter, HTTPException

from luckymodels_core.deployment_policy_scheduler import DeploymentPolicyScheduler
from luckymodels_core.schemas import (
    DeploymentPolicyAutomationState,
    DeploymentPolicyAutomationUpdateRequest,
    DeploymentPolicyResponse,
    TaskRoutingDecisionRecord,
    TaskRoutingDecisionSummary,
    TaskRoutingFeedbackOverview,
    TaskRoutingFeedbackRecord,
    TaskRoutingFeedbackSubmitRequest,
    TaskRoutingRequest,
    TaskRoutingResponse,
)
from luckymodels_core.settings import get_settings
from luckymodels_core.strategy_engine import (
    build_deployment_policy,
    build_task_routing_feedback_overview,
    run_task_routing_strategy,
)

from ..dependencies import (
    get_benchmark_store,
    get_deployment_policy_store,
    get_deployment_store,
    get_inference_activity_tracker,
    get_prediction_store,
    get_registry,
    get_task_routing_feedback_store,
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
    feedback_store = get_task_routing_feedback_store()
    response = run_task_routing_strategy(
        request,
        models=registry.list_models(),
        deployments=deployment_store.list(),
        benchmarks=benchmark_store.list(),
        prediction_archives=prediction_store.list(),
        decision_records=feedback_store.list_decisions(),
        feedback_records=feedback_store.list_feedback(),
        activity_tracker=get_inference_activity_tracker(),
    )
    decision = feedback_store.create_decision(
        TaskRoutingDecisionRecord(
            decision_id=f"trd-{uuid4().hex[:12]}",
            request_text=request.text,
            scene=request.scene,
            objective=response.objective,
            requested_language=response.requested_language,
            inferred_language=response.inferred_language,
            difficulty=response.difficulty,
            strategy_mode=response.strategy_mode,
            execution_path=response.execution_path,
            fusion_policy=response.fusion_policy,
            context_features=response.context_features,
            objective_weights=response.objective_weights,
            adaptive_adjustments=response.adaptive_adjustments,
            adaptation_reasons=response.adaptation_reasons,
            selected_deployments=response.selected_deployments,
            execution_results=response.execution_results,
            fused_result=response.fused_result,
        )
    )
    response.decision_id = decision.decision_id
    return response


@router.get("/strategies/task-routing/feedback/overview", response_model=TaskRoutingFeedbackOverview)
def get_task_routing_feedback_overview() -> TaskRoutingFeedbackOverview:
    registry = get_registry()
    deployment_store = get_deployment_store()
    benchmark_store = get_benchmark_store()
    feedback_store = get_task_routing_feedback_store()
    return build_task_routing_feedback_overview(
        models=registry.list_models(),
        deployments=deployment_store.list(),
        benchmarks=benchmark_store.list(),
        prediction_archives=get_prediction_store().list(),
        decision_records=feedback_store.list_decisions(),
        feedback_records=feedback_store.list_feedback(),
    )


@router.get("/strategies/task-routing/decisions", response_model=list[TaskRoutingDecisionSummary])
def list_task_routing_decisions(limit: int = 20) -> list[TaskRoutingDecisionSummary]:
    feedback_store = get_task_routing_feedback_store()
    overview = build_task_routing_feedback_overview(
        models=get_registry().list_models(),
        deployments=get_deployment_store().list(),
        benchmarks=get_benchmark_store().list(),
        prediction_archives=get_prediction_store().list(),
        decision_records=feedback_store.list_decisions(),
        feedback_records=feedback_store.list_feedback(),
        limit=max(1, min(limit, 100)),
    )
    return overview.recent_decisions


@router.post("/strategies/task-routing/feedback", response_model=TaskRoutingFeedbackRecord)
def submit_task_routing_feedback(request: TaskRoutingFeedbackSubmitRequest) -> TaskRoutingFeedbackRecord:
    store = get_task_routing_feedback_store()
    try:
        return store.save_feedback(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
