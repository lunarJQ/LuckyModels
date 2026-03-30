from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable

from .deployment_policy_store import DeploymentPolicyAutomationStore
from .deployment_store import DeploymentStore
from .launcher import launch_runtime_process, wait_for_runtime
from .schemas import (
    BenchmarkRecord,
    DeploymentCreateRequest,
    DeploymentPolicyAutomationAction,
    DeploymentPolicyAutomationState,
    DeploymentPolicyRecommendation,
    DeploymentPolicyResponse,
    DeploymentRecord,
    ModelDetail,
    PredictionArchiveRecord,
    utc_now,
)
from .settings import Settings
from .strategy_engine import build_deployment_policy


@dataclass
class _SchedulerContext:
    policy: DeploymentPolicyResponse
    deployments: list[DeploymentRecord]
    prediction_archives: list[PredictionArchiveRecord]
    protect_active_window_seconds: int


class DeploymentPolicyScheduler:
    def __init__(
        self,
        *,
        settings: Settings,
        get_registry: Callable[[], Any],
        get_deployment_store: Callable[[], DeploymentStore],
        get_benchmark_store: Callable[[], Any],
        get_prediction_store: Callable[[], Any],
        get_policy_store: Callable[[], DeploymentPolicyAutomationStore],
        get_gpu_snapshot: Callable[[], dict[str, Any]],
    ) -> None:
        self.settings = settings
        self.get_registry = get_registry
        self.get_deployment_store = get_deployment_store
        self.get_benchmark_store = get_benchmark_store
        self.get_prediction_store = get_prediction_store
        self.get_policy_store = get_policy_store
        self.get_gpu_snapshot = get_gpu_snapshot
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._run_lock = Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, name="luckymodels-policy-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_state(self) -> DeploymentPolicyAutomationState:
        return self.get_policy_store().get()

    def update_state(self, enabled: bool, interval_seconds: int, protect_active_window_seconds: int) -> DeploymentPolicyAutomationState:
        store = self.get_policy_store()
        request = store.get()
        request.enabled = bool(enabled)
        request.interval_seconds = max(60, int(interval_seconds or 900))
        request.protect_active_window_seconds = max(30, int(protect_active_window_seconds or 120))
        return store.save(request)

    def run_once(self, *, force: bool = False) -> DeploymentPolicyAutomationState:
        with self._run_lock:
            store = self.get_policy_store()
            state = store.get()
            if not force and not state.enabled:
                return state

            registry = self.get_registry()
            deployment_store = self.get_deployment_store()
            benchmark_store = self.get_benchmark_store()
            prediction_store = self.get_prediction_store()
            deployments = deployment_store.list()
            benchmarks: list[BenchmarkRecord] = benchmark_store.list()
            prediction_archives: list[PredictionArchiveRecord] = prediction_store.list()
            policy = build_deployment_policy(
                models=registry.list_models(),
                deployments=deployments,
                benchmarks=benchmarks,
                prediction_archives=prediction_archives,
                gpu_snapshot=self.get_gpu_snapshot(),
            )

            actions: list[DeploymentPolicyAutomationAction] = []
            status = "idle"
            if _gpu_under_pressure(policy.gpu_snapshot):
                status = "applied"
                context = _SchedulerContext(
                    policy=policy,
                    deployments=deployments,
                    prediction_archives=prediction_archives,
                    protect_active_window_seconds=state.protect_active_window_seconds,
                )
                for recommendation in policy.recommendations:
                    action = self._apply_safe_fallback(recommendation, context=context)
                    if action is not None:
                        actions.append(action)
            else:
                actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="scan_only",
                        model_id=policy.summary.busiest_model_id or "-",
                        status="skipped",
                        detail="当前 GPU 压力未达到阈值，自动优化本轮仅完成策略扫描，不执行迁移。",
                    )
                )

            return store.record_run(
                state=state,
                last_status=status,
                last_run_at=utc_now(),
                last_policy_generated_at=policy.generated_at,
                summary=policy.summary,
                actions=actions,
            )

    def _run_loop(self) -> None:
        while not self._stop_event.wait(10):
            state = self.get_policy_store().get()
            if not state.enabled:
                continue
            if not _is_due(state.last_run_at, state.interval_seconds):
                continue
            self.run_once(force=True)

    def _apply_safe_fallback(
        self,
        recommendation: DeploymentPolicyRecommendation,
        *,
        context: _SchedulerContext,
    ) -> DeploymentPolicyAutomationAction | None:
        if recommendation.gpu_pressure_action != "migrate_to_cpu":
            return None
        if recommendation.service_tier not in {"cold", "warm"}:
            return None

        active_model_deployments = [
            item for item in context.deployments if item.model_id == recommendation.model_id and item.status in {"running", "starting", "planned"}
        ]
        if not any(item.use_gpu for item in active_model_deployments):
            return DeploymentPolicyAutomationAction(
                action_type="skip_cpu_fallback",
                model_id=recommendation.model_id,
                status="skipped",
                detail="当前没有 GPU 部署实例，不需要转为 CPU 兜底。",
            )
        if any(not item.use_gpu for item in active_model_deployments):
            return DeploymentPolicyAutomationAction(
                action_type="skip_cpu_fallback",
                model_id=recommendation.model_id,
                status="skipped",
                detail="已经存在 CPU 兜底 deployment，本轮不重复创建。",
            )
        if _model_is_busy(
            recommendation.model_id,
            prediction_archives=context.prediction_archives,
            protect_active_window_seconds=context.protect_active_window_seconds,
        ):
            return DeploymentPolicyAutomationAction(
                action_type="skip_cpu_fallback",
                model_id=recommendation.model_id,
                status="busy",
                detail="模型近期仍有推理请求，跳过自动优化以保证推理任务畅通。",
            )

        registry = self.get_registry()
        deployment_store = self.get_deployment_store()
        model: ModelDetail = registry.get_model(recommendation.model_id)
        profile_name = recommendation.fallback_profile_name
        if profile_name not in model.resolved_profiles:
            profile_name = "default" if "default" in model.resolved_profiles else next(iter(model.resolved_profiles))
        request = DeploymentCreateRequest(
            model_id=recommendation.model_id,
            profile_name=profile_name,
            use_gpu=False,
            gpu_ids=[],
            launch_runtime=bool(model.summary.predict_ready),
            notes="auto_policy_cpu_fallback",
        )
        deployment = deployment_store.create(request)
        deployment.manifest_path = model.summary.package_root + "/manifest.yaml"
        deployment.profile_path = model.resolved_profiles[profile_name]
        deployment.service_url = deployment.service_url or f"http://127.0.0.1:{deployment_store._allocate_port()}"
        deployment.log_path = str(self.settings.logs_root / "deployments" / f"{deployment.deployment_id}.log")

        if request.launch_runtime:
            port = int(deployment.service_url.rsplit(":", maxsplit=1)[-1])
            pid, service_url = launch_runtime_process(
                root_dir=self.settings.root_dir,
                manifest_path=deployment.manifest_path,
                profile_path=deployment.profile_path,
                port=port,
                log_path=deployment.log_path,
                gpu_ids=[],
                use_gpu=False,
            )
            deployment.pid = pid
            deployment.service_url = service_url
            deployment.status = "running" if wait_for_runtime(service_url) else "starting"
        else:
            deployment.status = "planned"

        deployment_store.update(deployment)
        return DeploymentPolicyAutomationAction(
            action_type="create_cpu_fallback",
            model_id=recommendation.model_id,
            deployment_id=deployment.deployment_id,
            status="created",
            detail=(
                f"检测到 GPU 压力较高，已为 {recommendation.model_id} 创建 CPU 兜底 deployment，"
                f"使用 profile={profile_name}，不影响原有 GPU 推理实例。"
            ),
        )


def _is_due(last_run_at: str | None, interval_seconds: int) -> bool:
    if not last_run_at:
        return True
    try:
        last = datetime.fromisoformat(str(last_run_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return now >= last.astimezone(timezone.utc) + timedelta(seconds=max(60, interval_seconds))


def _gpu_under_pressure(gpu_snapshot: dict[str, Any]) -> bool:
    if not gpu_snapshot.get("available"):
        return False
    devices = list(gpu_snapshot.get("devices") or [])
    for item in devices:
        total = float(item.get("memory_total_mb") or 0.0)
        used = float(item.get("memory_used_mb") or 0.0)
        utilization = float(item.get("utilization_gpu_pct") or 0.0)
        free_ratio = ((total - used) / total) if total > 0 else 1.0
        if free_ratio <= 0.18 or utilization >= 85.0:
            return True
    return False


def _model_is_busy(
    model_id: str,
    *,
    prediction_archives: list[PredictionArchiveRecord],
    protect_active_window_seconds: int,
) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(30, protect_active_window_seconds))
    for item in prediction_archives:
        if item.model_id != model_id:
            continue
        timestamp = _parse_datetime(item.created_at)
        if timestamp is not None and timestamp >= cutoff:
            return True
    return False


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None
