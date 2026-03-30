from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable

from .deployment_policy_store import DeploymentPolicyAutomationStore
from .deployment_store import DeploymentStore
from .inference_activity_tracker import InferenceActivityTracker
from .launcher import launch_runtime_process, stop_runtime_process, wait_for_runtime
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
    inference_activity_tracker: InferenceActivityTracker
    cluster_overloaded: bool


@dataclass
class _ModelProtectionStatus:
    protected: bool
    state: str
    detail: str


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
        get_inference_activity_tracker: Callable[[], InferenceActivityTracker],
    ) -> None:
        self.settings = settings
        self.get_registry = get_registry
        self.get_deployment_store = get_deployment_store
        self.get_benchmark_store = get_benchmark_store
        self.get_prediction_store = get_prediction_store
        self.get_policy_store = get_policy_store
        self.get_gpu_snapshot = get_gpu_snapshot
        self.get_inference_activity_tracker = get_inference_activity_tracker
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
                cluster_overloaded = _gpu_cluster_overloaded(policy.gpu_snapshot)
                context = _SchedulerContext(
                    policy=policy,
                    deployments=deployments,
                    prediction_archives=prediction_archives,
                    protect_active_window_seconds=state.protect_active_window_seconds,
                    inference_activity_tracker=self.get_inference_activity_tracker(),
                    cluster_overloaded=cluster_overloaded,
                )
                rebalance_applied = 0
                for recommendation in policy.recommendations:
                    if rebalance_applied < 3:
                        action = self._apply_gpu_rebalance(recommendation, context=context)
                        if action is not None:
                            actions.append(action)
                            if action.status == "migrated":
                                rebalance_applied += 1
                if cluster_overloaded:
                    cpu_relief_applied = 0
                    pressure_relief_candidates = sorted(
                        policy.recommendations,
                        key=lambda item: (
                            {"cold": 0, "warm": 1, "hot": 2}.get(item.service_tier, 3),
                            item.demand_score,
                            item.model_id,
                        ),
                    )
                    for recommendation in pressure_relief_candidates:
                        if cpu_relief_applied >= 3:
                            break
                        action = self._apply_safe_fallback(recommendation, context=context)
                        if action is not None:
                            actions.append(action)
                            if action.status == "migrated":
                                cpu_relief_applied += 1
                else:
                    actions.append(
                        DeploymentPolicyAutomationAction(
                            action_type="rebalance_only",
                            model_id=policy.summary.busiest_model_id or "-",
                            status="applied",
                            detail="检测到局部 GPU 压力，当前优先执行 GPU 内部换卡与重平衡，不触发 CPU 迁移。",
                        )
                    )
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
        if not context.cluster_overloaded:
            return None
        if recommendation.gpu_pressure_action != "migrate_to_cpu":
            return None
        if recommendation.service_tier not in {"cold", "warm"}:
            return None

        active_gpu_deployments = [
            item
            for item in context.deployments
            if item.model_id == recommendation.model_id and item.status in {"running", "starting", "planned"} and item.use_gpu
        ]
        if not active_gpu_deployments:
            return DeploymentPolicyAutomationAction(
                action_type="skip_cpu_migration",
                model_id=recommendation.model_id,
                status="skipped",
                detail="当前没有可迁移的 GPU deployment，不需要执行 CPU 释放动作。",
            )
        protection = _get_model_protection_status(
            recommendation.model_id,
            inference_activity_tracker=context.inference_activity_tracker,
            prediction_archives=context.prediction_archives,
            protect_active_window_seconds=context.protect_active_window_seconds,
        )
        if protection.protected:
            return DeploymentPolicyAutomationAction(
                action_type="skip_cpu_migration",
                model_id=recommendation.model_id,
                status="busy",
                detail=protection.detail,
            )

        registry = self.get_registry()
        deployment_store = self.get_deployment_store()
        model: ModelDetail = registry.get_model(recommendation.model_id)
        profile_name = recommendation.fallback_profile_name
        if profile_name not in model.resolved_profiles:
            profile_name = "default" if "default" in model.resolved_profiles else next(iter(model.resolved_profiles))
        deployment = sorted(
            active_gpu_deployments,
            key=lambda item: (
                {"planned": 0, "starting": 1, "running": 2}.get(item.status, 3),
                item.profile_name != profile_name,
                item.deployment_id,
            ),
        )[0]
        previous_gpu_ids = list(deployment.gpu_ids or [])
        previous_profile_name = deployment.profile_name
        previous_status = deployment.status

        try:
            if deployment.pid:
                stop_runtime_process(deployment.pid)

            deployment.manifest_path = model.summary.package_root + "/manifest.yaml"
            deployment.profile_name = profile_name
            deployment.profile_path = model.resolved_profiles[profile_name]
            deployment.use_gpu = False
            deployment.gpu_ids = []
            deployment.log_path = deployment.log_path or str(self.settings.logs_root / "deployments" / f"{deployment.deployment_id}.log")
            deployment.service_url = deployment.service_url or f"http://127.0.0.1:{deployment_store._allocate_port()}"
            deployment.notes = " | ".join(
                item
                for item in [
                    deployment.notes,
                    "auto_policy_cpu_pressure_relief",
                ]
                if item
            )

            if model.summary.predict_ready:
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
                deployment.pid = None
                deployment.status = "planned"

            deployment_store.update(deployment)
            return DeploymentPolicyAutomationAction(
                action_type="migrate_to_cpu",
                model_id=recommendation.model_id,
                deployment_id=deployment.deployment_id,
                status="migrated",
                detail=(
                    f"检测到全局 GPU 高压，已将 deployment {deployment.deployment_id} 从 GPU "
                    f"{','.join(str(item) for item in previous_gpu_ids) or '-'} 迁移到 CPU，"
                    f"profile {previous_profile_name} -> {profile_name}，释放 GPU 资源给更高优先级模型。"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            deployment.profile_name = previous_profile_name
            deployment.use_gpu = True
            deployment.gpu_ids = previous_gpu_ids
            deployment.status = previous_status
            return DeploymentPolicyAutomationAction(
                action_type="migrate_to_cpu",
                model_id=recommendation.model_id,
                deployment_id=deployment.deployment_id,
                status="failed",
                detail=f"CPU 迁移失败：{type(exc).__name__}: {exc}",
            )

    def _apply_gpu_rebalance(
        self,
        recommendation: DeploymentPolicyRecommendation,
        *,
        context: _SchedulerContext,
    ) -> DeploymentPolicyAutomationAction | None:
        if not recommendation.recommended_use_gpu or not recommendation.recommended_gpu_ids:
            return None
        if context.cluster_overloaded and recommendation.gpu_pressure_action == "migrate_to_cpu":
            return None
        if recommendation.service_tier not in {"hot", "warm"}:
            return None

        target_gpu_ids = [int(item) for item in recommendation.recommended_gpu_ids]
        active_gpu_deployments = [
            item
            for item in context.deployments
            if item.model_id == recommendation.model_id and item.status in {"running", "starting"} and item.use_gpu
        ]
        if not active_gpu_deployments:
            return None
        if any(list(item.gpu_ids or []) == target_gpu_ids for item in active_gpu_deployments):
            return None

        protection = _get_model_protection_status(
            recommendation.model_id,
            inference_activity_tracker=context.inference_activity_tracker,
            prediction_archives=context.prediction_archives,
            protect_active_window_seconds=context.protect_active_window_seconds,
        )
        if protection.protected:
            return DeploymentPolicyAutomationAction(
                action_type="rebalance_gpu",
                model_id=recommendation.model_id,
                status="busy",
                detail=protection.detail,
            )

        deployment = sorted(
            active_gpu_deployments,
            key=lambda item: (
                item.profile_name != recommendation.recommended_profile_name,
                ",".join(str(gpu_id) for gpu_id in item.gpu_ids or []),
                item.deployment_id,
            ),
        )[0]
        previous_gpu_ids = list(deployment.gpu_ids or [])
        deployment_store = self.get_deployment_store()

        try:
            registry = self.get_registry()
            model: ModelDetail = registry.get_model(recommendation.model_id)
            profile_name = recommendation.recommended_profile_name
            if profile_name not in model.resolved_profiles:
                profile_name = deployment.profile_name if deployment.profile_name in model.resolved_profiles else (
                    "default" if "default" in model.resolved_profiles else next(iter(model.resolved_profiles))
                )

            if deployment.pid:
                stop_runtime_process(deployment.pid)

            deployment.manifest_path = model.summary.package_root + "/manifest.yaml"
            deployment.profile_name = profile_name
            deployment.profile_path = model.resolved_profiles[profile_name]
            deployment.use_gpu = True
            deployment.gpu_ids = target_gpu_ids
            deployment.log_path = deployment.log_path or str(self.settings.logs_root / "deployments" / f"{deployment.deployment_id}.log")
            deployment.service_url = deployment.service_url or f"http://127.0.0.1:{deployment_store._allocate_port()}"
            port = int(deployment.service_url.rsplit(":", maxsplit=1)[-1])

            pid, service_url = launch_runtime_process(
                root_dir=self.settings.root_dir,
                manifest_path=deployment.manifest_path,
                profile_path=deployment.profile_path,
                port=port,
                log_path=deployment.log_path,
                gpu_ids=target_gpu_ids,
                use_gpu=True,
            )
            deployment.pid = pid
            deployment.service_url = service_url
            deployment.status = "running" if wait_for_runtime(service_url) else "starting"
            deployment_store.update(deployment)
            return DeploymentPolicyAutomationAction(
                action_type="rebalance_gpu",
                model_id=recommendation.model_id,
                deployment_id=deployment.deployment_id,
                status="migrated",
                detail=(
                    f"已将 deployment {deployment.deployment_id} 从 GPU "
                    f"{','.join(str(item) for item in previous_gpu_ids) or '-'} 迁移到 GPU "
                    f"{','.join(str(item) for item in target_gpu_ids)}，并切换到 profile={profile_name}。"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return DeploymentPolicyAutomationAction(
                action_type="rebalance_gpu",
                model_id=recommendation.model_id,
                deployment_id=deployment.deployment_id,
                status="failed",
                detail=f"GPU 重平衡失败：{type(exc).__name__}: {exc}",
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


def _gpu_cluster_overloaded(gpu_snapshot: dict[str, Any]) -> bool:
    if not gpu_snapshot.get("available"):
        return False
    devices = list(gpu_snapshot.get("devices") or [])
    if not devices:
        return False
    pressured_count = 0
    for item in devices:
        total = float(item.get("memory_total_mb") or 0.0)
        used = float(item.get("memory_used_mb") or 0.0)
        utilization = float(item.get("utilization_gpu_pct") or 0.0)
        free_ratio = ((total - used) / total) if total > 0 else 1.0
        if free_ratio <= 0.18 or utilization >= 85.0:
            pressured_count += 1
    return pressured_count >= len(devices)


def _get_model_protection_status(
    model_id: str,
    *,
    inference_activity_tracker: InferenceActivityTracker,
    prediction_archives: list[PredictionArchiveRecord],
    protect_active_window_seconds: int,
) -> _ModelProtectionStatus:
    tracker_status = inference_activity_tracker.get_protection_status(
        model_id,
        cooldown_seconds=protect_active_window_seconds,
    )
    if tracker_status.get("protected"):
        if tracker_status.get("state") == "active":
            active_count = int(tracker_status.get("active_count") or 0)
            return _ModelProtectionStatus(
                protected=True,
                state="active",
                detail=(
                    f"模型当前正在执行推理任务（进行中请求 {active_count} 个），"
                    "跳过自动优化以保证正在进行的推理不中断。"
                ),
            )
        cooldown_remaining_seconds = int(tracker_status.get("cooldown_remaining_seconds") or 0)
        return _ModelProtectionStatus(
            protected=True,
            state="cooldown",
            detail=(
                f"模型处于推理后冷却期，剩余约 {cooldown_remaining_seconds} 秒，"
                "跳过自动优化以避免刚完成推理后立即迁移。"
            ),
        )

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(30, protect_active_window_seconds))
    latest_archive_at: datetime | None = None
    for item in prediction_archives:
        if item.model_id != model_id:
            continue
        timestamp = _parse_datetime(item.created_at)
        if timestamp is None:
            continue
        if latest_archive_at is None or timestamp > latest_archive_at:
            latest_archive_at = timestamp
        if timestamp >= cutoff:
            return _ModelProtectionStatus(
                protected=True,
                state="archive_cooldown",
                detail=(
                    "模型近期存在推理归档记录，按保守策略视为处于推理后冷却期，"
                    "本轮跳过自动优化。"
                ),
            )
    return _ModelProtectionStatus(
        protected=False,
        state="idle",
        detail="模型当前不在推理保护窗口内。",
    )


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
