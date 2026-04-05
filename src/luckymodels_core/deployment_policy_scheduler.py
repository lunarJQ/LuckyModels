from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable

from .benchmark_evidence import default_benchmark_text, infer_benchmark_scene
from .benchmark_runner import run_benchmark
from .deployment_policy_store import DeploymentPolicyAutomationStore
from .deployment_revision import (
    advance_deployment_revision,
    classify_runtime_change,
    initialize_deployment_revision_metadata,
)
from .deployment_store import DeploymentStore
from .inference_activity_tracker import InferenceActivityTracker
from .launcher import launch_runtime_process, stop_runtime_process, wait_for_runtime
from .manifest import load_profile
from .schemas import (
    BenchmarkRecord,
    DeploymentCreateRequest,
    DeploymentPolicyAutomationAction,
    DeploymentPolicyAutomationState,
    DeploymentPolicyRecommendation,
    DeploymentPolicyResponse,
    DeploymentRecord,
    GpuPoolState,
    GpuPoolUpdateResult,
    BenchmarkCreateRequest,
    ModelDetail,
    PredictionArchiveRecord,
    utc_now,
)
from .settings import Settings
from .task_routing_feedback_store import TaskRoutingFeedbackStore
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
class _GpuMemoryHotspot:
    deployment_id: str
    model_id: str
    pid: int
    used_gpu_memory_mb: float
    gpu_ids: list[int]


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
        get_task_routing_feedback_store: Callable[[], TaskRoutingFeedbackStore],
        get_policy_store: Callable[[], DeploymentPolicyAutomationStore],
        get_gpu_snapshot: Callable[[], dict[str, Any]],
        get_inference_activity_tracker: Callable[[], InferenceActivityTracker],
    ) -> None:
        self.settings = settings
        self.get_registry = get_registry
        self.get_deployment_store = get_deployment_store
        self.get_benchmark_store = get_benchmark_store
        self.get_prediction_store = get_prediction_store
        self.get_task_routing_feedback_store = get_task_routing_feedback_store
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

    def update_state(
        self,
        enabled: bool,
        interval_seconds: int,
        protect_active_window_seconds: int,
        gpu_memory_restart_enabled: bool,
        gpu_memory_restart_interval_seconds: int,
        gpu_memory_restart_threshold_mb: int,
    ) -> DeploymentPolicyAutomationState:
        store = self.get_policy_store()
        request = store.get()
        request.enabled = bool(enabled)
        request.interval_seconds = max(60, int(interval_seconds or 900))
        request.protect_active_window_seconds = max(30, int(protect_active_window_seconds or 120))
        request.gpu_memory_restart_enabled = bool(gpu_memory_restart_enabled)
        request.gpu_memory_restart_interval_seconds = max(60, int(gpu_memory_restart_interval_seconds or 60))
        request.gpu_memory_restart_threshold_mb = max(1024, int(gpu_memory_restart_threshold_mb or 5120))
        return store.save(request)

    def run_once(
        self,
        *,
        force: bool = False,
        include_policy_optimization: bool | None = None,
        include_gpu_memory_restart: bool | None = None,
    ) -> DeploymentPolicyAutomationState:
        with self._run_lock:
            store = self.get_policy_store()
            state = store.get()
            run_policy_optimization = state.enabled if include_policy_optimization is None else bool(include_policy_optimization)
            run_gpu_memory_restart = (
                state.gpu_memory_restart_enabled
                if include_gpu_memory_restart is None
                else bool(include_gpu_memory_restart)
            )
            if not force and not run_policy_optimization and not run_gpu_memory_restart:
                return state

            registry = self.get_registry()
            deployment_store = self.get_deployment_store()
            benchmark_store = self.get_benchmark_store()
            prediction_store = self.get_prediction_store()
            task_routing_feedback_store = self.get_task_routing_feedback_store()
            deployments = deployment_store.list()
            benchmarks: list[BenchmarkRecord] = benchmark_store.list()
            prediction_archives: list[PredictionArchiveRecord] = prediction_store.list()
            gpu_snapshot = self.get_gpu_snapshot()
            policy: DeploymentPolicyResponse | None = None
            if run_policy_optimization:
                policy = build_deployment_policy(
                    models=registry.list_models(),
                    deployments=deployments,
                    benchmarks=benchmarks,
                    prediction_archives=prediction_archives,
                    gpu_snapshot=gpu_snapshot,
                    decision_records=task_routing_feedback_store.list_decisions(),
                    feedback_records=task_routing_feedback_store.list_feedback(),
                    automation_actions=state.recent_actions,
                )

            actions: list[DeploymentPolicyAutomationAction] = []
            status = "idle"
            restarted_model_ids: set[str] = set()
            memory_restart_actions = []
            if run_gpu_memory_restart:
                memory_restart_actions = self._restart_gpu_memory_hotspots(
                    deployments=deployments,
                    prediction_archives=prediction_archives,
                    gpu_snapshot=gpu_snapshot,
                    protect_active_window_seconds=state.protect_active_window_seconds,
                    gpu_memory_restart_enabled=state.gpu_memory_restart_enabled,
                    gpu_memory_restart_threshold_mb=state.gpu_memory_restart_threshold_mb,
                )
            actions.extend(memory_restart_actions)
            restarted_model_ids.update(
                item.model_id for item in memory_restart_actions if item.status == "restarted"
            )

            if run_policy_optimization and policy is not None and _gpu_under_pressure(policy.gpu_snapshot):
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
                    if recommendation.model_id in restarted_model_ids:
                        continue
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
                        if recommendation.model_id in restarted_model_ids:
                            continue
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
            elif run_policy_optimization and policy is not None:
                context = _SchedulerContext(
                    policy=policy,
                    deployments=deployments,
                    prediction_archives=prediction_archives,
                    protect_active_window_seconds=state.protect_active_window_seconds,
                    inference_activity_tracker=self.get_inference_activity_tracker(),
                    cluster_overloaded=False,
                )
                recovery_actions = self._apply_resource_recovery(
                    context=context,
                    restarted_model_ids=restarted_model_ids,
                )
                if recovery_actions:
                    actions.extend(recovery_actions)
                elif not memory_restart_actions:
                    actions.append(
                        DeploymentPolicyAutomationAction(
                            action_type="scan_only",
                            model_id=policy.summary.busiest_model_id or "-",
                            status="skipped",
                            detail="当前 GPU 压力未达到阈值，自动优化本轮完成策略扫描；没有发现需要执行的资源恢复动作。",
                        )
                    )

            if any(item.status in {"restarted", "migrated", "applied"} for item in actions):
                status = "applied"

            benchmark_refresh_actions = self._refresh_benchmarks_for_actions(actions)
            if benchmark_refresh_actions:
                actions.extend(benchmark_refresh_actions)
                status = "applied"

            return store.record_run(
                state=state,
                last_status=status,
                last_run_at=utc_now() if run_policy_optimization else None,
                last_gpu_memory_restart_at=utc_now() if run_gpu_memory_restart else None,
                last_policy_generated_at=policy.generated_at if policy is not None else None,
                summary=policy.summary if policy is not None else None,
                actions=actions,
            )

    def apply_gpu_pool_change(
        self,
        *,
        previous_state: GpuPoolState,
        next_state: GpuPoolState,
        trigger_policy_optimization: bool = True,
    ) -> GpuPoolUpdateResult:
        with self._run_lock:
            policy_store = self.get_policy_store()
            automation_state = policy_store.get()
            registry = self.get_registry()
            deployment_store = self.get_deployment_store()
            benchmark_store = self.get_benchmark_store()
            prediction_store = self.get_prediction_store()
            feedback_store = self.get_task_routing_feedback_store()
            prediction_archives: list[PredictionArchiveRecord] = prediction_store.list()

            previous_effective = sorted({int(item) for item in previous_state.effective_gpu_ids})
            next_effective = sorted({int(item) for item in next_state.effective_gpu_ids})
            disabled_gpu_ids = [item for item in previous_effective if item not in next_effective]
            enabled_gpu_ids = [item for item in next_effective if item not in previous_effective]

            actions: list[DeploymentPolicyAutomationAction] = []
            affected_deployment_ids: list[str] = []
            migrated_deployment_ids: list[str] = []
            cpu_fallback_deployment_ids: list[str] = []
            busy_deployment_ids: list[str] = []
            skipped_deployment_ids: list[str] = []

            policy: DeploymentPolicyResponse | None = None
            gpu_snapshot = self.get_gpu_snapshot()
            if trigger_policy_optimization:
                policy = build_deployment_policy(
                    models=registry.list_models(),
                    deployments=deployment_store.list(),
                    benchmarks=benchmark_store.list(),
                    prediction_archives=prediction_archives,
                    gpu_snapshot=gpu_snapshot,
                    decision_records=feedback_store.list_decisions(),
                    feedback_records=feedback_store.list_feedback(),
                    automation_actions=automation_state.recent_actions,
                )

            context = _SchedulerContext(
                policy=policy or DeploymentPolicyResponse(gpu_snapshot=gpu_snapshot, gpu_pool_state=next_state),
                deployments=deployment_store.list(),
                prediction_archives=prediction_archives,
                protect_active_window_seconds=automation_state.protect_active_window_seconds,
                inference_activity_tracker=self.get_inference_activity_tracker(),
                cluster_overloaded=_gpu_cluster_overloaded(gpu_snapshot),
            )
            recommendation_by_model_id = {
                item.model_id: item for item in (policy.recommendations if policy is not None else [])
            }

            if disabled_gpu_ids:
                affected_deployments = [
                    item
                    for item in deployment_store.list()
                    if item.use_gpu
                    and item.status in {"running", "starting", "planned"}
                    and any(int(gpu_id) in disabled_gpu_ids for gpu_id in (item.gpu_ids or []))
                ]
                affected_deployment_ids = [item.deployment_id for item in affected_deployments]

                for deployment in affected_deployments:
                    recommendation = recommendation_by_model_id.get(deployment.model_id)
                    action = self._relocate_deployment_for_gpu_pool_change(
                        deployment=deployment,
                        recommendation=recommendation,
                        context=context,
                    )
                    actions.append(action)
                    if action.status == "migrated" and action.deployment_id:
                        migrated_deployment_ids.append(action.deployment_id)
                        if action.action_type == "gpu_pool_fallback_cpu":
                            cpu_fallback_deployment_ids.append(action.deployment_id)
                    elif action.status == "busy" and action.deployment_id:
                        busy_deployment_ids.append(action.deployment_id)
                    elif action.deployment_id:
                        skipped_deployment_ids.append(action.deployment_id)

            follow_up_actions: list[DeploymentPolicyAutomationAction] = []
            if trigger_policy_optimization:
                gpu_snapshot = self.get_gpu_snapshot()
                policy = build_deployment_policy(
                    models=registry.list_models(),
                    deployments=deployment_store.list(),
                    benchmarks=benchmark_store.list(),
                    prediction_archives=prediction_archives,
                    gpu_snapshot=gpu_snapshot,
                    decision_records=feedback_store.list_decisions(),
                    feedback_records=feedback_store.list_feedback(),
                    automation_actions=(actions + automation_state.recent_actions)[:20],
                )
                context = _SchedulerContext(
                    policy=policy,
                    deployments=deployment_store.list(),
                    prediction_archives=prediction_archives,
                    protect_active_window_seconds=automation_state.protect_active_window_seconds,
                    inference_activity_tracker=self.get_inference_activity_tracker(),
                    cluster_overloaded=_gpu_cluster_overloaded(gpu_snapshot),
                )
                follow_up_actions = self._apply_follow_up_gpu_pool_rebalance(
                    policy=policy,
                    context=context,
                    excluded_model_ids={
                        item.model_id
                        for item in actions
                        if item.status == "migrated" and item.model_id
                    },
                )
                actions.extend(follow_up_actions)
                for action in follow_up_actions:
                    if action.status == "migrated" and action.deployment_id:
                        migrated_deployment_ids.append(action.deployment_id)
                    elif action.status == "busy" and action.deployment_id:
                        busy_deployment_ids.append(action.deployment_id)

            if not actions:
                actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="gpu_pool_update",
                        model_id="-",
                        status="applied",
                        detail=(
                            f"GPU 使用范围已更新为 "
                            f"{','.join(str(item) for item in next_effective) or 'CPU only'}，"
                            "当前没有需要迁移的 deployment。"
                        ),
                    )
                )

            benchmark_refresh_actions = self._refresh_benchmarks_for_actions(actions)
            if benchmark_refresh_actions:
                actions.extend(benchmark_refresh_actions)

            status = "applied" if any(item.status in {"migrated", "applied", "restarted"} for item in actions) else "idle"
            policy_store.record_run(
                state=automation_state,
                last_status=status,
                last_run_at=utc_now() if trigger_policy_optimization else None,
                last_gpu_memory_restart_at=None,
                last_policy_generated_at=policy.generated_at if policy is not None else None,
                summary=policy.summary if policy is not None else None,
                actions=actions,
            )

            detail = (
                f"已批量更新平台可用 GPU：{','.join(str(item) for item in next_effective) or '仅 CPU'}。"
                f" 受影响 deployment {len(affected_deployment_ids)} 个，实际迁移 {len(migrated_deployment_ids)} 个。"
            )
            return GpuPoolUpdateResult(
                state=next_state,
                previous_effective_gpu_ids=previous_effective,
                enabled_gpu_ids=enabled_gpu_ids,
                disabled_gpu_ids=disabled_gpu_ids,
                affected_deployment_ids=affected_deployment_ids,
                migrated_deployment_ids=sorted(set(migrated_deployment_ids)),
                cpu_fallback_deployment_ids=sorted(set(cpu_fallback_deployment_ids)),
                busy_deployment_ids=sorted(set(busy_deployment_ids)),
                skipped_deployment_ids=sorted(set(skipped_deployment_ids)),
                actions=actions,
                policy_run_triggered=bool(trigger_policy_optimization),
                detail=detail,
            )

    def _run_loop(self) -> None:
        while not self._stop_event.wait(10):
            state = self.get_policy_store().get()
            run_policy_optimization = bool(state.enabled) and _is_due(state.last_run_at, state.interval_seconds)
            run_gpu_memory_restart = bool(state.gpu_memory_restart_enabled) and _is_due(
                state.last_gpu_memory_restart_at,
                state.gpu_memory_restart_interval_seconds,
            )
            if not run_policy_optimization and not run_gpu_memory_restart:
                continue
            self.run_once(
                force=True,
                include_policy_optimization=run_policy_optimization,
                include_gpu_memory_restart=run_gpu_memory_restart,
            )

    def _restart_gpu_memory_hotspots(
        self,
        *,
        deployments: list[DeploymentRecord],
        prediction_archives: list[PredictionArchiveRecord],
        gpu_snapshot: dict[str, Any],
        protect_active_window_seconds: int,
        gpu_memory_restart_enabled: bool,
        gpu_memory_restart_threshold_mb: int,
    ) -> list[DeploymentPolicyAutomationAction]:
        if not gpu_memory_restart_enabled:
            return []

        hotspots = _collect_gpu_memory_hotspots(
            deployments=deployments,
            gpu_snapshot=gpu_snapshot,
            threshold_mb=float(gpu_memory_restart_threshold_mb),
        )
        if not hotspots:
            return []

        inference_activity_tracker = self.get_inference_activity_tracker()
        actions: list[DeploymentPolicyAutomationAction] = []
        for hotspot in hotspots[:5]:
            protection = _get_model_protection_status(
                hotspot.model_id,
                inference_activity_tracker=inference_activity_tracker,
                prediction_archives=prediction_archives,
                protect_active_window_seconds=protect_active_window_seconds,
            )
            if protection.protected:
                actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="restart_high_gpu_memory",
                        model_id=hotspot.model_id,
                        deployment_id=hotspot.deployment_id,
                        status="busy",
                        detail=(
                            f"deployment {hotspot.deployment_id} 当前显存占用约 "
                            f"{hotspot.used_gpu_memory_mb:.0f} MB，已超过阈值 "
                            f"{gpu_memory_restart_threshold_mb} MB，但{protection.detail}"
                        ),
                    )
                )
                continue

            action = self._restart_deployment_for_gpu_memory_hotspot(
                hotspot=hotspot,
                threshold_mb=gpu_memory_restart_threshold_mb,
            )
            actions.append(action)
        return actions

    def _restart_deployment_for_gpu_memory_hotspot(
        self,
        *,
        hotspot: _GpuMemoryHotspot,
        threshold_mb: int,
    ) -> DeploymentPolicyAutomationAction:
        deployment_store = self.get_deployment_store()
        current = next(
            (item for item in deployment_store.list() if item.deployment_id == hotspot.deployment_id),
            None,
        )
        if current is None:
            return DeploymentPolicyAutomationAction(
                action_type="restart_high_gpu_memory",
                model_id=hotspot.model_id,
                deployment_id=hotspot.deployment_id,
                status="failed",
                detail="显存巡检命中该 deployment，但执行重启前已找不到对应部署记录。",
            )
        if not current.use_gpu:
            return DeploymentPolicyAutomationAction(
                action_type="restart_high_gpu_memory",
                model_id=current.model_id,
                deployment_id=current.deployment_id,
                status="skipped",
                detail="该 deployment 当前已不在 GPU 上运行，跳过显存超阈值重启。",
            )
        if not current.manifest_path or not Path(current.manifest_path).exists():
            return DeploymentPolicyAutomationAction(
                action_type="restart_high_gpu_memory",
                model_id=current.model_id,
                deployment_id=current.deployment_id,
                status="failed",
                detail="deployment 缺少有效 manifest_path，无法执行显存超阈值重启。",
            )
        if not current.profile_path or not Path(current.profile_path).exists():
            return DeploymentPolicyAutomationAction(
                action_type="restart_high_gpu_memory",
                model_id=current.model_id,
                deployment_id=current.deployment_id,
                status="failed",
                detail="deployment 缺少有效 profile_path，无法执行显存超阈值重启。",
            )

        previous_pid = current.pid
        previous_status = current.status
        service_url = deployment_store.ensure_service_url(current)
        current.log_path = current.log_path or str(self.settings.logs_root / "deployments" / f"{current.deployment_id}.log")
        port = int(service_url.rsplit(":", maxsplit=1)[-1])

        try:
            if current.pid:
                stop_runtime_process(current.pid)

            pid, service_url = launch_runtime_process(
                root_dir=self.settings.root_dir,
                manifest_path=current.manifest_path,
                profile_path=current.profile_path,
                port=port,
                log_path=current.log_path,
                gpu_ids=current.gpu_ids,
                use_gpu=current.use_gpu,
            )
            current.pid = pid
            current.service_url = service_url
            current.status = "running" if wait_for_runtime(service_url) else "starting"
            current.notes = " | ".join(
                item
                for item in [
                    current.notes,
                    f"auto_restart_gpu_memory_over_{threshold_mb}mb",
                ]
                if item
            )
            deployment_store.update(current)
            return DeploymentPolicyAutomationAction(
                action_type="restart_high_gpu_memory",
                model_id=current.model_id,
                deployment_id=current.deployment_id,
                status="restarted",
                detail=(
                    f"检测到 deployment {current.deployment_id} 在 GPU "
                    f"{','.join(str(item) for item in hotspot.gpu_ids) or '-'} 上的显存占用约 "
                    f"{hotspot.used_gpu_memory_mb:.0f} MB，超过阈值 {threshold_mb} MB，"
                    "已自动重启该部署服务。"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            current.pid = previous_pid
            current.status = previous_status
            deployment_store.update(current)
            return DeploymentPolicyAutomationAction(
                action_type="restart_high_gpu_memory",
                model_id=current.model_id,
                deployment_id=current.deployment_id,
                status="failed",
                detail=f"显存超阈值自动重启失败：{type(exc).__name__}: {exc}",
            )

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
            deployment.service_url = deployment_store.ensure_service_url(deployment)
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
        if any(
            list(item.gpu_ids or []) == target_gpu_ids and item.profile_name == recommendation.recommended_profile_name
            for item in active_gpu_deployments
        ):
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
            deployment.service_url = deployment_store.ensure_service_url(deployment)
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

    def _apply_resource_recovery(
        self,
        *,
        context: _SchedulerContext,
        restarted_model_ids: set[str],
    ) -> list[DeploymentPolicyAutomationAction]:
        if not _gpu_has_recovery_capacity(context.policy.gpu_snapshot):
            return []

        actions: list[DeploymentPolicyAutomationAction] = []
        promoted_count = 0
        recovered_profile_count = 0
        for recommendation in context.policy.recommendations:
            if recommendation.model_id in restarted_model_ids:
                continue

            if promoted_count < 3 and recommendation.recommended_use_gpu and recommendation.service_tier in {"hot", "warm"}:
                promote_action = self._promote_cpu_deployment_to_gpu(
                    recommendation=recommendation,
                    context=context,
                    action_type="recover_to_gpu",
                    success_detail=(
                        f"当前 GPU 资源已恢复，依据最新部署推荐，"
                        f"已将 deployment 从 CPU 回迁到 GPU {','.join(str(item) for item in recommendation.recommended_gpu_ids)}。"
                    ),
                )
                if promote_action is not None:
                    actions.append(promote_action)
                    if promote_action.status == "migrated":
                        promoted_count += 1
                        continue

            if recovered_profile_count < 3:
                profile_action = self._apply_profile_alignment(
                    recommendation=recommendation,
                    context=context,
                    action_type="recover_profile",
                )
                if profile_action is not None:
                    actions.append(profile_action)
                    if profile_action.status == "migrated":
                        recovered_profile_count += 1
        return actions

    def _relocate_deployment_for_gpu_pool_change(
        self,
        *,
        deployment: DeploymentRecord,
        recommendation: DeploymentPolicyRecommendation | None,
        context: _SchedulerContext,
    ) -> DeploymentPolicyAutomationAction:
        protection = _get_model_protection_status(
            deployment.model_id,
            inference_activity_tracker=context.inference_activity_tracker,
            prediction_archives=context.prediction_archives,
            protect_active_window_seconds=context.protect_active_window_seconds,
        )
        if protection.protected:
            return DeploymentPolicyAutomationAction(
                action_type="gpu_pool_scope_change",
                model_id=deployment.model_id,
                deployment_id=deployment.deployment_id,
                status="busy",
                detail=(
                    f"deployment {deployment.deployment_id} 当前绑定的 GPU 已被平台禁用，但{protection.detail}"
                ),
            )

        registry = self.get_registry()
        model: ModelDetail = registry.get_model(deployment.model_id)
        if (
            recommendation is not None
            and recommendation.recommended_use_gpu
            and recommendation.recommended_gpu_ids
        ):
            return self._reconfigure_deployment_target(
                deployment=deployment,
                model=model,
                profile_name=recommendation.recommended_profile_name,
                use_gpu=True,
                gpu_ids=[int(item) for item in recommendation.recommended_gpu_ids],
                action_type="gpu_pool_reassign_gpu",
                success_detail=(
                    f"deployment {deployment.deployment_id} 原绑定 GPU 已不在平台允许范围内，"
                    f"已迁移到 GPU {','.join(str(item) for item in recommendation.recommended_gpu_ids)}。"
                ),
            )

        fallback_profile_name = (
            recommendation.fallback_profile_name
            if recommendation is not None and recommendation.fallback_profile_name
            else ("memory" if "memory" in model.resolved_profiles else "default")
        )
        return self._reconfigure_deployment_target(
            deployment=deployment,
            model=model,
            profile_name=fallback_profile_name,
            use_gpu=False,
            gpu_ids=[],
            action_type="gpu_pool_fallback_cpu",
            success_detail=(
                f"deployment {deployment.deployment_id} 原绑定 GPU 已不在平台允许范围内，"
                "当前没有合适的目标 GPU，已迁移到 CPU 保持可用。"
            ),
        )

    def _apply_follow_up_gpu_pool_rebalance(
        self,
        *,
        policy: DeploymentPolicyResponse,
        context: _SchedulerContext,
        excluded_model_ids: set[str],
    ) -> list[DeploymentPolicyAutomationAction]:
        actions: list[DeploymentPolicyAutomationAction] = []
        applied_count = 0
        for recommendation in policy.recommendations:
            if recommendation.model_id in excluded_model_ids:
                continue
            if not recommendation.recommended_use_gpu or not recommendation.recommended_gpu_ids:
                continue
            if recommendation.service_tier not in {"hot", "warm"}:
                continue
            action = self._apply_gpu_rebalance(recommendation, context=context)
            if action is None:
                action = self._promote_cpu_deployment_to_gpu(
                    recommendation=recommendation,
                    context=context,
                )
            if action is None:
                continue
            actions.append(action)
            if action.status == "migrated":
                applied_count += 1
            if applied_count >= 3:
                break
        return actions

    def _reconfigure_deployment_target(
        self,
        *,
        deployment: DeploymentRecord,
        model: ModelDetail,
        profile_name: str,
        use_gpu: bool,
        gpu_ids: list[int],
        action_type: str,
        success_detail: str,
    ) -> DeploymentPolicyAutomationAction:
        deployment_store = self.get_deployment_store()
        previous_profile_name = deployment.profile_name
        previous_status = deployment.status
        previous_gpu_ids = list(deployment.gpu_ids or [])
        previous_use_gpu = bool(deployment.use_gpu)
        previous_pid = deployment.pid
        previous_provider_preference = deployment.provider_preference
        previous_profile_path = deployment.profile_path
        previous_profile_runtime = deployment.profile_runtime
        previous_device = deployment.device
        previous_execution = dict(deployment.execution or {})
        previous_runtime_revision = deployment.runtime_revision
        previous_revision_started_at = deployment.revision_started_at
        previous_previous_revision_started_at = deployment.previous_revision_started_at
        previous_revision_change_level = deployment.revision_change_level
        previous_runtime_signature = deployment.runtime_signature

        if profile_name not in model.resolved_profiles:
            profile_name = (
                deployment.profile_name
                if deployment.profile_name in model.resolved_profiles
                else ("default" if "default" in model.resolved_profiles else next(iter(model.resolved_profiles)))
            )

        should_restart_runtime = bool(deployment.pid) or deployment.status in {"running", "starting"}
        if use_gpu and not (previous_use_gpu and sorted(previous_gpu_ids) == sorted(int(item) for item in gpu_ids)):
            gpu_assignment_error = _validate_gpu_assignment(
                model=model,
                gpu_ids=gpu_ids,
                gpu_snapshot=self.get_gpu_snapshot(),
            )
            if gpu_assignment_error is not None:
                return DeploymentPolicyAutomationAction(
                    action_type=action_type,
                    model_id=deployment.model_id,
                    deployment_id=deployment.deployment_id,
                    status="skipped",
                    detail=gpu_assignment_error,
                )

        try:
            target_profile_path = model.resolved_profiles[profile_name]
            target_provider_preference = "cuda" if use_gpu else "cpu"
            change_level, target_runtime_signature = classify_runtime_change(
                deployment,
                manifest_path=model.summary.package_root + "/manifest.yaml",
                profile_name=profile_name,
                profile_path=target_profile_path,
                use_gpu=bool(use_gpu),
                gpu_ids=[int(item) for item in gpu_ids] if use_gpu else [],
                provider_preference=target_provider_preference,
            )
            if deployment.pid:
                stop_runtime_process(deployment.pid)

            deployment.manifest_path = model.summary.package_root + "/manifest.yaml"
            deployment.profile_name = profile_name
            deployment.profile_path = target_profile_path
            deployment.use_gpu = bool(use_gpu)
            deployment.gpu_ids = [int(item) for item in gpu_ids] if use_gpu else []
            deployment.provider_preference = target_provider_preference
            try:
                profile = load_profile(target_profile_path)
                deployment.profile_runtime = profile.runtime
                deployment.execution = dict(profile.execution or {})
                deployment.device = "cuda" if deployment.use_gpu else str(profile.device or "cpu")
            except Exception:  # noqa: BLE001
                deployment.profile_runtime = None
                deployment.execution = {}
                deployment.device = "cuda" if deployment.use_gpu else "cpu"
            deployment.log_path = deployment.log_path or str(self.settings.logs_root / "deployments" / f"{deployment.deployment_id}.log")
            deployment.service_url = deployment_store.ensure_service_url(deployment)
            deployment.notes = " | ".join(
                item
                for item in [
                    deployment.notes,
                    action_type,
                ]
                if item
            )

            if should_restart_runtime and model.summary.predict_ready:
                port = int(deployment.service_url.rsplit(":", maxsplit=1)[-1])
                pid, service_url = launch_runtime_process(
                    root_dir=self.settings.root_dir,
                    manifest_path=deployment.manifest_path,
                    profile_path=deployment.profile_path,
                    port=port,
                    log_path=deployment.log_path,
                    gpu_ids=deployment.gpu_ids,
                    use_gpu=deployment.use_gpu,
                    provider_preference=deployment.provider_preference,
                )
                deployment.pid = pid
                deployment.service_url = service_url
                deployment.status = "running" if wait_for_runtime(service_url) else "starting"
            elif should_restart_runtime:
                deployment.pid = None
                deployment.status = "planned"

            deployment = initialize_deployment_revision_metadata(deployment)
            if change_level != "none":
                deployment = advance_deployment_revision(
                    deployment,
                    change_level=change_level,
                    runtime_signature=target_runtime_signature,
                )
            deployment_store.update(deployment)
            return DeploymentPolicyAutomationAction(
                action_type=action_type,
                model_id=deployment.model_id,
                deployment_id=deployment.deployment_id,
                status="migrated",
                detail=success_detail,
            )
        except Exception as exc:  # noqa: BLE001
            deployment.profile_name = previous_profile_name
            deployment.profile_path = previous_profile_path
            deployment.use_gpu = previous_use_gpu
            deployment.gpu_ids = previous_gpu_ids
            deployment.provider_preference = previous_provider_preference
            deployment.profile_runtime = previous_profile_runtime
            deployment.device = previous_device
            deployment.execution = previous_execution
            deployment.pid = previous_pid
            deployment.status = previous_status
            deployment.runtime_revision = previous_runtime_revision
            deployment.revision_started_at = previous_revision_started_at
            deployment.previous_revision_started_at = previous_previous_revision_started_at
            deployment.revision_change_level = previous_revision_change_level
            deployment.runtime_signature = previous_runtime_signature
            deployment_store.update(deployment)
            return DeploymentPolicyAutomationAction(
                action_type=action_type,
                model_id=deployment.model_id,
                deployment_id=deployment.deployment_id,
                status="failed",
                detail=f"Deployment 优化执行失败：{type(exc).__name__}: {exc}",
            )

    def _promote_cpu_deployment_to_gpu(
        self,
        *,
        recommendation: DeploymentPolicyRecommendation,
        context: _SchedulerContext,
        action_type: str = "gpu_pool_promote_gpu",
        success_detail: str | None = None,
    ) -> DeploymentPolicyAutomationAction | None:
        candidate_deployments = [
            item
            for item in context.deployments
            if item.model_id == recommendation.model_id
            and item.status in {"running", "starting", "planned"}
            and not item.use_gpu
        ]
        if not candidate_deployments:
            return None

        protection = _get_model_protection_status(
            recommendation.model_id,
            inference_activity_tracker=context.inference_activity_tracker,
            prediction_archives=context.prediction_archives,
            protect_active_window_seconds=context.protect_active_window_seconds,
        )
        if protection.protected:
            return DeploymentPolicyAutomationAction(
                action_type=action_type,
                model_id=recommendation.model_id,
                deployment_id=candidate_deployments[0].deployment_id,
                status="busy",
                detail=protection.detail,
            )

        registry = self.get_registry()
        model: ModelDetail = registry.get_model(recommendation.model_id)
        deployment = sorted(candidate_deployments, key=lambda item: item.deployment_id)[0]
        return self._reconfigure_deployment_target(
            deployment=deployment,
            model=model,
            profile_name=recommendation.recommended_profile_name,
            use_gpu=True,
            gpu_ids=[int(item) for item in recommendation.recommended_gpu_ids],
            action_type=action_type,
            success_detail=success_detail
            or (
                f"检测到平台重新开放 GPU {','.join(str(item) for item in recommendation.recommended_gpu_ids)}，"
                f"已将 deployment {deployment.deployment_id} 从 CPU 回迁到 GPU。"
            ),
        )

    def _apply_profile_alignment(
        self,
        *,
        recommendation: DeploymentPolicyRecommendation,
        context: _SchedulerContext,
        action_type: str,
    ) -> DeploymentPolicyAutomationAction | None:
        if recommendation.recommended_use_gpu:
            candidate_deployments = [
                item
                for item in context.deployments
                if item.model_id == recommendation.model_id
                and item.status in {"running", "starting", "planned"}
                and item.use_gpu
                and list(item.gpu_ids or []) == [int(gpu_id) for gpu_id in recommendation.recommended_gpu_ids]
                and item.profile_name != recommendation.recommended_profile_name
            ]
        else:
            candidate_deployments = [
                item
                for item in context.deployments
                if item.model_id == recommendation.model_id
                and item.status in {"running", "starting", "planned"}
                and not item.use_gpu
                and item.profile_name != recommendation.recommended_profile_name
            ]

        if not candidate_deployments:
            return None

        protection = _get_model_protection_status(
            recommendation.model_id,
            inference_activity_tracker=context.inference_activity_tracker,
            prediction_archives=context.prediction_archives,
            protect_active_window_seconds=context.protect_active_window_seconds,
        )
        if protection.protected:
            return DeploymentPolicyAutomationAction(
                action_type=action_type,
                model_id=recommendation.model_id,
                deployment_id=candidate_deployments[0].deployment_id,
                status="busy",
                detail=protection.detail,
            )

        registry = self.get_registry()
        model: ModelDetail = registry.get_model(recommendation.model_id)
        deployment = sorted(candidate_deployments, key=lambda item: item.deployment_id)[0]
        target_device = (
            f"GPU {','.join(str(item) for item in recommendation.recommended_gpu_ids)}"
            if recommendation.recommended_use_gpu and recommendation.recommended_gpu_ids
            else "CPU"
        )
        return self._reconfigure_deployment_target(
            deployment=deployment,
            model=model,
            profile_name=recommendation.recommended_profile_name,
            use_gpu=bool(recommendation.recommended_use_gpu),
            gpu_ids=[int(item) for item in recommendation.recommended_gpu_ids] if recommendation.recommended_use_gpu else [],
            action_type=action_type,
            success_detail=(
                f"当前资源已恢复且最新部署推荐发生变化，"
                f"已将 deployment {deployment.deployment_id} 在 {target_device} 上的 profile 调整为 "
                f"{recommendation.recommended_profile_name}。"
            ),
        )

    def _refresh_benchmarks_for_actions(
        self,
        actions: list[DeploymentPolicyAutomationAction],
    ) -> list[DeploymentPolicyAutomationAction]:
        benchmark_trigger_actions = {
            "rebalance_gpu",
            "migrate_to_cpu",
            "recover_to_gpu",
            "recover_profile",
            "gpu_pool_reassign_gpu",
            "gpu_pool_fallback_cpu",
            "gpu_pool_promote_gpu",
        }
        eligible_actions = [
            item
            for item in actions
            if item.action_type in benchmark_trigger_actions
            and item.status == "migrated"
            and item.deployment_id
        ]
        if not eligible_actions:
            return []

        deployment_store = self.get_deployment_store()
        benchmark_store = self.get_benchmark_store()
        registry = self.get_registry()
        deployments_by_id = {
            item.deployment_id: item
            for item in deployment_store.list()
            if item.deployment_id
        }
        refresh_actions: list[DeploymentPolicyAutomationAction] = []
        seen_deployment_ids: set[str] = set()
        for action in eligible_actions:
            deployment_id = str(action.deployment_id or "").strip()
            if not deployment_id or deployment_id in seen_deployment_ids:
                continue
            seen_deployment_ids.add(deployment_id)
            deployment = deployments_by_id.get(deployment_id)
            if deployment is None:
                refresh_actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="refresh_benchmark",
                        model_id=action.model_id,
                        deployment_id=deployment_id,
                        status="failed",
                        detail="部署优化已生效，但自动刷新 benchmark 时未找到对应 deployment。",
                    )
                )
                continue
            if deployment.status != "running" or not deployment.service_url:
                refresh_actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="refresh_benchmark",
                        model_id=deployment.model_id,
                        deployment_id=deployment_id,
                        status="skipped",
                        detail="deployment 当前未处于 running 状态，跳过自动 benchmark 刷新。",
                    )
                )
                continue

            try:
                model_summary = registry.get_model(deployment.model_id).summary
                scene = infer_benchmark_scene(model=model_summary)
                request = BenchmarkCreateRequest(
                    deployment_id=deployment.deployment_id,
                    text=default_benchmark_text(scene),
                    iterations=4,
                    warmup_runs=1,
                    timeout_seconds=30.0,
                    label="自动部署刷新基准",
                    scene=scene,
                    trigger_source="post_policy_adjustment",
                    options={},
                )
                record = benchmark_store.create(
                    request,
                    model_id=deployment.model_id,
                    profile_name=deployment.profile_name,
                    service_url=deployment.service_url,
                    deployment=deployment,
                    model=model_summary,
                )
                batch_size = 1
                if deployment.profile_path:
                    profile = load_profile(deployment.profile_path)
                    batch_size = max(1, int(profile.execution.get("batch_size", 1) or 1))
                finished = run_benchmark(request, record, batch_size=batch_size)
                benchmark_store.update(finished)
                refresh_actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="refresh_benchmark",
                        model_id=deployment.model_id,
                        deployment_id=deployment.deployment_id,
                        status="applied" if finished.status == "completed" else "failed",
                        detail=(
                            f"部署方案已变化，平台已自动补跑 {scene} 场景 benchmark。"
                            f" 当前结果状态：{finished.status}。"
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                refresh_actions.append(
                    DeploymentPolicyAutomationAction(
                        action_type="refresh_benchmark",
                        model_id=deployment.model_id,
                        deployment_id=deployment.deployment_id,
                        status="failed",
                        detail=f"自动刷新 benchmark 失败：{type(exc).__name__}: {exc}",
                    )
                )
        return refresh_actions


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
    devices = _get_platform_enabled_gpu_devices(gpu_snapshot)
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
    devices = _get_platform_enabled_gpu_devices(gpu_snapshot)
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


def _get_platform_enabled_gpu_devices(gpu_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in list(gpu_snapshot.get("devices") or []) if item.get("platform_enabled", True)]


def _gpu_has_recovery_capacity(gpu_snapshot: dict[str, Any]) -> bool:
    devices = _get_platform_enabled_gpu_devices(gpu_snapshot)
    for item in devices:
        total = float(item.get("memory_total_mb") or 0.0)
        used = float(item.get("memory_used_mb") or 0.0)
        free_mb = max(total - used, 0.0)
        free_ratio = (free_mb / total) if total > 0 else 0.0
        utilization = float(item.get("utilization_gpu_pct") or 0.0)
        if free_mb >= 4096.0 and free_ratio >= 0.2 and utilization <= 75.0:
            return True
    return False


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


def _collect_gpu_memory_hotspots(
    *,
    deployments: list[DeploymentRecord],
    gpu_snapshot: dict[str, Any],
    threshold_mb: float,
) -> list[_GpuMemoryHotspot]:
    if not gpu_snapshot.get("available"):
        return []

    deployment_map = {
        item.deployment_id: item
        for item in deployments
        if item.deployment_id and item.use_gpu and item.status in {"running", "starting"}
    }
    aggregated: dict[str, _GpuMemoryHotspot] = {}
    for device in gpu_snapshot.get("devices") or []:
        gpu_index = int(device.get("index") or 0)
        for process in device.get("active_processes") or []:
            deployment_id = str(process.get("deployment_id") or "").strip()
            if not deployment_id:
                continue
            deployment = deployment_map.get(deployment_id)
            if deployment is None:
                continue
            used_mb = float(process.get("used_gpu_memory_mb") or 0.0)
            if deployment_id not in aggregated:
                aggregated[deployment_id] = _GpuMemoryHotspot(
                    deployment_id=deployment_id,
                    model_id=deployment.model_id,
                    pid=int(process.get("pid") or deployment.pid or 0),
                    used_gpu_memory_mb=used_mb,
                    gpu_ids=[gpu_index],
                )
            else:
                aggregated[deployment_id].used_gpu_memory_mb += used_mb
                if gpu_index not in aggregated[deployment_id].gpu_ids:
                    aggregated[deployment_id].gpu_ids.append(gpu_index)

    return sorted(
        [
            item
            for item in aggregated.values()
            if item.used_gpu_memory_mb > threshold_mb
        ],
        key=lambda item: item.used_gpu_memory_mb,
        reverse=True,
    )


def _validate_gpu_assignment(
    *,
    model: ModelDetail,
    gpu_ids: list[int],
    gpu_snapshot: dict[str, Any],
) -> str | None:
    normalized_gpu_ids = [int(item) for item in gpu_ids]
    if not normalized_gpu_ids:
        return "当前优化动作需要使用 GPU，但没有提供目标 GPU 编号。"

    devices = {
        int(item.get("index")): item
        for item in _get_platform_enabled_gpu_devices(gpu_snapshot)
        if item.get("index") is not None
    }
    missing_gpu_ids = [item for item in normalized_gpu_ids if item not in devices]
    if missing_gpu_ids:
        return f"目标 GPU {','.join(str(item) for item in missing_gpu_ids)} 当前不在平台允许范围内，跳过本轮优化。"

    required_memory_mb = _estimate_required_gpu_memory_mb(
        package_root=model.summary.package_root,
        stage_count=model.summary.stage_count,
    )
    for gpu_id in normalized_gpu_ids:
        device = devices[gpu_id]
        free_mb = max(
            float(device.get("memory_total_mb") or 0.0) - float(device.get("memory_used_mb") or 0.0),
            0.0,
        )
        if free_mb < required_memory_mb:
            return (
                f"目标 GPU {gpu_id} 当前空闲显存约 {free_mb:.0f} MB，"
                f"低于该模型预计需求 {required_memory_mb:.0f} MB，跳过本轮优化。"
            )
    return None


def _estimate_required_gpu_memory_mb(*, package_root: str | None, stage_count: int) -> float:
    package_size_mb = _directory_size_mb(package_root)
    base = max(1536.0, (package_size_mb or 512.0) * 1.8)
    stage_overhead = max(256.0, float(stage_count or 1) * 320.0)
    return base + stage_overhead


def _directory_size_mb(path: str | None) -> float | None:
    if not path or not os.path.isdir(path):
        return None
    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                total_bytes += os.path.getsize(file_path)
            except OSError:
                continue
    return total_bytes / (1024 * 1024)
