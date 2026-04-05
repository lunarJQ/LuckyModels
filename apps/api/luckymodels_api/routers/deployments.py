import json
import os
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from luckymodels_core.deployment_revision import (
    advance_deployment_revision,
    classify_runtime_change,
    initialize_deployment_revision_metadata,
)
from luckymodels_core.benchmark_evidence import default_benchmark_text, infer_benchmark_scene
from luckymodels_core.benchmark_runner import run_benchmark
from luckymodels_core.dataset_samples import get_dataset_random_sample_text
from luckymodels_core.launcher import (
    launch_runtime_process,
    process_is_running,
    stop_runtime_process,
    tail_log,
    wait_for_runtime,
)
from luckymodels_core.manifest import load_profile
from luckymodels_core.model_package_manager import read_package_metadata
from luckymodels_core.runtime_capabilities import inspect_runtime_capabilities
from luckymodels_core.schemas import BenchmarkCreateRequest, DeploymentCreateRequest, DeploymentRecord
from luckymodels_core.settings import get_settings
from luckymodels_core.strategy_engine import build_deployment_policy

from ..dependencies import (
    get_benchmark_store,
    get_deployment_store,
    get_inference_activity_tracker,
    get_prediction_store,
    get_registry,
    get_task_routing_feedback_store,
)
from ..permissions import require_permission
from .system import get_gpu_status

router = APIRouter(tags=["deployments"])


class DeploymentPredictRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    scene: str | None = None
    options: dict[str, object] | None = None


@router.get("/deployments", response_model=list[DeploymentRecord])
def list_deployments() -> list[DeploymentRecord]:
    require_permission("deployments.view")
    return [_hydrate_deployment_runtime_snapshot(item) for item in get_deployment_store().list()]


@router.post("/deployments", response_model=DeploymentRecord)
def create_deployment(request: DeploymentCreateRequest) -> DeploymentRecord:
    require_permission("deployments.manage")
    registry = get_registry()
    try:
        model = registry.get_model(request.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request.launch_runtime and not model.summary.predict_ready:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model {request.model_id} is package-ready only. "
                "It can be listed and inspected in LuckyModels, but live predict "
                "runtime is not enabled for this model yet."
            ),
        )

    effective_request, policy_adaptation_note = _adapt_deployment_request_with_policy(
        request=request,
        model=model,
    )

    if effective_request.profile_name not in model.resolved_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Profile not found for model {request.model_id}: {effective_request.profile_name}",
        )

    store = get_deployment_store()
    deployment = store.create(effective_request)
    deployment.manifest_path = model.summary.package_root + "/manifest.yaml"
    deployment.profile_path = model.resolved_profiles[effective_request.profile_name]
    deployment.service_url = store.ensure_service_url(deployment)
    settings = get_settings()
    deployment.log_path = str(settings.logs_root / "deployments" / f"{deployment.deployment_id}.log")
    deployment = _hydrate_deployment_runtime_snapshot(deployment)
    deployment.use_gpu, deployment.gpu_ids, device_note = _resolve_runtime_device(
        model=model,
        requested_use_gpu=deployment.use_gpu,
        requested_gpu_ids=deployment.gpu_ids,
        profile_path=deployment.profile_path,
    )
    deployment.provider_preference = _resolve_provider_preference(
        requested_preference=request.provider_preference,
        use_gpu=deployment.use_gpu,
        settings=settings,
    )
    deployment.device = _resolve_display_device(
        use_gpu=deployment.use_gpu,
        provider_preference=deployment.provider_preference,
        profile_device=deployment.device,
        gpu_ids=deployment.gpu_ids,
    )
    deployment = initialize_deployment_revision_metadata(deployment)
    _validate_runtime_backend_choice(
        settings=settings,
        use_gpu=deployment.use_gpu,
        provider_preference=deployment.provider_preference,
    )
    if device_note:
        deployment.notes = " | ".join(item for item in [deployment.notes, device_note] if item)
    if policy_adaptation_note:
        deployment.notes = " | ".join(item for item in [deployment.notes, policy_adaptation_note] if item)

    if effective_request.launch_runtime:
        port = int(deployment.service_url.rsplit(":", maxsplit=1)[-1])
        pid, service_url = launch_runtime_process(
            root_dir=settings.root_dir,
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
    else:
        deployment.status = "planned"

    # Deployment creation now stays manual-only for benchmark execution.
    # Users can launch benchmark runs explicitly from the benchmark page.
    return store.update(deployment)


@router.get("/deployments/{deployment_id}/health")
def check_deployment_health(deployment_id: str) -> dict[str, object]:
    require_permission("deployments.view")
    deployment = _get_deployment_or_404(deployment_id)
    pid_alive = process_is_running(deployment.pid)
    if not deployment.service_url:
        return {
            "deployment_id": deployment_id,
            "reachable": False,
            "pid_alive": pid_alive,
            "detail": "missing service_url",
        }
    try:
        with urlopen(deployment.service_url + "/healthz", timeout=2) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "deployment_id": deployment_id,
            "reachable": True,
            "pid_alive": pid_alive,
            "service_url": deployment.service_url,
            "log_path": deployment.log_path,
            "payload": payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "deployment_id": deployment_id,
            "reachable": False,
            "pid_alive": pid_alive,
            "service_url": deployment.service_url,
            "log_path": deployment.log_path,
            "detail": str(exc),
        }


@router.get("/deployments/{deployment_id}/logs")
def get_deployment_logs(deployment_id: str, lines: int = 100) -> dict[str, object]:
    require_permission("resources.view")
    deployment = _get_deployment_or_404(deployment_id)
    return {
        "deployment_id": deployment_id,
        "log_path": deployment.log_path,
        "content": tail_log(deployment.log_path, lines=lines),
    }


@router.post("/deployments/{deployment_id}/predict")
def predict_with_deployment(
    deployment_id: str,
    request: DeploymentPredictRequest,
) -> dict[str, object]:
    require_permission("predictions.view")
    deployment = _get_deployment_or_404(deployment_id)
    if not deployment.service_url:
        raise HTTPException(status_code=400, detail="Deployment is missing a service_url")

    payload = json.dumps(request.model_dump(), ensure_ascii=False).encode("utf-8")
    runtime_request = Request(
        deployment.service_url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    activity_token = get_inference_activity_tracker().begin(
        model_id=deployment.model_id,
        deployment_id=deployment.deployment_id,
        source="deployment_predict",
    )
    try:
        with urlopen(runtime_request, timeout=90) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            archive = get_prediction_store().create(
                deployment_id=deployment.deployment_id,
                model_id=deployment.model_id,
                profile_name=deployment.profile_name,
                deployment=deployment,
                service_url=deployment.service_url,
                scene=request.scene,
                request_text=request.text,
                request_texts=list(request.texts or []),
                options=dict(request.options or {}),
                response_payload=payload,
            )
            payload["archive_id"] = archive.archive_id
        return payload
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        get_inference_activity_tracker().end(activity_token)


@router.post("/deployments/{deployment_id}/stop", response_model=DeploymentRecord)
def stop_deployment(deployment_id: str) -> DeploymentRecord:
    require_permission("deployments.manage")
    store = get_deployment_store()
    deployment = _get_deployment_or_404(deployment_id)
    stopped = stop_runtime_process(deployment.pid)
    if stopped:
        deployment.status = "stopped"
        deployment.pid = None
    else:
        deployment.status = "stopping"
    return store.update(deployment)


@router.post("/deployments/{deployment_id}/restart", response_model=DeploymentRecord)
def restart_deployment(deployment_id: str) -> DeploymentRecord:
    require_permission("deployments.manage")
    store = get_deployment_store()
    deployment = _get_deployment_or_404(deployment_id)
    registry = get_registry()
    model = registry.get_model(deployment.model_id)

    if deployment.pid:
        stop_runtime_process(deployment.pid)

    if not deployment.manifest_path or not Path(deployment.manifest_path).exists():
        raise HTTPException(status_code=400, detail="Deployment manifest_path is missing or invalid")
    if not deployment.profile_path or not Path(deployment.profile_path).exists():
        raise HTTPException(status_code=400, detail="Deployment profile_path is missing or invalid")

    resolved_use_gpu, resolved_gpu_ids, device_note = _resolve_runtime_device(
        model=model,
        requested_use_gpu=deployment.use_gpu,
        requested_gpu_ids=deployment.gpu_ids,
        profile_path=deployment.profile_path,
    )
    settings = get_settings()
    resolved_provider_preference = _resolve_provider_preference(
        requested_preference=deployment.provider_preference,
        use_gpu=resolved_use_gpu,
        settings=settings,
    )
    change_level, runtime_signature = classify_runtime_change(
        deployment,
        manifest_path=deployment.manifest_path,
        profile_name=deployment.profile_name,
        profile_path=deployment.profile_path,
        use_gpu=resolved_use_gpu,
        gpu_ids=resolved_gpu_ids,
        provider_preference=resolved_provider_preference,
    )
    deployment.use_gpu = resolved_use_gpu
    deployment.gpu_ids = resolved_gpu_ids
    deployment.provider_preference = resolved_provider_preference
    deployment.device = _resolve_display_device(
        use_gpu=deployment.use_gpu,
        provider_preference=deployment.provider_preference,
        profile_device=deployment.device,
        gpu_ids=deployment.gpu_ids,
    )
    _validate_runtime_backend_choice(
        settings=settings,
        use_gpu=deployment.use_gpu,
        provider_preference=deployment.provider_preference,
    )
    if device_note:
        deployment.notes = " | ".join(item for item in [deployment.notes, device_note] if item)

    service_url = store.ensure_service_url(deployment)
    port = int(service_url.rsplit(":", maxsplit=1)[-1])
    pid, service_url = launch_runtime_process(
        root_dir=settings.root_dir,
        manifest_path=deployment.manifest_path,
        profile_path=deployment.profile_path,
        port=port,
        log_path=deployment.log_path,
        gpu_ids=deployment.gpu_ids,
        use_gpu=deployment.use_gpu,
        provider_preference=deployment.provider_preference,
    )
    deployment = _hydrate_deployment_runtime_snapshot(deployment)
    if change_level != "none":
        deployment = advance_deployment_revision(
            deployment,
            change_level=change_level,
            runtime_signature=runtime_signature,
        )
    else:
        deployment = initialize_deployment_revision_metadata(deployment)
    deployment.pid = pid
    deployment.service_url = service_url
    deployment.status = "running" if wait_for_runtime(service_url) else "starting"
    return store.update(deployment)


@router.delete("/deployments/{deployment_id}", response_model=DeploymentRecord)
def delete_deployment(deployment_id: str) -> DeploymentRecord:
    require_permission("deployments.manage")
    store = get_deployment_store()
    deployment = _get_deployment_or_404(deployment_id)
    if deployment.pid:
        stop_runtime_process(deployment.pid)
        deployment.pid = None
    deployment.status = "deleted"
    deleted = store.delete(deployment_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")
    deleted.status = "deleted"
    deleted.pid = None
    return deleted


def _get_deployment_or_404(deployment_id: str) -> DeploymentRecord:
    store = get_deployment_store()
    for deployment in store.list():
        if deployment.deployment_id == deployment_id:
            return _hydrate_deployment_runtime_snapshot(deployment)
    raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")


def _hydrate_deployment_runtime_snapshot(deployment: DeploymentRecord) -> DeploymentRecord:
    if not deployment.profile_path:
        return deployment

    profile_file = Path(deployment.profile_path)
    if not profile_file.exists():
        return deployment

    try:
        profile = load_profile(profile_file)
    except Exception:  # noqa: BLE001
        return deployment

    deployment.profile_runtime = profile.runtime
    if not deployment.gpu_ids and profile.gpu_ids:
        deployment.gpu_ids = list(profile.gpu_ids)
    if not deployment.provider_preference:
        deployment.provider_preference = str(profile.execution.get("provider_preference", "") or "").strip() or None
    deployment.device = _resolve_display_device(
        use_gpu=deployment.use_gpu,
        provider_preference=deployment.provider_preference,
        profile_device=profile.device,
        gpu_ids=deployment.gpu_ids,
    )
    deployment.execution = profile.execution
    return deployment


def _resolve_display_device(
    *,
    use_gpu: bool,
    provider_preference: str | None,
    profile_device: str | None,
    gpu_ids: list[int] | None,
) -> str:
    if not use_gpu:
        return "cpu"
    if str(provider_preference or "").strip().lower() == "cuda":
        return "cuda"
    if gpu_ids:
        return "cuda"
    if str(profile_device or "").strip().lower().startswith("cuda"):
        return "cuda"
    return "cuda"


def _resolve_provider_preference(
    *,
    requested_preference: str | None,
    use_gpu: bool,
    settings,
) -> str:
    if not use_gpu:
        return "cpu"
    normalized = str(requested_preference or "").strip().lower()
    if normalized == "cuda":
        return normalized
    capabilities = inspect_runtime_capabilities(settings)
    if _runtime_provider_usable(capabilities, "CUDAExecutionProvider"):
        return "cuda"
    return "cpu"


def _validate_runtime_backend_choice(
    *,
    settings,
    use_gpu: bool,
    provider_preference: str | None,
) -> None:
    normalized = str(provider_preference or "").strip().lower() or ("cpu" if not use_gpu else "")
    capabilities = inspect_runtime_capabilities(settings)
    if not use_gpu:
        return
    if normalized == "cuda" and not _runtime_provider_usable(capabilities, "CUDAExecutionProvider"):
        raise HTTPException(status_code=400, detail="当前服务器环境不支持 CUDA Execution Provider，平台已禁止创建该部署选项。")
    if normalized == "cpu":
        raise HTTPException(status_code=400, detail="当前服务器环境没有可用的 GPU Execution Provider，请改为 CPU 部署。")


def _runtime_provider_usable(capabilities: dict[str, object], provider_name: str) -> bool:
    for row in capabilities.get("providers") or []:
        if str(row.get("provider")) == provider_name:
            return bool(row.get("runtime_usable"))
    return False


def _resolve_runtime_device(
    *,
    model,
    requested_use_gpu: bool,
    requested_gpu_ids: list[int] | None,
    profile_path: str | None = None,
) -> tuple[bool, list[int], str | None]:
    cpu_force_note = _resolve_cpu_preferred_runtime_note(model=model, profile_path=profile_path)
    if cpu_force_note:
        return (False, [], cpu_force_note)

    if not requested_use_gpu:
        return (False, [], None)

    gpu_status = get_gpu_status()
    all_devices = list(gpu_status.get("devices") or [])
    devices = [item for item in all_devices if item.get("platform_enabled", True)]
    disabled_gpu_ids = sorted(
        {
            int(item.get("index"))
            for item in all_devices
            if item.get("index") is not None and not item.get("platform_enabled", True)
        }
    )
    if not gpu_status.get("available") or not devices:
        return (False, [], "gpu_unavailable_or_disabled_auto_fallback_cpu")

    required_memory_mb = _estimate_required_gpu_memory_mb(
        package_root=getattr(getattr(model, "summary", None), "package_root", None),
        stage_count=int(getattr(getattr(model, "summary", None), "stage_count", 1) or 1),
    )
    preferred_gpu_ids = [int(item) for item in (requested_gpu_ids or [])]
    disabled_requested_gpu_ids = [item for item in preferred_gpu_ids if item in disabled_gpu_ids]
    preferred_devices = [item for item in devices if int(item.get("index")) in preferred_gpu_ids] if preferred_gpu_ids else devices
    suitable = [item for item in preferred_devices if _gpu_has_enough_memory(item, required_memory_mb)]
    if suitable:
        best = sorted(suitable, key=_gpu_free_memory_mb, reverse=True)[0]
        if disabled_requested_gpu_ids:
            return (
                True,
                [int(best.get("index"))],
                "gpu_disabled_auto_reassign_to_"
                f"{int(best.get('index'))}_disabled_{','.join(str(item) for item in disabled_requested_gpu_ids)}",
            )
        return (True, [int(best.get("index"))], None)

    global_suitable = [item for item in devices if _gpu_has_enough_memory(item, required_memory_mb)]
    if global_suitable:
        best = sorted(global_suitable, key=_gpu_free_memory_mb, reverse=True)[0]
        return (
            True,
            [int(best.get("index"))],
            (
                "gpu_disabled_auto_reassign_to_"
                f"{int(best.get('index'))}_disabled_{','.join(str(item) for item in disabled_requested_gpu_ids)}"
                if disabled_requested_gpu_ids
                else f"gpu_auto_reassign_to_{int(best.get('index'))}_required_{int(required_memory_mb)}mb"
            ),
        )

    return (False, [], f"gpu_insufficient_auto_fallback_cpu_required_{int(required_memory_mb)}mb")


def _resolve_cpu_preferred_runtime_note(*, model, profile_path: str | None) -> str | None:
    optimization_gpu_restriction_note = _resolve_optimization_gpu_restriction_note(model=model)
    if optimization_gpu_restriction_note:
        return optimization_gpu_restriction_note

    try:
        if profile_path and Path(profile_path).exists():
            profile = load_profile(Path(profile_path))
            precision = str(profile.execution.get("precision", "") or "").strip().lower()
            if "int8" in precision:
                return "int8_profile_force_cpu"
    except Exception:  # noqa: BLE001
        pass

    model_hints = " ".join(
        [
            str(getattr(getattr(model, "summary", None), "model_id", "") or ""),
            str(getattr(getattr(model, "summary", None), "model_name", "") or ""),
        ]
    ).strip().lower()
    if "int8" in model_hints:
        return "int8_model_force_cpu"

    return None


def _resolve_optimization_gpu_restriction_note(*, model) -> str | None:
    package_root = str(getattr(getattr(model, "summary", None), "package_root", "") or "").strip()
    if not package_root:
        return None
    metadata = read_package_metadata(package_root)
    optimization_type = str(metadata.get("optimization_type", "") or "").strip().lower()
    if optimization_type == "ort_format":
        return "optimization_gpu_restricted_force_cpu:ort_format"
    if optimization_type == "int8_dynamic":
        return "optimization_gpu_restricted_force_cpu:int8_dynamic"
    return None


def _estimate_required_gpu_memory_mb(*, package_root: str | None, stage_count: int) -> float:
    package_size_mb = _directory_size_mb(package_root)
    base = max(1536.0, (package_size_mb or 512.0) * 1.8)
    stage_overhead = max(256.0, float(stage_count) * 320.0)
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


def _gpu_free_memory_mb(device: dict[str, object]) -> float:
    total = float(device.get("memory_total_mb") or 0.0)
    used = float(device.get("memory_used_mb") or 0.0)
    return max(total - used, 0.0)


def _gpu_has_enough_memory(device: dict[str, object], required_memory_mb: float) -> bool:
    return _gpu_free_memory_mb(device) >= required_memory_mb


def _adapt_deployment_request_with_policy(
    *,
    request: DeploymentCreateRequest,
    model,
) -> tuple[DeploymentCreateRequest, str | None]:
    next_request = request.model_copy(deep=True)
    adaptation_reason = _resolve_policy_adaptation_reason(request=request, model=model)
    if adaptation_reason is None:
        return next_request, None

    recommendation = _get_policy_recommendation_for_model(request.model_id)
    if recommendation is None:
        return next_request, None

    adaptation_messages: list[str] = []
    if recommendation.recommended_profile_name in model.resolved_profiles and (
        recommendation.recommended_profile_name != next_request.profile_name
    ):
        next_request.profile_name = recommendation.recommended_profile_name
        adaptation_messages.append(f"profile->{recommendation.recommended_profile_name}")

    target_use_gpu = bool(recommendation.recommended_use_gpu)
    target_gpu_ids = [int(item) for item in recommendation.recommended_gpu_ids] if target_use_gpu else []
    if target_use_gpu != bool(next_request.use_gpu):
        next_request.use_gpu = target_use_gpu
        adaptation_messages.append(f"device->{'gpu' if target_use_gpu else 'cpu'}")
    if list(next_request.gpu_ids or []) != target_gpu_ids:
        next_request.gpu_ids = target_gpu_ids
        adaptation_messages.append(
            f"gpu_ids->{','.join(str(item) for item in target_gpu_ids) if target_gpu_ids else 'none'}"
        )

    target_provider = "cuda" if target_use_gpu else "cpu"
    if str(next_request.provider_preference or "").strip().lower() != target_provider:
        next_request.provider_preference = target_provider
        adaptation_messages.append(f"provider->{target_provider}")

    if not adaptation_messages:
        return next_request, None

    return (
        next_request,
        f"{adaptation_reason}:" + ",".join(adaptation_messages),
    )


def _is_model_library_quick_deploy(request: DeploymentCreateRequest) -> bool:
    notes = str(request.notes or "").strip()
    return "从模型页快速创建" in notes or "快速部署" in notes.lower()


def _resolve_policy_adaptation_reason(
    *,
    request: DeploymentCreateRequest,
    model,
) -> str | None:
    if _is_model_library_quick_deploy(request):
        return "deployment_policy_quick_deploy_auto_adapted"
    if _has_invalid_requested_gpu_ids(request=request, model=model):
        return "deployment_policy_invalid_gpu_request_auto_adapted"
    return None


def _has_invalid_requested_gpu_ids(
    *,
    request: DeploymentCreateRequest,
    model,
) -> bool:
    if not bool(request.use_gpu):
        return False
    requested_gpu_ids = [int(item) for item in list(request.gpu_ids or [])]
    if not requested_gpu_ids:
        return False

    gpu_status = get_gpu_status()
    devices = [item for item in list(gpu_status.get("devices") or []) if item.get("platform_enabled", True)]
    enabled_gpu_ids = {int(item.get("index")) for item in devices}
    if any(item not in enabled_gpu_ids for item in requested_gpu_ids):
        return True

    required_memory_mb = _estimate_required_gpu_memory_mb(
        package_root=getattr(getattr(model, "summary", None), "package_root", None),
        stage_count=int(getattr(getattr(model, "summary", None), "stage_count", 1) or 1),
    )
    preferred_devices = [item for item in devices if int(item.get("index")) in requested_gpu_ids]
    if not preferred_devices:
        return True
    return not any(_gpu_has_enough_memory(item, required_memory_mb) for item in preferred_devices)


def _get_policy_recommendation_for_model(model_id: str):
    registry = get_registry()
    feedback_store = get_task_routing_feedback_store()
    policy = build_deployment_policy(
        models=registry.list_models(),
        deployments=get_deployment_store().list(),
        benchmarks=get_benchmark_store().list(),
        prediction_archives=get_prediction_store().list(),
        gpu_snapshot=get_gpu_status(),
        decision_records=feedback_store.list_decisions(),
        feedback_records=feedback_store.list_feedback(),
        automation_actions=[],
    )
    for recommendation in policy.recommendations:
        if recommendation.model_id == model_id:
            return recommendation
    return None


def _schedule_auto_benchmark_after_deployment(
    *,
    background_tasks: BackgroundTasks,
    deployment: DeploymentRecord,
    model_summary,
    profile_path: str | None,
) -> None:
    if deployment.status != "running" or not deployment.service_url:
        return
    if not getattr(model_summary, "predict_ready", False):
        return
    background_tasks.add_task(
        _run_auto_benchmark_after_deployment_async,
        deployment.deployment_id,
        deployment.model_id,
        profile_path,
    )


def _run_auto_benchmark_after_deployment_async(
    deployment_id: str,
    model_id: str,
    profile_path: str | None,
) -> None:
    store = get_deployment_store()
    registry = get_registry()
    benchmark_store = get_benchmark_store()
    try:
        deployment = next(item for item in store.list() if item.deployment_id == deployment_id)
    except StopIteration:
        return
    deployment = _hydrate_deployment_runtime_snapshot(deployment)
    try:
        model_summary = registry.get_model(model_id).summary
    except KeyError:
        return

    updated_deployment = _run_auto_benchmark_after_deployment(
        deployment=deployment,
        model_summary=model_summary,
        profile_path=profile_path,
        benchmark_store=benchmark_store,
    )
    store.update(updated_deployment)


def _run_auto_benchmark_after_deployment(
    *,
    deployment: DeploymentRecord,
    model_summary,
    profile_path: str | None,
    benchmark_store=None,
) -> DeploymentRecord:
    if deployment.status != "running" or not deployment.service_url:
        return deployment
    if not getattr(model_summary, "predict_ready", False):
        return deployment

    benchmark_store = benchmark_store or get_benchmark_store()
    existing_completed = [
        item
        for item in benchmark_store.list()
        if item.deployment_id == deployment.deployment_id
        and item.trigger_source == "deployment_auto"
        and item.status == "completed"
    ]
    if existing_completed:
        return deployment

    if not wait_for_runtime(deployment.service_url, timeout_seconds=20.0, interval_seconds=0.5):
        deployment.notes = _append_deployment_note(
            deployment.notes,
            "deployment_auto_benchmark_skipped_runtime_not_ready",
        )
        return deployment

    scene = infer_benchmark_scene(model=model_summary)
    sample_text = _resolve_benchmark_text_for_deployment(
        deployment=deployment,
        model_dataset_name=getattr(model_summary, "dataset_name", None),
        scene=scene,
        requested_text=None,
    )
    if not _wait_for_runtime_predict_ready(
        deployment.service_url,
        sample_text=sample_text,
        timeout_seconds=36.0,
        interval_seconds=2.0,
    ):
        deployment.notes = _append_deployment_note(
            deployment.notes,
            "deployment_auto_benchmark_skipped_predict_not_ready",
        )
        return deployment

    request = BenchmarkCreateRequest(
        deployment_id=deployment.deployment_id,
        text=sample_text,
        iterations=3,
        warmup_runs=1,
        scene=scene,
        label="部署后自动基准",
        trigger_source="deployment_auto",
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
    try:
        if profile_path and Path(profile_path).exists():
            profile = load_profile(profile_path)
            batch_size = max(1, int(profile.execution.get("batch_size", 1) or 1))
        finished = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                if attempt > 0:
                    wait_for_runtime(deployment.service_url, timeout_seconds=12.0, interval_seconds=0.5)
                    _wait_for_runtime_predict_ready(
                        deployment.service_url,
                        sample_text=sample_text,
                        timeout_seconds=12.0,
                        interval_seconds=2.0,
                    )
                finished = run_benchmark(request, record, batch_size=batch_size)
                benchmark_store.update(finished)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if finished is None:
            raise RuntimeError(str(last_error or "deployment_auto_benchmark_failed"))
        deployment.notes = _append_deployment_note(
            deployment.notes,
            f"deployment_auto_benchmark:{finished.benchmark_id}",
        )
    except Exception as exc:  # noqa: BLE001
        record.status = "failed"
        record.errors = [str(exc)]
        benchmark_store.update(record)
        deployment.notes = _append_deployment_note(
            deployment.notes,
            f"deployment_auto_benchmark_failed:{exc}",
        )
    return deployment


def _append_deployment_note(current_notes: str | None, note: str) -> str:
    normalized_note = str(note or "").strip()
    if not normalized_note:
        return str(current_notes or "").strip()
    current_parts = [str(item).strip() for item in str(current_notes or "").split("|") if str(item).strip()]
    if normalized_note in current_parts:
        return " | ".join(current_parts)
    return " | ".join([*current_parts, normalized_note])


def _wait_for_runtime_predict_ready(
    service_url: str,
    *,
    sample_text: str,
    timeout_seconds: float,
    interval_seconds: float,
) -> bool:
    attempts = max(1, int(timeout_seconds / max(interval_seconds, 0.5)))
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            _post_runtime_predict_probe(service_url, sample_text=sample_text, timeout_seconds=max(10.0, interval_seconds * 3))
            return True
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep(interval_seconds)
    if last_error:
        return False
    return False


def _post_runtime_predict_probe(
    service_url: str,
    *,
    sample_text: str,
    timeout_seconds: float,
) -> dict[str, object]:
    payload = json.dumps(
        {
            "texts": [sample_text],
            "options": {},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        service_url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    return json.loads(body)


def _resolve_benchmark_dataset_name(
    *,
    model_dataset_name: str | None,
    scene: str | None,
) -> str | None:
    if str(model_dataset_name or "").strip():
        return model_dataset_name
    normalized_scene = str(scene or "").strip().lower()
    if normalized_scene == "zh_general":
        return "DuIE2.0"
    if normalized_scene == "zh_industrial":
        return "IndustrialEquipmentMix15k"
    if normalized_scene == "en_general":
        return model_dataset_name or "WebNLG"
    return model_dataset_name


def _resolve_benchmark_text_for_deployment(
    *,
    deployment: DeploymentRecord,
    model_dataset_name: str | None,
    scene: str | None,
    requested_text: str | None,
) -> str:
    normalized_requested_text = str(requested_text or "").strip()
    if normalized_requested_text:
        return normalized_requested_text
    dataset_name = _resolve_benchmark_dataset_name(
        model_dataset_name=model_dataset_name,
        scene=scene,
    )
    fallback_text = default_benchmark_text(scene)
    try:
        if dataset_name:
            sample = get_dataset_random_sample_text(
                dataset_name,
                split_preference="train",
            )
            return str(sample.get("text") or fallback_text)
    except Exception:  # noqa: BLE001
        pass
    return fallback_text
