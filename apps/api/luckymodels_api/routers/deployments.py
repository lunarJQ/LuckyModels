import json
from pathlib import Path
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from luckymodels_core.launcher import (
    launch_runtime_process,
    process_is_running,
    stop_runtime_process,
    tail_log,
    wait_for_runtime,
)
from luckymodels_core.manifest import load_profile
from luckymodels_core.schemas import DeploymentCreateRequest, DeploymentRecord
from luckymodels_core.settings import get_settings

from ..dependencies import (
    get_deployment_store,
    get_inference_activity_tracker,
    get_prediction_store,
    get_registry,
)

router = APIRouter(tags=["deployments"])


class DeploymentPredictRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None
    scene: str | None = None
    options: dict[str, object] | None = None


@router.get("/deployments", response_model=list[DeploymentRecord])
def list_deployments() -> list[DeploymentRecord]:
    return [_hydrate_deployment_runtime_snapshot(item) for item in get_deployment_store().list()]


@router.post("/deployments", response_model=DeploymentRecord)
def create_deployment(request: DeploymentCreateRequest) -> DeploymentRecord:
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

    if request.profile_name not in model.resolved_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Profile not found for model {request.model_id}: {request.profile_name}",
        )

    store = get_deployment_store()
    deployment = store.create(request)
    deployment.manifest_path = model.summary.package_root + "/manifest.yaml"
    deployment.profile_path = model.resolved_profiles[request.profile_name]
    deployment.service_url = deployment.service_url or f"http://127.0.0.1:{store._allocate_port()}"
    settings = get_settings()
    deployment.log_path = str(settings.logs_root / "deployments" / f"{deployment.deployment_id}.log")
    deployment = _hydrate_deployment_runtime_snapshot(deployment)

    if request.launch_runtime:
        port = int(deployment.service_url.rsplit(":", maxsplit=1)[-1])
        pid, service_url = launch_runtime_process(
            root_dir=settings.root_dir,
            manifest_path=deployment.manifest_path,
            profile_path=deployment.profile_path,
            port=port,
            log_path=deployment.log_path,
            gpu_ids=deployment.gpu_ids,
            use_gpu=deployment.use_gpu,
        )
        deployment.pid = pid
        deployment.service_url = service_url
        deployment.status = "running" if wait_for_runtime(service_url) else "starting"
    else:
        deployment.status = "planned"

    return store.update(deployment)


@router.get("/deployments/{deployment_id}/health")
def check_deployment_health(deployment_id: str) -> dict[str, object]:
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
    store = get_deployment_store()
    deployment = _get_deployment_or_404(deployment_id)

    if deployment.pid:
        stop_runtime_process(deployment.pid)

    if not deployment.manifest_path or not Path(deployment.manifest_path).exists():
        raise HTTPException(status_code=400, detail="Deployment manifest_path is missing or invalid")
    if not deployment.profile_path or not Path(deployment.profile_path).exists():
        raise HTTPException(status_code=400, detail="Deployment profile_path is missing or invalid")

    service_url = deployment.service_url or f"http://127.0.0.1:{store._allocate_port()}"
    port = int(service_url.rsplit(":", maxsplit=1)[-1])
    settings = get_settings()
    pid, service_url = launch_runtime_process(
        root_dir=settings.root_dir,
        manifest_path=deployment.manifest_path,
        profile_path=deployment.profile_path,
        port=port,
        log_path=deployment.log_path,
        gpu_ids=deployment.gpu_ids,
        use_gpu=deployment.use_gpu,
    )
    deployment = _hydrate_deployment_runtime_snapshot(deployment)
    deployment.pid = pid
    deployment.service_url = service_url
    deployment.status = "running" if wait_for_runtime(service_url) else "starting"
    return store.update(deployment)


@router.delete("/deployments/{deployment_id}", response_model=DeploymentRecord)
def delete_deployment(deployment_id: str) -> DeploymentRecord:
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
    deployment.device = "cpu" if not deployment.use_gpu else profile.device
    if not deployment.gpu_ids and profile.gpu_ids:
        deployment.gpu_ids = list(profile.gpu_ids)
    deployment.execution = profile.execution
    return deployment
