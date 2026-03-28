from pathlib import Path
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException

from luckymodels_core.launcher import (
    launch_runtime_process,
    process_is_running,
    stop_runtime_process,
    tail_log,
    wait_for_runtime,
)
from luckymodels_core.schemas import DeploymentCreateRequest, DeploymentRecord
from luckymodels_core.settings import get_settings

from ..dependencies import get_deployment_store, get_registry

router = APIRouter(tags=["deployments"])


@router.get("/deployments", response_model=list[DeploymentRecord])
def list_deployments() -> list[DeploymentRecord]:
    return get_deployment_store().list()


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

    if request.launch_runtime:
        port = int(deployment.service_url.rsplit(":", maxsplit=1)[-1])
        pid, service_url = launch_runtime_process(
            root_dir=settings.root_dir,
            manifest_path=deployment.manifest_path,
            profile_path=deployment.profile_path,
            port=port,
            log_path=deployment.log_path,
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
            payload = response.read().decode("utf-8")
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
    )
    deployment.pid = pid
    deployment.service_url = service_url
    deployment.status = "running" if wait_for_runtime(service_url) else "starting"
    return store.update(deployment)


def _get_deployment_or_404(deployment_id: str) -> DeploymentRecord:
    store = get_deployment_store()
    for deployment in store.list():
        if deployment.deployment_id == deployment_id:
            return deployment
    raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")
