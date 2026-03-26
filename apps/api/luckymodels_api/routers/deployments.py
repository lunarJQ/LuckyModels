from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import DeploymentCreateRequest, DeploymentRecord

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

    if request.profile_name not in model.resolved_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Profile not found for model {request.model_id}: {request.profile_name}",
        )

    return get_deployment_store().create(request)
