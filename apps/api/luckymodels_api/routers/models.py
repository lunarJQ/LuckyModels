from fastapi import APIRouter, HTTPException

from luckymodels_core.manifest import load_profile
from luckymodels_core.schemas import ModelDetail, ModelSummary, ProfileConfig

from ..dependencies import get_registry

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelSummary])
def list_models() -> list[ModelSummary]:
    return get_registry().list_models()


@router.get("/models/{model_id}", response_model=ModelDetail)
def get_model(model_id: str) -> ModelDetail:
    try:
        return get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/models/{model_id}/profiles", response_model=dict[str, ProfileConfig])
def get_model_profiles(model_id: str) -> dict[str, ProfileConfig]:
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {name: load_profile(path) for name, path in model.resolved_profiles.items()}
