from fastapi import APIRouter, HTTPException

from luckymodels_core.dataset_samples import export_dataset_eval_cases, get_dataset_eval_cases
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


@router.get("/models/{model_id}/sample-cases")
def get_model_sample_cases(model_id: str, limit: int = 20) -> dict[str, object]:
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_name = model.summary.dataset_name
    if not dataset_name:
        raise HTTPException(status_code=400, detail=f"Model has no dataset_name: {model_id}")

    try:
        payload = get_dataset_eval_cases(dataset_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        **payload,
    }


@router.post("/models/{model_id}/prepare-eval")
def prepare_model_eval_cases(model_id: str, limit: int = 128) -> dict[str, object]:
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_name = model.summary.dataset_name
    if not dataset_name:
        raise HTTPException(status_code=400, detail=f"Model has no dataset_name: {model_id}")

    try:
        payload = export_dataset_eval_cases(dataset_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        **payload,
    }
