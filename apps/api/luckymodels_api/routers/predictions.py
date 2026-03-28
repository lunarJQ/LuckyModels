from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import PredictionArchiveRecord

from ..dependencies import get_prediction_store

router = APIRouter(tags=["predictions"])


@router.get("/prediction-archives", response_model=list[PredictionArchiveRecord])
def list_prediction_archives() -> list[PredictionArchiveRecord]:
    return get_prediction_store().list()


@router.get("/prediction-archives/{archive_id}", response_model=PredictionArchiveRecord)
def get_prediction_archive(archive_id: str) -> PredictionArchiveRecord:
    for item in get_prediction_store().list():
        if item.archive_id == archive_id:
            return item
    raise HTTPException(status_code=404, detail=f"Prediction archive not found: {archive_id}")


@router.delete("/prediction-archives/{archive_id}", response_model=PredictionArchiveRecord)
def delete_prediction_archive(archive_id: str) -> PredictionArchiveRecord:
    deleted = get_prediction_store().delete(archive_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail=f"Prediction archive not found: {archive_id}")
    return deleted
