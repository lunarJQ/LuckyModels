from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import (
    PredictionArchiveFeedbackUpdateRequest,
    PredictionArchiveRecord,
    TaskRoutingFeedbackSubmitRequest,
)

from ..dependencies import get_prediction_store, get_task_routing_feedback_store
from ..permissions import require_admin, require_permission

router = APIRouter(tags=["predictions"])


@router.get("/prediction-archives", response_model=list[PredictionArchiveRecord])
def list_prediction_archives() -> list[PredictionArchiveRecord]:
    require_permission("predictions.view")
    return get_prediction_store().list()


@router.get("/prediction-archives/{archive_id}", response_model=PredictionArchiveRecord)
def get_prediction_archive(archive_id: str) -> PredictionArchiveRecord:
    require_permission("predictions.view")
    item = get_prediction_store().get(archive_id)
    if item is not None:
        return item
    raise HTTPException(status_code=404, detail=f"Prediction archive not found: {archive_id}")


@router.post("/prediction-archives/{archive_id}/feedback", response_model=PredictionArchiveRecord)
def update_prediction_archive_feedback(
    archive_id: str,
    request: PredictionArchiveFeedbackUpdateRequest,
) -> PredictionArchiveRecord:
    require_permission("predictions.feedback")
    feedback_status = str(request.feedback_status or "pending").strip().lower()
    if feedback_status not in {"pending", "satisfied", "unsatisfied"}:
        raise HTTPException(status_code=400, detail=f"Unsupported feedback_status: {request.feedback_status}")
    updated = get_prediction_store().update_feedback(
        archive_id,
        feedback_status=feedback_status,
        feedback_note=request.feedback_note,
    )
    if updated is not None:
        if updated.task_routing_decision_id:
            feedback_store = get_task_routing_feedback_store()
            if feedback_status == "pending":
                feedback_store.delete_feedback(updated.task_routing_decision_id)
            else:
                adopted_model_id = (
                    updated.task_routing_selected_model_ids[0]
                    if len(updated.task_routing_selected_model_ids or []) == 1
                    else None
                )
                try:
                    feedback_store.save_feedback(
                        TaskRoutingFeedbackSubmitRequest(
                            decision_id=updated.task_routing_decision_id,
                            accepted=feedback_status == "satisfied",
                            feedback_type="accepted" if feedback_status == "satisfied" else "rejected",
                            adopted_model_id=adopted_model_id,
                            note=request.feedback_note or "来自结果归档页的任务调度反馈。",
                        )
                    )
                except KeyError:
                    pass
        return updated
    raise HTTPException(status_code=404, detail=f"Prediction archive not found: {archive_id}")


@router.delete("/prediction-archives/{archive_id}", response_model=PredictionArchiveRecord)
def delete_prediction_archive(archive_id: str) -> PredictionArchiveRecord:
    require_admin()
    deleted = get_prediction_store().delete(archive_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail=f"Prediction archive not found: {archive_id}")
    return deleted
