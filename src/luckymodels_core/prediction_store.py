from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .deployment_revision import effective_revision_started_at, ensure_runtime_signature, initialize_deployment_revision_metadata
from .schemas import DeploymentRecord, PredictionArchiveRecord, utc_now


class PredictionStore:
    def __init__(self, prediction_archives_file: Path) -> None:
        self.prediction_archives_file = prediction_archives_file
        self.prediction_archives_file.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[PredictionArchiveRecord]:
        if not self.prediction_archives_file.exists():
            return []

        with self.prediction_archives_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or []
        return [PredictionArchiveRecord.model_validate(item) for item in data]

    def create(
        self,
        *,
        deployment_id: str,
        model_id: str,
        profile_name: str,
        deployment: DeploymentRecord | None = None,
        archive_source: str = "deployment_predict",
        service_url: str | None,
        scene: str | None,
        request_text: str | None,
        request_texts: list[str],
        options: dict,
        response_payload: dict,
        task_routing_decision_id: str | None = None,
        task_routing_objective: str | None = None,
        task_routing_execution_path: str | None = None,
        task_routing_fusion_policy: str | None = None,
        task_routing_weight_snapshot: dict | None = None,
        task_routing_adjustment_snapshot: dict | None = None,
        task_routing_unsupervised_adjustment_snapshot: dict | None = None,
        task_routing_feedback_adjustment_snapshot: dict | None = None,
        task_routing_selected_model_ids: list[str] | None = None,
    ) -> PredictionArchiveRecord:
        if deployment is not None:
            deployment = initialize_deployment_revision_metadata(deployment)
        archive = PredictionArchiveRecord(
            archive_id=f"pred-{uuid4().hex[:12]}",
            deployment_id=deployment_id,
            model_id=model_id,
            profile_name=profile_name,
            archive_source=archive_source,
            service_url=service_url,
            scene=scene,
            request_text=request_text,
            request_texts=request_texts,
            options=options,
            latency_ms=_safe_float(response_payload.get("latency_ms")),
            triple_count=_safe_int(response_payload.get("triple_count")),
            item_count=_safe_int(response_payload.get("item_count")),
            response_payload=response_payload,
            task_routing_decision_id=task_routing_decision_id,
            task_routing_objective=task_routing_objective,
            task_routing_execution_path=task_routing_execution_path,
            task_routing_fusion_policy=task_routing_fusion_policy,
            task_routing_weight_snapshot=dict(task_routing_weight_snapshot or {}),
            task_routing_adjustment_snapshot=dict(task_routing_adjustment_snapshot or {}),
            task_routing_unsupervised_adjustment_snapshot=dict(task_routing_unsupervised_adjustment_snapshot or {}),
            task_routing_feedback_adjustment_snapshot=dict(task_routing_feedback_adjustment_snapshot or {}),
            task_routing_selected_model_ids=list(task_routing_selected_model_ids or []),
            deployment_revision=int(deployment.runtime_revision) if deployment is not None else None,
            revision_started_at=effective_revision_started_at(deployment) if deployment is not None else None,
            runtime_signature=ensure_runtime_signature(deployment) if deployment is not None else None,
        )
        archives = self.list()
        archives.append(archive)
        self._write(archives)
        return archive

    def get(self, archive_id: str) -> PredictionArchiveRecord | None:
        for item in self.list():
            if item.archive_id == archive_id:
                return item
        return None

    def update_feedback(
        self,
        archive_id: str,
        *,
        feedback_status: str,
        feedback_note: str | None = None,
    ) -> PredictionArchiveRecord | None:
        archives = self.list()
        updated: PredictionArchiveRecord | None = None
        next_archives: list[PredictionArchiveRecord] = []
        for item in archives:
            if item.archive_id != archive_id:
                next_archives.append(item)
                continue
            updated = item.model_copy(
                update={
                    "feedback_status": feedback_status,
                    "feedback_note": feedback_note,
                    "feedback_updated_at": utc_now(),
                }
            )
            next_archives.append(updated)

        if updated is None:
            return None

        self._write(next_archives)
        return updated

    def delete(self, archive_id: str) -> PredictionArchiveRecord | None:
        archives = self.list()
        remaining: list[PredictionArchiveRecord] = []
        deleted: PredictionArchiveRecord | None = None
        for item in archives:
            if item.archive_id == archive_id:
                deleted = item
            else:
                remaining.append(item)

        if deleted is None:
            return None

        self._write(remaining)
        return deleted

    def _write(self, archives: list[PredictionArchiveRecord]) -> None:
        with self.prediction_archives_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in archives], handle, ensure_ascii=False, indent=2)


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
