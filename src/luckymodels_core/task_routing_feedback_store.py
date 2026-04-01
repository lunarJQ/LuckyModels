from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .schemas import (
    TaskRoutingDecisionRecord,
    TaskRoutingFeedbackRecord,
    TaskRoutingFeedbackSubmitRequest,
    utc_now,
)


class TaskRoutingFeedbackStore:
    def __init__(self, decisions_file: Path, feedback_file: Path) -> None:
        self.decisions_file = decisions_file
        self.feedback_file = feedback_file
        self.decisions_file.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)

    def list_decisions(self) -> list[TaskRoutingDecisionRecord]:
        return self._read_records(self.decisions_file, TaskRoutingDecisionRecord)

    def list_feedback(self) -> list[TaskRoutingFeedbackRecord]:
        return self._read_records(self.feedback_file, TaskRoutingFeedbackRecord)

    def get_decision(self, decision_id: str) -> TaskRoutingDecisionRecord | None:
        for item in self.list_decisions():
            if item.decision_id == decision_id:
                return item
        return None

    def create_decision(self, record: TaskRoutingDecisionRecord) -> TaskRoutingDecisionRecord:
        decisions = self.list_decisions()
        decisions.append(record)
        self._write_records(self.decisions_file, decisions)
        return record

    def save_feedback(self, request: TaskRoutingFeedbackSubmitRequest) -> TaskRoutingFeedbackRecord:
        decision = self.get_decision(request.decision_id)
        if decision is None:
            raise KeyError(f"Task routing decision not found: {request.decision_id}")

        feedback_items = self.list_feedback()
        existing = next((item for item in feedback_items if item.decision_id == request.decision_id), None)
        selected_model_ids = [item.model_id for item in decision.execution_results] or [
            item.model_id for item in decision.selected_deployments
        ]
        if existing is None:
            feedback = TaskRoutingFeedbackRecord(
                feedback_id=f"trf-{uuid4().hex[:12]}",
                decision_id=request.decision_id,
                accepted=request.accepted,
                feedback_type=request.feedback_type,
                adopted_model_id=request.adopted_model_id,
                objective=decision.objective,
                inferred_language=decision.inferred_language,
                context_domain=str(decision.context_features.get("domain") or "general"),
                execution_path=decision.execution_path,
                selected_model_ids=selected_model_ids,
                note=request.note,
            )
            feedback_items.append(feedback)
        else:
            feedback = existing.model_copy(
                update={
                    "accepted": request.accepted,
                    "feedback_type": request.feedback_type,
                    "adopted_model_id": request.adopted_model_id,
                    "note": request.note,
                    "updated_at": utc_now(),
                    "selected_model_ids": selected_model_ids,
                }
            )
            feedback_items = [feedback if item.decision_id == request.decision_id else item for item in feedback_items]

        self._write_records(self.feedback_file, feedback_items)
        return feedback

    def delete_feedback(self, decision_id: str) -> bool:
        feedback_items = self.list_feedback()
        remaining_feedback = [item for item in feedback_items if item.decision_id != decision_id]
        if len(remaining_feedback) == len(feedback_items):
            return False
        self._write_records(self.feedback_file, remaining_feedback)
        return True

    def delete_by_model(self, model_id: str) -> tuple[int, int]:
        decisions = self.list_decisions()
        removed_decision_ids = {
            item.decision_id
            for item in decisions
            if any(candidate.model_id == model_id for candidate in item.selected_deployments)
            or any(result.model_id == model_id for result in item.execution_results)
        }
        remaining_decisions = [item for item in decisions if item.decision_id not in removed_decision_ids]
        self._write_records(self.decisions_file, remaining_decisions)

        feedback_items = self.list_feedback()
        remaining_feedback = [
            item
            for item in feedback_items
            if item.decision_id not in removed_decision_ids and item.adopted_model_id != model_id
        ]
        self._write_records(self.feedback_file, remaining_feedback)
        return (len(removed_decision_ids), len(feedback_items) - len(remaining_feedback))

    def _read_records(self, path: Path, schema):
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or []
        return [schema.model_validate(item) for item in data]

    def _write_records(self, path: Path, records: list) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in records], handle, ensure_ascii=False, indent=2)
