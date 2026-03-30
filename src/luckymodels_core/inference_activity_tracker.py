from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class _ActiveRequest:
    token: str
    model_id: str
    deployment_id: str | None
    source: str | None
    started_at: datetime


@dataclass
class _ModelActivity:
    active_requests: dict[str, _ActiveRequest] = field(default_factory=dict)
    last_started_at: datetime | None = None
    last_finished_at: datetime | None = None


class InferenceActivityTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._model_activity: dict[str, _ModelActivity] = {}
        self._token_to_model: dict[str, str] = {}

    def begin(
        self,
        *,
        model_id: str,
        deployment_id: str | None = None,
        source: str | None = None,
    ) -> str:
        token = f"act-{uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        with self._lock:
            activity = self._model_activity.setdefault(model_id, _ModelActivity())
            activity.last_started_at = now
            activity.active_requests[token] = _ActiveRequest(
                token=token,
                model_id=model_id,
                deployment_id=deployment_id,
                source=source,
                started_at=now,
            )
            self._token_to_model[token] = model_id
        return token

    def end(self, token: str | None) -> None:
        if not token:
            return
        now = datetime.now(timezone.utc)
        with self._lock:
            model_id = self._token_to_model.pop(token, None)
            if model_id is None:
                return
            activity = self._model_activity.get(model_id)
            if activity is None:
                return
            activity.active_requests.pop(token, None)
            activity.last_finished_at = now

    def get_protection_status(self, model_id: str, *, cooldown_seconds: int) -> dict[str, Any]:
        cooldown_seconds = max(30, int(cooldown_seconds or 120))
        now = datetime.now(timezone.utc)
        with self._lock:
            activity = self._model_activity.get(model_id)
            if activity is None:
                return {
                    "protected": False,
                    "state": "idle",
                    "active_count": 0,
                    "last_finished_at": None,
                    "cooldown_remaining_seconds": 0,
                    "active_deployment_ids": [],
                    "active_sources": [],
                }

            active_requests = list(activity.active_requests.values())
            active_deployment_ids = sorted(
                {
                    item.deployment_id
                    for item in active_requests
                    if item.deployment_id
                }
            )
            active_sources = sorted(
                {
                    item.source
                    for item in active_requests
                    if item.source
                }
            )
            if active_requests:
                return {
                    "protected": True,
                    "state": "active",
                    "active_count": len(active_requests),
                    "last_finished_at": _format_datetime(activity.last_finished_at),
                    "cooldown_remaining_seconds": cooldown_seconds,
                    "active_deployment_ids": active_deployment_ids,
                    "active_sources": active_sources,
                }

            last_finished_at = activity.last_finished_at
            if last_finished_at is None:
                return {
                    "protected": False,
                    "state": "idle",
                    "active_count": 0,
                    "last_finished_at": None,
                    "cooldown_remaining_seconds": 0,
                    "active_deployment_ids": [],
                    "active_sources": [],
                }

            protected_until = last_finished_at + timedelta(seconds=cooldown_seconds)
            remaining = max(0, int((protected_until - now).total_seconds()))
            return {
                "protected": remaining > 0,
                "state": "cooldown" if remaining > 0 else "idle",
                "active_count": 0,
                "last_finished_at": _format_datetime(last_finished_at),
                "cooldown_remaining_seconds": remaining,
                "active_deployment_ids": [],
                "active_sources": [],
            }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                model_id: {
                    "active_count": len(activity.active_requests),
                    "active_deployment_ids": sorted(
                        {
                            request.deployment_id
                            for request in activity.active_requests.values()
                            if request.deployment_id
                        }
                    ),
                    "active_sources": sorted(
                        {
                            request.source
                            for request in activity.active_requests.values()
                            if request.source
                        }
                    ),
                    "last_started_at": _format_datetime(activity.last_started_at),
                    "last_finished_at": _format_datetime(activity.last_finished_at),
                }
                for model_id, activity in self._model_activity.items()
            }


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
