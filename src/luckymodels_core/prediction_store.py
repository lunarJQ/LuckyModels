from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .schemas import PredictionArchiveRecord


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
        service_url: str | None,
        scene: str | None,
        request_text: str | None,
        request_texts: list[str],
        options: dict,
        response_payload: dict,
    ) -> PredictionArchiveRecord:
        archive = PredictionArchiveRecord(
            archive_id=f"pred-{uuid4().hex[:12]}",
            deployment_id=deployment_id,
            model_id=model_id,
            profile_name=profile_name,
            service_url=service_url,
            scene=scene,
            request_text=request_text,
            request_texts=request_texts,
            options=options,
            latency_ms=_safe_float(response_payload.get("latency_ms")),
            triple_count=_safe_int(response_payload.get("triple_count")),
            item_count=_safe_int(response_payload.get("item_count")),
            response_payload=response_payload,
        )
        archives = self.list()
        archives.append(archive)
        self._write(archives)
        return archive

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
