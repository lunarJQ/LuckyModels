from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .benchmark_evidence import (
    build_benchmark_scope_key,
    build_deployment_signature,
    infer_benchmark_scene,
    select_latest_effective_benchmarks,
)
from .deployment_revision import effective_revision_started_at, ensure_runtime_signature, initialize_deployment_revision_metadata
from .schemas import BenchmarkCreateRequest, BenchmarkRecord, DeploymentRecord, ModelSummary


class BenchmarkStore:
    def __init__(self, benchmarks_file: Path) -> None:
        self.benchmarks_file = benchmarks_file
        self.benchmarks_file.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[BenchmarkRecord]:
        if not self.benchmarks_file.exists():
            return []

        with self.benchmarks_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or []
        return [BenchmarkRecord.model_validate(item) for item in data]

    def create(
        self,
        request: BenchmarkCreateRequest,
        *,
        model_id: str,
        profile_name: str,
        service_url: str,
        deployment: DeploymentRecord | None = None,
        model: ModelSummary | None = None,
    ) -> BenchmarkRecord:
        if deployment is not None:
            deployment = initialize_deployment_revision_metadata(deployment)
        scene = infer_benchmark_scene(
            model=model,
            explicit_scene=request.scene,
            request_text=request.text,
        )
        record = BenchmarkRecord(
            benchmark_id=f"bm-{uuid4().hex[:12]}",
            deployment_id=request.deployment_id,
            model_id=model_id,
            profile_name=profile_name,
            service_url=service_url,
            label=request.label,
            scene=scene,
            trigger_source=request.trigger_source,
            benchmark_scope_key=build_benchmark_scope_key(request.deployment_id, scene),
            deployment_signature=build_deployment_signature(deployment) if deployment is not None else None,
            deployment_revision=int(deployment.runtime_revision) if deployment is not None else None,
            revision_started_at=effective_revision_started_at(deployment) if deployment is not None else None,
            runtime_signature=ensure_runtime_signature(deployment) if deployment is not None else None,
            is_effective=False,
            superseded_by=None,
            request_text=request.text,
            iterations=request.iterations,
            warmup_runs=request.warmup_runs,
            timeout_seconds=request.timeout_seconds,
        )
        benchmarks = self.list()
        benchmarks.append(record)
        self._write(benchmarks)
        return record

    def update(self, record: BenchmarkRecord) -> BenchmarkRecord:
        benchmarks = self.list()
        updated: list[BenchmarkRecord] = []
        found = False
        record = self._normalize_record(record)
        for item in benchmarks:
            if item.benchmark_id == record.benchmark_id:
                updated.append(record)
                found = True
            else:
                updated.append(item)

        if not found:
            updated.append(record)

        updated = self._refresh_effective_scope(updated, record.benchmark_scope_key)
        self._write(updated)
        return record

    def delete(self, benchmark_id: str) -> BenchmarkRecord | None:
        benchmarks = self.list()
        remaining: list[BenchmarkRecord] = []
        deleted: BenchmarkRecord | None = None
        for item in benchmarks:
            if item.benchmark_id == benchmark_id:
                deleted = item
            else:
                remaining.append(item)

        if deleted is None:
            return None

        remaining = self._refresh_effective_scope(remaining, deleted.benchmark_scope_key)
        self._write(remaining)
        return deleted

    def _write(self, benchmarks: list[BenchmarkRecord]) -> None:
        with self.benchmarks_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in benchmarks], handle, ensure_ascii=False, indent=2)

    def migrate_legacy_records(
        self,
        *,
        model_lookup: dict[str, ModelSummary] | None = None,
        deployment_lookup: dict[str, DeploymentRecord] | None = None,
    ) -> dict[str, int]:
        model_lookup = dict(model_lookup or {})
        deployment_lookup = dict(deployment_lookup or {})
        benchmarks = self.list()
        if not benchmarks:
            return {
                "total_records": 0,
                "updated_records": 0,
                "scene_backfilled": 0,
                "signature_backfilled": 0,
                "scope_backfilled": 0,
                "effective_refreshed": 0,
            }

        updated_records = 0
        scene_backfilled = 0
        signature_backfilled = 0
        scope_backfilled = 0
        effective_refreshed = 0

        normalized: list[BenchmarkRecord] = []
        for item in benchmarks:
            before = item.model_dump()
            if not str(item.scene or "").strip():
                model = model_lookup.get(item.model_id)
                item.scene = infer_benchmark_scene(
                    model=model,
                    explicit_scene=None,
                    language=str((item.sample_response or {}).get("language") or ""),
                    request_text=item.request_text,
                )
                scene_backfilled += 1
            if not str(item.trigger_source or "").strip():
                item.trigger_source = "manual"
            if not str(item.benchmark_scope_key or "").strip():
                item.benchmark_scope_key = build_benchmark_scope_key(item.deployment_id, item.scene)
                scope_backfilled += 1
            if not str(item.deployment_signature or "").strip():
                deployment = deployment_lookup.get(item.deployment_id)
                if deployment is not None:
                    item.deployment_signature = build_deployment_signature(deployment)
                    signature_backfilled += 1
            normalized_item = self._normalize_record(item)
            if normalized_item.model_dump() != before:
                updated_records += 1
            normalized.append(normalized_item)

        scope_keys = {
            item.benchmark_scope_key or build_benchmark_scope_key(item.deployment_id, item.scene)
            for item in normalized
        }
        refreshed = normalized
        for scope_key in sorted(scope_keys):
            before_snapshot = [item.model_dump() for item in refreshed if (item.benchmark_scope_key or build_benchmark_scope_key(item.deployment_id, item.scene)) == scope_key]
            refreshed = self._refresh_effective_scope(refreshed, scope_key)
            after_snapshot = [item.model_dump() for item in refreshed if (item.benchmark_scope_key or build_benchmark_scope_key(item.deployment_id, item.scene)) == scope_key]
            if after_snapshot != before_snapshot:
                effective_refreshed += 1

        self._write(refreshed)
        return {
            "total_records": len(refreshed),
            "updated_records": updated_records,
            "scene_backfilled": scene_backfilled,
            "signature_backfilled": signature_backfilled,
            "scope_backfilled": scope_backfilled,
            "effective_refreshed": effective_refreshed,
        }

    def _normalize_record(self, record: BenchmarkRecord) -> BenchmarkRecord:
        if not record.benchmark_scope_key:
            record.benchmark_scope_key = build_benchmark_scope_key(record.deployment_id, record.scene)
        if record.status != "completed":
            record.is_effective = False
            record.superseded_by = None
        else:
            record.is_effective = True
            record.superseded_by = None
        return record

    def _refresh_effective_scope(
        self,
        benchmarks: list[BenchmarkRecord],
        scope_key: str | None,
    ) -> list[BenchmarkRecord]:
        if not scope_key:
            return benchmarks

        latest_effective_by_id = {
            item.benchmark_id
            for item in select_latest_effective_benchmarks(
                [item for item in benchmarks if (item.benchmark_scope_key or build_benchmark_scope_key(item.deployment_id, item.scene)) == scope_key],
                allowed_statuses={"completed"},
            )
        }
        latest_effective_id = next(iter(latest_effective_by_id), None)
        refreshed: list[BenchmarkRecord] = []
        for item in benchmarks:
            item_scope = item.benchmark_scope_key or build_benchmark_scope_key(item.deployment_id, item.scene)
            if item_scope != scope_key:
                refreshed.append(item)
                continue
            if item.status == "completed" and latest_effective_id is not None and item.benchmark_id == latest_effective_id:
                item.is_effective = True
                item.superseded_by = None
            elif item.status == "completed" and latest_effective_id is not None:
                item.is_effective = False
                item.superseded_by = latest_effective_id
            else:
                item.is_effective = False
                item.superseded_by = None
            refreshed.append(item)
        return refreshed
