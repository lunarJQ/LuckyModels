from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .schemas import BenchmarkCreateRequest, BenchmarkRecord


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
    ) -> BenchmarkRecord:
        record = BenchmarkRecord(
            benchmark_id=f"bm-{uuid4().hex[:12]}",
            deployment_id=request.deployment_id,
            model_id=model_id,
            profile_name=profile_name,
            service_url=service_url,
            label=request.label,
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
        for item in benchmarks:
            if item.benchmark_id == record.benchmark_id:
                updated.append(record)
                found = True
            else:
                updated.append(item)

        if not found:
            updated.append(record)

        self._write(updated)
        return record

    def _write(self, benchmarks: list[BenchmarkRecord]) -> None:
        with self.benchmarks_file.open("w", encoding="utf-8") as handle:
            json.dump([item.model_dump() for item in benchmarks], handle, ensure_ascii=False, indent=2)
