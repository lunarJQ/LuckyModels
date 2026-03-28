from __future__ import annotations

import json
import math
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen

from .schemas import BenchmarkCreateRequest, BenchmarkRecord


def run_benchmark(
    request: BenchmarkCreateRequest,
    record: BenchmarkRecord,
    *,
    batch_size: int = 1,
) -> BenchmarkRecord:
    latencies_ms: list[float] = []
    remote_latencies_ms: list[float] = []
    errors: list[str] = []
    last_response: dict[str, Any] | None = None

    effective_batch_size = max(1, int(batch_size))
    batch_texts = [request.text] * effective_batch_size

    for _ in range(max(0, request.warmup_runs)):
        try:
            _post_predict(
                record.service_url,
                texts=batch_texts,
                options=request.options,
                timeout_seconds=request.timeout_seconds,
            )
        except Exception:
            pass

    bench_started = perf_counter()
    for _ in range(max(0, request.iterations)):
        started = perf_counter()
        try:
            response = _post_predict(
                record.service_url,
                texts=batch_texts,
                options=request.options,
                timeout_seconds=request.timeout_seconds,
            )
            last_response = response
            latency_ms = (perf_counter() - started) * 1000
            latencies_ms.append(round(latency_ms, 3))
            remote_latency = response.get("latency_ms")
            if isinstance(remote_latency, (int, float)):
                remote_latencies_ms.append(float(remote_latency))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    total_elapsed = perf_counter() - bench_started
    record.total_requests = request.iterations * effective_batch_size
    record.success_count = len(latencies_ms) * effective_batch_size
    record.failure_count = len(errors)
    record.avg_latency_ms = _average(latencies_ms)
    record.p95_latency_ms = _percentile(latencies_ms, 95)
    record.throughput_qps = round((record.success_count / total_elapsed), 3) if total_elapsed > 0 else None
    record.remote_avg_latency_ms = _average(remote_latencies_ms)
    record.sample_response = last_response
    record.errors = errors[:20]
    if record.success_count == 0:
        record.status = "failed"
    elif record.failure_count > 0:
        record.status = "partial_success"
    else:
        record.status = "completed"
    return record


def _post_predict(
    service_url: str,
    *,
    texts: list[str],
    options: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "texts": texts,
            "options": options,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        service_url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        body = response.read().decode("utf-8")
    return json.loads(body)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1))
    return round(ordered[rank], 3)
