from __future__ import annotations

import csv
import subprocess
from io import StringIO
from typing import Any

from fastapi import APIRouter

from luckymodels_core.schemas import SystemSummary
from luckymodels_core.settings import get_settings

from ..dependencies import get_benchmark_store, get_deployment_store, get_prediction_store, get_registry

router = APIRouter(tags=["system"])


@router.get("/system/summary", response_model=SystemSummary)
def get_system_summary() -> SystemSummary:
    registry = get_registry()
    deployment_store = get_deployment_store()
    benchmark_store = get_benchmark_store()
    prediction_store = get_prediction_store()
    settings = get_settings()
    return SystemSummary(
        model_count=len(registry.list_models()),
        deployment_count=len(deployment_store.list()),
        benchmark_count=len(benchmark_store.list()),
        prediction_archive_count=len(prediction_store.list()),
        artifact_root=str(settings.artifacts_root),
        logs_root=str(settings.logs_root),
    )


@router.get("/system/gpus")
def get_gpu_status() -> dict[str, Any]:
    deployment_store = get_deployment_store()
    deployments = deployment_store.list()

    try:
        devices = _query_nvidia_smi(
            [
                "index",
                "uuid",
                "name",
                "memory.total",
                "memory.used",
                "utilization.gpu",
                "temperature.gpu",
                "power.draw",
            ],
            scope="gpu",
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "detail": str(exc),
            "gpu_count": 0,
            "devices": [],
        }

    processes: list[dict[str, Any]]
    try:
        processes = _query_nvidia_smi(
            ["gpu_uuid", "pid", "process_name", "used_gpu_memory"],
            scope="compute-apps",
        )
    except Exception:
        processes = []

    deployment_by_pid = {
        int(item.pid): item
        for item in deployments
        if item.pid and item.status in {"running", "starting", "planned", "stopping"}
    }

    process_rows: list[dict[str, Any]] = []
    for row in processes:
        pid = _to_int(row.get("pid"))
        deployment = deployment_by_pid.get(pid) if pid is not None else None
        process_rows.append(
            {
                "gpu_uuid": row.get("gpu_uuid", ""),
                "pid": pid,
                "process_name": row.get("process_name", ""),
                "used_gpu_memory_mb": _to_float(row.get("used_gpu_memory")),
                "deployment_id": deployment.deployment_id if deployment else None,
                "model_id": deployment.model_id if deployment else None,
                "profile_name": deployment.profile_name if deployment else None,
                "assigned_gpu_ids": deployment.gpu_ids if deployment else [],
            }
        )

    devices_by_uuid = {str(item.get("uuid", "")): item for item in devices}
    for device in devices:
        device["index"] = _to_int(device.get("index"))
        device["memory_total_mb"] = _to_float(device.pop("memory.total", None))
        device["memory_used_mb"] = _to_float(device.pop("memory.used", None))
        device["utilization_gpu_pct"] = _to_float(device.pop("utilization.gpu", None))
        device["temperature_c"] = _to_float(device.pop("temperature.gpu", None))
        device["power_draw_w"] = _to_float(device.pop("power.draw", None))
        device["assigned_deployments"] = []
        device["active_processes"] = []

    for deployment in deployments:
        if not deployment.gpu_ids:
            continue
        for gpu_id in deployment.gpu_ids:
            matched = next((item for item in devices if item.get("index") == gpu_id), None)
            if matched is None:
                continue
            matched["assigned_deployments"].append(
                {
                    "deployment_id": deployment.deployment_id,
                    "model_id": deployment.model_id,
                    "profile_name": deployment.profile_name,
                    "status": deployment.status,
                    "use_gpu": deployment.use_gpu,
                    "service_url": deployment.service_url,
                    "pid": deployment.pid,
                }
            )

    for process in process_rows:
        device = devices_by_uuid.get(process.get("gpu_uuid", ""))
        if device is None:
            continue
        device["active_processes"].append(process)

    return {
        "available": True,
        "gpu_count": len(devices),
        "devices": devices,
        "process_count": len(process_rows),
    }


def _query_nvidia_smi(fields: list[str], *, scope: str) -> list[dict[str, str]]:
    query = ",".join(fields)
    result = subprocess.run(  # noqa: S603
        [
            "nvidia-smi",
            f"--query-{scope}={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    if not raw:
        return []
    reader = csv.reader(StringIO(raw))
    rows: list[dict[str, str]] = []
    for line in reader:
        if not line:
            continue
        rows.append(
            {
                field: (line[index].strip() if index < len(line) else "")
                for index, field in enumerate(fields)
            }
        )
    return rows


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"[N/A]", "N/A", "Not Supported"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    if number is None:
        return None
    return int(number)
