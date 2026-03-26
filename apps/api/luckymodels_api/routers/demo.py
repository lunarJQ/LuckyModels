from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import BenchmarkCreateRequest, DeploymentCreateRequest
from luckymodels_core.settings import get_settings

from .benchmarks import create_benchmark
from .deployments import (
    check_deployment_health,
    create_deployment,
    get_deployment_logs,
    stop_deployment,
)

router = APIRouter(tags=["demo"])


@router.get("/demo/casrel/report")
def get_casrel_demo_report() -> dict[str, object]:
    report_file = get_settings().casrel_demo_report_file
    if not report_file.exists():
        return {
            "status": "empty",
            "detail": "No CasRel demo report has been generated yet.",
        }

    with report_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@router.post("/demo/casrel/run")
def run_casrel_demo() -> dict[str, object]:
    settings = get_settings()
    settings.reports_root.mkdir(parents=True, exist_ok=True)

    deployment = None
    report: dict[str, object] = {}
    try:
        deployment = create_deployment(
            DeploymentCreateRequest(
                model_id="casrel-zh-v1",
                profile_name="default",
                launch_runtime=True,
                notes="casrel_demo_run",
            )
        )
        if not deployment.service_url:
            raise HTTPException(status_code=500, detail="Deployment did not produce a service_url")

        health = check_deployment_health(deployment.deployment_id)
        benchmark = create_benchmark(
            BenchmarkCreateRequest(
                deployment_id=deployment.deployment_id,
                text="The 1st runway length in feet is 14000.",
                iterations=5,
                warmup_runs=1,
                label="casrel_demo_run",
                options={
                    "subject_threshold": 0.001,
                    "object_threshold": 0.001,
                },
            )
        )
        chinese_debug = _predict_debug(
            deployment.service_url,
            "ABC设备主轴发生振动异常，需要检查主轴轴承。",
        )
        english_debug = _predict_debug(
            deployment.service_url,
            "The 1st runway length in feet is 14000.",
        )
        logs = get_deployment_logs(deployment.deployment_id, lines=80)

        report = {
            "status": "ready",
            "deployment": deployment.model_dump(),
            "health": health,
            "benchmark": benchmark.model_dump(),
            "samples": {
                "chinese_debug": chinese_debug,
                "english_debug": english_debug,
            },
            "logs": logs,
            "report_note": (
                "This report demonstrates the LuckyModels control plane, runtime launch, "
                "benchmark path, and CasRel debug output for the current uploaded artifact."
            ),
        }
    finally:
        if deployment is not None:
            try:
                stopped = stop_deployment(deployment.deployment_id)
                report["deployment_stopped"] = stopped.model_dump()
            except Exception as exc:  # noqa: BLE001
                report["stop_error"] = str(exc)

    with Path(settings.casrel_demo_report_file).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def _predict_debug(service_url: str, text: str) -> dict[str, object]:
    payload = json.dumps(
        {
            "text": text,
            "options": {
                "subject_threshold": 0.001,
                "object_threshold": 0.001,
                "debug": True,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        service_url.rstrip("/") + "/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))
