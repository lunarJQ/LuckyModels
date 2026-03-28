from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException

from luckymodels_core.schemas import BenchmarkCreateRequest, DeploymentCreateRequest
from luckymodels_core.settings import get_settings
from luckymodels_core.webnlg_demo import export_webnlg_eval_cases, get_webnlg_demo_cases

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
    demo_cases = get_webnlg_demo_cases()
    prepared_eval = export_webnlg_eval_cases()
    benchmark_case = demo_cases["language_case"]
    primary_case = demo_cases["location_case"]
    secondary_case = demo_cases["ethnic_group_case"]

    deployment = None
    report: dict[str, object] = {}
    try:
        deployment = create_deployment(
            DeploymentCreateRequest(
                model_id="casrel-en-webnlg-v1",
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
                text=benchmark_case["sentText"],
                iterations=5,
                warmup_runs=1,
                label=f"casrel_demo_{benchmark_case['name']}",
                options={
                    "subject_threshold": benchmark_case["threshold"],
                    "object_threshold": benchmark_case["threshold"],
                },
            )
        )
        primary_runtime = _predict_debug(
            deployment.service_url,
            primary_case["sentText"],
            threshold=primary_case["threshold"],
        )
        secondary_runtime = _predict_debug(
            deployment.service_url,
            secondary_case["sentText"],
            threshold=secondary_case["threshold"],
        )
        logs = get_deployment_logs(deployment.deployment_id, lines=80)

        report = {
            "status": "ready",
            "deployment": deployment.model_dump(),
            "health": health,
            "benchmark": benchmark.model_dump(),
            "dataset": {
                "name": "WebNLG dev",
                "source": prepared_eval["dataset_source"],
                "prepared_eval_file": prepared_eval["output_path"],
                "prepared_eval_count": prepared_eval["record_count"],
                "selected_cases": [
                    benchmark_case["name"],
                    primary_case["name"],
                    secondary_case["name"],
                ],
                "note": "CasRel uploaded artifact is evaluated with real WebNLG text samples reconstructed from the local dataset.",
            },
            "samples": {
                "primary_case": {
                    "case_name": primary_case["name"],
                    "description": primary_case["description"],
                    "threshold": primary_case["threshold"],
                    "sentText": primary_case["sentText"],
                    "gold_relations": primary_case["relationMentions"],
                    "runtime": primary_runtime,
                },
                "secondary_case": {
                    "case_name": secondary_case["name"],
                    "description": secondary_case["description"],
                    "threshold": secondary_case["threshold"],
                    "sentText": secondary_case["sentText"],
                    "gold_relations": secondary_case["relationMentions"],
                    "runtime": secondary_runtime,
                },
            },
            "logs": logs,
            "report_note": (
                "This report demonstrates the LuckyModels control plane, runtime launch, "
                "benchmark path, and real WebNLG dataset-driven CasRel outputs for the current uploaded artifact."
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


@router.post("/demo/casrel/prepare-webnlg")
def prepare_casrel_webnlg_eval(limit: int = 128) -> dict[str, object]:
    return export_webnlg_eval_cases(limit=limit)


def _predict_debug(service_url: str, text: str, threshold: float) -> dict[str, object]:
    payload = json.dumps(
        {
            "text": text,
            "options": {
                "subject_threshold": threshold,
                "object_threshold": threshold,
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
