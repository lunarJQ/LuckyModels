from fastapi import APIRouter, HTTPException

from luckymodels_core.benchmark_runner import run_benchmark
from luckymodels_core.dataset_samples import get_dataset_random_sample_text
from luckymodels_core.manifest import load_profile
from luckymodels_core.schemas import BenchmarkCreateRequest, BenchmarkRecord

from ..dependencies import get_benchmark_store, get_deployment_store, get_registry
from ..permissions import require_permission
from .deployments import _resolve_benchmark_dataset_name, _resolve_benchmark_text_for_deployment

router = APIRouter(tags=["benchmarks"])


@router.get("/benchmarks", response_model=list[BenchmarkRecord])
def list_benchmarks() -> list[BenchmarkRecord]:
    require_permission("benchmarks.view")
    return get_benchmark_store().list()


@router.get("/benchmarks/sample-text")
def get_benchmark_sample_text(
    deployment_id: str,
    scene: str | None = None,
    exclude_case_id: str | None = None,
) -> dict[str, object]:
    require_permission("benchmarks.manage")
    deployment = _get_deployment_or_404(deployment_id)
    try:
        model = get_registry().get_model(deployment.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_name = _resolve_benchmark_dataset_name(
        model_dataset_name=model.summary.dataset_name,
        scene=scene,
    )
    if not dataset_name:
        raise HTTPException(status_code=400, detail=f"Model has no dataset_name: {deployment.model_id}")

    try:
        payload = get_dataset_random_sample_text(
            dataset_name,
            split_preference="train",
            exclude_case_id=exclude_case_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "deployment_id": deployment.deployment_id,
        "model_id": deployment.model_id,
        "profile_name": deployment.profile_name,
        "scene": scene,
        "dataset_name": dataset_name,
        **payload,
    }


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkRecord)
def get_benchmark(benchmark_id: str) -> BenchmarkRecord:
    require_permission("benchmarks.view")
    for record in get_benchmark_store().list():
        if record.benchmark_id == benchmark_id:
            return record
    raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")


@router.post("/benchmarks", response_model=BenchmarkRecord)
def create_benchmark(request: BenchmarkCreateRequest) -> BenchmarkRecord:
    require_permission("benchmarks.manage")
    deployment = _get_deployment_or_404(request.deployment_id)
    if not deployment.service_url:
        raise HTTPException(status_code=400, detail="Deployment is missing a service_url")
    if deployment.status not in {"running", "planned", "starting"}:
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not runnable for benchmark: {deployment.status}",
        )

    model_summary = get_registry().get_model(deployment.model_id).summary
    effective_request = request.model_copy(
        update={
            "text": _resolve_benchmark_text_for_deployment(
                deployment=deployment,
                model_dataset_name=model_summary.dataset_name,
                scene=request.scene,
                requested_text=request.text,
            )
        }
    )
    store = get_benchmark_store()
    record = store.create(
        effective_request,
        model_id=deployment.model_id,
        profile_name=deployment.profile_name,
        service_url=deployment.service_url,
        deployment=deployment,
        model=model_summary,
    )
    batch_size = 1
    if deployment.profile_path:
        profile = load_profile(deployment.profile_path)
        batch_size = max(1, int(profile.execution.get("batch_size", 1) or 1))
    finished = run_benchmark(effective_request, record, batch_size=batch_size)
    return store.update(finished)


@router.delete("/benchmarks/{benchmark_id}", response_model=BenchmarkRecord)
def delete_benchmark(benchmark_id: str) -> BenchmarkRecord:
    require_permission("benchmarks.manage")
    store = get_benchmark_store()
    deleted = store.delete(benchmark_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")
    return deleted


def _get_deployment_or_404(deployment_id: str):
    for deployment in get_deployment_store().list():
        if deployment.deployment_id == deployment_id:
            return deployment
    raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")
