from fastapi import APIRouter, HTTPException

from luckymodels_core.benchmark_runner import run_benchmark
from luckymodels_core.manifest import load_profile
from luckymodels_core.schemas import BenchmarkCreateRequest, BenchmarkRecord

from ..dependencies import get_benchmark_store, get_deployment_store

router = APIRouter(tags=["benchmarks"])


@router.get("/benchmarks", response_model=list[BenchmarkRecord])
def list_benchmarks() -> list[BenchmarkRecord]:
    return get_benchmark_store().list()


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkRecord)
def get_benchmark(benchmark_id: str) -> BenchmarkRecord:
    for record in get_benchmark_store().list():
        if record.benchmark_id == benchmark_id:
            return record
    raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")


@router.post("/benchmarks", response_model=BenchmarkRecord)
def create_benchmark(request: BenchmarkCreateRequest) -> BenchmarkRecord:
    deployment = _get_deployment_or_404(request.deployment_id)
    if not deployment.service_url:
        raise HTTPException(status_code=400, detail="Deployment is missing a service_url")
    if deployment.status not in {"running", "planned", "starting"}:
        raise HTTPException(
            status_code=400,
            detail=f"Deployment is not runnable for benchmark: {deployment.status}",
        )

    store = get_benchmark_store()
    record = store.create(
        request,
        model_id=deployment.model_id,
        profile_name=deployment.profile_name,
        service_url=deployment.service_url,
    )
    batch_size = 1
    if deployment.profile_path:
        profile = load_profile(deployment.profile_path)
        batch_size = max(1, int(profile.execution.get("batch_size", 1) or 1))
    finished = run_benchmark(request, record, batch_size=batch_size)
    return store.update(finished)


@router.delete("/benchmarks/{benchmark_id}", response_model=BenchmarkRecord)
def delete_benchmark(benchmark_id: str) -> BenchmarkRecord:
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
