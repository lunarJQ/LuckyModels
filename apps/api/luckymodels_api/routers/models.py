import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from luckymodels_core.dataset_samples import (
    DEFAULT_ONLINE_SAMPLE_TEXT,
    export_dataset_eval_cases,
    get_dataset_eval_cases,
    get_dataset_sample_text,
    list_dataset_catalog,
)
from luckymodels_core.launcher import stop_runtime_process
from luckymodels_core.manifest import load_profile
from luckymodels_core.model_optimizer import (
    list_supported_optimization_types,
    optimize_model_package,
)
from luckymodels_core.model_package_manager import (
    export_model_package,
    get_model_import_options,
    import_model_archive,
)
from luckymodels_core.schemas import (
    CapabilityMatrixRow,
    ModelDetail,
    ModelDeleteResult,
    ModelImportOptions,
    ModelImportResult,
    ModelOptimizeRequest,
    ModelOptimizationOptions,
    ModelOptimizationResult,
    ModelSummary,
    ProfileConfig,
)
from luckymodels_core.settings import get_settings

from ..dependencies import (
    get_benchmark_store,
    get_deployment_store,
    get_prediction_store,
    get_registry,
    get_task_routing_feedback_store,
)
from ..permissions import require_admin, require_permission

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelSummary])
def list_models() -> list[ModelSummary]:
    require_permission("models.view")
    return get_registry().list_models()


@router.get("/models/import-options", response_model=ModelImportOptions)
def get_supported_model_import_options() -> ModelImportOptions:
    require_permission("models.import")
    return get_model_import_options()


@router.get("/models/optimization-options", response_model=ModelOptimizationOptions)
def get_supported_model_optimization_options() -> ModelOptimizationOptions:
    require_admin()
    return ModelOptimizationOptions(options=list_supported_optimization_types())


@router.get("/datasets/catalog")
def get_dataset_catalog() -> dict[str, object]:
    require_permission("predictions.view")
    return {
        "datasets": list_dataset_catalog(),
        "default_fallback_text": DEFAULT_ONLINE_SAMPLE_TEXT,
    }


@router.get("/datasets/sample-text")
def get_dataset_sample_text_preview(dataset_name: str, index: int = 0) -> dict[str, object]:
    require_permission("predictions.view")
    return get_dataset_sample_text(dataset_name, index=index)


@router.post("/models/import", response_model=ModelImportResult)
def import_model(
    archive: UploadFile = File(...),
    import_mode: str = Form(...),
    model_family: str | None = Form(None),
    language: str | None = Form(None),
    dataset_name: str | None = Form(None),
    version: str | None = Form(None),
    model_name: str | None = Form(None),
    backbone: str | None = Form(None),
    alias: str | None = Form(None),
) -> ModelImportResult:
    require_permission("models.import")
    settings = get_settings()
    settings.imports_root.mkdir(parents=True, exist_ok=True)
    suffix = "".join(Path(archive.filename or "").suffixes) or ".zip"
    with tempfile.NamedTemporaryFile(
        dir=settings.imports_root,
        suffix=suffix,
        delete=False,
    ) as handle:
        temp_archive_path = Path(handle.name)
        shutil.copyfileobj(archive.file, handle)

    try:
        return import_model_archive(
            settings=settings,
            archive_path=temp_archive_path,
            archive_filename=archive.filename or temp_archive_path.name,
            import_mode=import_mode,
            source_type="user_upload",
            model_family=model_family,
            language=language,
            dataset_name=dataset_name,
            version=version,
            model_name=model_name,
            backbone=backbone,
            alias=alias,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temp_archive_path.unlink(missing_ok=True)
        archive.file.close()


@router.get("/models/capability-matrix", response_model=list[CapabilityMatrixRow])
def get_model_capability_matrix() -> list[CapabilityMatrixRow]:
    require_permission("models.view")
    registry = get_registry()
    deployments = get_deployment_store().list()
    benchmarks = get_benchmark_store().list()
    rows: list[CapabilityMatrixRow] = []

    for model in registry.list_models():
        detail = registry.get_model(model.model_id)
        model_deployments = [item for item in deployments if item.model_id == model.model_id]
        model_benchmarks = [item for item in benchmarks if item.model_id == model.model_id]
        completed_benchmarks = [item for item in model_benchmarks if item.status == "completed"]
        limitations = _collect_limitations(detail, completed_benchmarks)
        rows.append(
            CapabilityMatrixRow(
                model_id=model.model_id,
                model_name=model.model_name,
                model_family=model.model_family,
                language=model.language,
                dataset_name=model.dataset_name,
                runtime_status=model.runtime_status,
                inference_ready=model.runtime_status == "inference_ready",
                supports_online_predict=bool(model.predict_ready and model.adapter_available),
                supports_benchmark=bool(model.predict_ready and detail.resolved_profiles),
                supports_batch_predict=bool(model.predict_ready and model.adapter_available),
                profile_count=len(detail.resolved_profiles),
                benchmark_count=len(model_benchmarks),
                deployment_count=len(model_deployments),
                latest_benchmark_id=completed_benchmarks[-1].benchmark_id if completed_benchmarks else None,
                latest_deployment_id=model_deployments[-1].deployment_id if model_deployments else None,
                known_limitations="；".join(limitations),
            )
        )

    return rows


@router.get("/models/{model_id}", response_model=ModelDetail)
def get_model(model_id: str) -> ModelDetail:
    require_permission("models.view")
    try:
        return get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/models/{model_id}/optimize", response_model=ModelOptimizationResult)
def optimize_model(model_id: str, payload: ModelOptimizeRequest) -> ModelOptimizationResult:
    require_admin()
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        return optimize_model_package(
            settings=get_settings(),
            model=model,
            optimization_type=payload.optimization_type,
            alias=payload.alias,
        )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/{model_id}", response_model=ModelDeleteResult)
def delete_model(model_id: str) -> ModelDeleteResult:
    require_admin()
    registry = get_registry()
    try:
        model = registry.get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not model.summary.deletable:
        raise HTTPException(status_code=403, detail=f"模型制品不可删除：{model_id}（仅用户上传或平台生成模型支持删除）")

    settings = get_settings()
    package_root = Path(model.summary.package_root).resolve()
    artifacts_root = settings.artifacts_root.resolve()
    try:
        package_root.relative_to(artifacts_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"非法模型包路径：{package_root}") from exc

    deployment_store = get_deployment_store()
    benchmark_store = get_benchmark_store()
    prediction_store = get_prediction_store()
    task_routing_feedback_store = get_task_routing_feedback_store()

    stopped_runtime_count = 0
    deleted_log_file_count = 0

    deployments = [item for item in deployment_store.list() if item.model_id == model_id]
    for deployment in deployments:
        if deployment.pid and stop_runtime_process(deployment.pid):
            stopped_runtime_count += 1
        if deployment.log_path:
            log_path = Path(deployment.log_path)
            if log_path.exists():
                log_path.unlink(missing_ok=True)
                deleted_log_file_count += 1
        deployment_store.delete(deployment.deployment_id)

    benchmarks = [item for item in benchmark_store.list() if item.model_id == model_id]
    for benchmark in benchmarks:
        benchmark_store.delete(benchmark.benchmark_id)

    prediction_archives = [item for item in prediction_store.list() if item.model_id == model_id]
    for archive in prediction_archives:
        prediction_store.delete(archive.archive_id)

    deleted_task_routing_decision_count, deleted_task_routing_feedback_count = (
        task_routing_feedback_store.delete_by_model(model_id)
    )

    if package_root.exists():
        shutil.rmtree(package_root)

    return ModelDeleteResult(
        model_id=model_id,
        deleted_deployment_count=len(deployments),
        stopped_runtime_count=stopped_runtime_count,
        deleted_benchmark_count=len(benchmarks),
        deleted_prediction_archive_count=len(prediction_archives),
        deleted_task_routing_decision_count=deleted_task_routing_decision_count,
        deleted_task_routing_feedback_count=deleted_task_routing_feedback_count,
        deleted_log_file_count=deleted_log_file_count,
        removed_package_root=str(package_root),
    )


@router.get("/models/{model_id}/export")
def export_model(model_id: str) -> Response:
    require_permission("archive.export")
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        filename, payload = export_model_package(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(payload)),
        },
    )


@router.get("/models/{model_id}/profiles", response_model=dict[str, ProfileConfig])
def get_model_profiles(model_id: str) -> dict[str, ProfileConfig]:
    require_permission("models.view")
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {name: load_profile(path) for name, path in model.resolved_profiles.items()}


@router.get("/models/{model_id}/sample-cases")
def get_model_sample_cases(model_id: str, limit: int = 20) -> dict[str, object]:
    require_permission("models.view")
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_name = model.summary.dataset_name
    if not dataset_name:
        raise HTTPException(status_code=400, detail=f"Model has no dataset_name: {model_id}")

    try:
        payload = get_dataset_eval_cases(dataset_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        **payload,
    }


@router.post("/models/{model_id}/prepare-eval")
def prepare_model_eval_cases(model_id: str, limit: int = 128) -> dict[str, object]:
    require_admin()
    try:
        model = get_registry().get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    dataset_name = model.summary.dataset_name
    if not dataset_name:
        raise HTTPException(status_code=400, detail=f"Model has no dataset_name: {model_id}")

    try:
        payload = export_dataset_eval_cases(dataset_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "model_id": model_id,
        "dataset_name": dataset_name,
        **payload,
    }


def _collect_limitations(
    detail: ModelDetail,
    completed_benchmarks,
) -> list[str]:
    limitations: list[str] = []
    for note in detail.manifest.runtime_notes:
        text = str(note).strip()
        if text and text not in limitations:
            limitations.append(text)
    for note in detail.manifest.deployment_requirements:
        text = str(note).strip()
        if text and text not in limitations:
            limitations.append(text)
    if completed_benchmarks:
        warnings = completed_benchmarks[-1].sample_response.get("warnings") or []
        for warning in warnings[:3]:
            text = str(warning).strip()
            if text and text not in limitations:
                limitations.append(text)
    else:
        limitations.append("尚未生成该模型的 benchmark 证据。")
    if not limitations:
        limitations.append("暂无显式限制。")
    return limitations[:4]
