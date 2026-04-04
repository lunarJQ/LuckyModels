from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16

from .manifest import load_manifest, load_profile
from .model_package_manager import (
    _allocate_model_id,
    _normalize_dataset_name,
    _normalize_language,
    _normalize_version,
    _unique_text_items,
    _write_package_metadata,
    _write_yaml,
    read_package_metadata,
)
from .schemas import ModelDetail, ModelOptimizationResult
from .settings import Settings

OPTIMIZATION_TYPE_ORT_GRAPH = "ort_graph"
OPTIMIZATION_TYPE_ORT_FORMAT = "ort_format"
OPTIMIZATION_TYPE_FP16 = "fp16"
OPTIMIZATION_TYPE_INT8_DYNAMIC = "int8_dynamic"

OPTIMIZATION_LABELS = {
    OPTIMIZATION_TYPE_ORT_GRAPH: "ORT 图优化",
    OPTIMIZATION_TYPE_ORT_FORMAT: "ORT Format 优化",
    OPTIMIZATION_TYPE_FP16: "FP16 精度优化",
    OPTIMIZATION_TYPE_INT8_DYNAMIC: "INT8 动态量化",
}

OPTIMIZATION_ALIAS_SUFFIX = {
    OPTIMIZATION_TYPE_ORT_GRAPH: "optgraph",
    OPTIMIZATION_TYPE_ORT_FORMAT: "ort",
    OPTIMIZATION_TYPE_FP16: "fp16",
    OPTIMIZATION_TYPE_INT8_DYNAMIC: "int8dyn",
}

SUPPORTED_OPTIMIZATION_TYPES = tuple(OPTIMIZATION_LABELS.keys())


def optimize_model_package(
    *,
    settings: Settings,
    model: ModelDetail,
    optimization_type: str,
    alias: str | None = None,
) -> ModelOptimizationResult:
    normalized_type = str(optimization_type or "").strip().lower()
    if normalized_type not in SUPPORTED_OPTIMIZATION_TYPES:
        raise ValueError(f"Unsupported optimization_type: {optimization_type}")

    source_root = Path(model.summary.package_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Package root not found: {source_root}")

    final_language = _normalize_language(model.summary.language)
    final_dataset_name = _normalize_dataset_name(model.summary.dataset_name or "dataset")
    final_version = _normalize_version(model.summary.version or "v1")
    final_alias = str(alias or OPTIMIZATION_ALIAS_SUFFIX[normalized_type]).strip() or OPTIMIZATION_ALIAS_SUFFIX[normalized_type]
    final_model_id = _allocate_model_id(
        artifacts_root=settings.artifacts_root,
        family_code=model.summary.model_family,
        language=final_language,
        dataset_name=final_dataset_name,
        version=final_version,
        alias=f"{model.summary.model_id}-{final_alias}",
    )
    target_root = settings.artifacts_root / model.summary.model_family / final_language / final_model_id
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        raise FileExistsError(f"Target package already exists: {target_root}")

    warnings: list[str] = []
    optimized_artifacts: dict[str, str] = {}
    shutil.copytree(source_root, target_root)

    try:
        manifest_data = model.manifest.model_dump(mode="python", exclude_none=True)
        manifest_data["model_id"] = final_model_id
        manifest_data["model_name"] = _build_optimized_model_name(model.summary.model_name, normalized_type)
        manifest_data["runtime_notes"] = _unique_text_items(
            [
                *[str(item).strip() for item in manifest_data.get("runtime_notes", []) if str(item).strip()],
                f"Optimized by LuckyModels workbench: {OPTIMIZATION_LABELS[normalized_type]}.",
                f"Source model: {model.summary.model_id}.",
            ]
        )

        for artifact_name, relative_path in (manifest_data.get("artifacts") or {}).items():
            artifact_relative = Path(relative_path)
            source_artifact = source_root / artifact_relative
            target_artifact = _resolve_target_artifact_path(
                target_root=target_root,
                artifact_relative=artifact_relative,
                optimization_type=normalized_type,
            )
            if source_artifact.suffix.lower() != ".onnx":
                warnings.append(f"{artifact_name} 不是 ONNX 文件，已原样保留。")
                continue
            if not source_artifact.exists():
                raise FileNotFoundError(f"Artifact not found: {source_artifact}")
            _optimize_single_artifact(
                source_path=source_artifact,
                target_path=target_artifact,
                optimization_type=normalized_type,
            )
            relative_target = str(target_artifact.relative_to(target_root))
            manifest_data["artifacts"][artifact_name] = relative_target
            optimized_artifacts[artifact_name] = relative_target

        if normalized_type == OPTIMIZATION_TYPE_INT8_DYNAMIC:
            _rewrite_profiles_for_int8(target_root, manifest_data.get("profiles") or {})

        _write_yaml(target_root / "manifest.yaml", manifest_data)
        _write_package_metadata(
            target_root=target_root,
            origin="generated",
            import_mode="platform_optimization",
            archive_filename="",
            extra_payload={
                "source_model_id": model.summary.model_id,
                "optimization_type": normalized_type,
                "optimization_label": OPTIMIZATION_LABELS[normalized_type],
                "optimized_artifact_count": len(optimized_artifacts),
            },
        )
        optimized_loaded = load_manifest(target_root / "manifest.yaml")
        _validate_optimized_package(optimized_loaded)
    except Exception:
        shutil.rmtree(target_root, ignore_errors=True)
        raise

    return ModelOptimizationResult(
        model_id=final_model_id,
        model_name=manifest_data["model_name"],
        model_family=model.summary.model_family,
        language=final_language,
        dataset_name=final_dataset_name,
        version=final_version,
        package_root=str(target_root),
        source_model_id=model.summary.model_id,
        optimization_type=normalized_type,
        optimization_label=OPTIMIZATION_LABELS[normalized_type],
        optimized_artifact_count=len(optimized_artifacts),
        optimized_artifacts=optimized_artifacts,
        warnings=warnings,
    )


def _build_optimized_model_name(model_name: str, optimization_type: str) -> str:
    suffix = {
        OPTIMIZATION_TYPE_ORT_GRAPH: "[ORT-Optimized]",
        OPTIMIZATION_TYPE_ORT_FORMAT: "[ORT-Format]",
        OPTIMIZATION_TYPE_FP16: "[FP16]",
        OPTIMIZATION_TYPE_INT8_DYNAMIC: "[INT8-Dynamic]",
    }[optimization_type]
    base_name = str(model_name or "").strip() or "Optimized Model"
    if suffix in base_name:
        return base_name
    return f"{base_name} {suffix}"


def _optimize_single_artifact(*, source_path: Path, target_path: Path, optimization_type: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(target_path.parent)) as temp_dir:
        temp_output = Path(temp_dir) / target_path.name
        if optimization_type == OPTIMIZATION_TYPE_ORT_GRAPH:
            _run_ort_graph_optimization(source_path, temp_output)
        elif optimization_type == OPTIMIZATION_TYPE_ORT_FORMAT:
            _run_ort_format_optimization(source_path, temp_output)
        elif optimization_type == OPTIMIZATION_TYPE_FP16:
            _run_fp16_conversion(source_path, temp_output)
        elif optimization_type == OPTIMIZATION_TYPE_INT8_DYNAMIC:
            _run_int8_dynamic_quantization(source_path, temp_output)
        else:
            raise ValueError(f"Unsupported optimization_type: {optimization_type}")
        shutil.move(str(temp_output), str(target_path))


def _resolve_target_artifact_path(*, target_root: Path, artifact_relative: Path, optimization_type: str) -> Path:
    if optimization_type == OPTIMIZATION_TYPE_ORT_FORMAT and artifact_relative.suffix.lower() == ".onnx":
        artifact_relative = artifact_relative.with_suffix(".ort")
    return target_root / artifact_relative


def _run_ort_graph_optimization(source_path: Path, output_path: Path) -> None:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = str(output_path)
    ort.InferenceSession(
        str(source_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    if not output_path.exists():
        raise RuntimeError(f"ORT graph optimization did not produce output: {output_path}")


def _run_ort_format_optimization(source_path: Path, output_path: Path) -> None:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session_options.optimized_model_filepath = str(output_path)
    session_options.add_session_config_entry("session.save_model_format", "ORT")
    ort.InferenceSession(
        str(source_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    if not output_path.exists():
        raise RuntimeError(f"ORT format optimization did not produce output: {output_path}")


def _run_fp16_conversion(source_path: Path, output_path: Path) -> None:
    model = onnx.load(str(source_path))
    converted = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save(converted, str(output_path))


def _run_int8_dynamic_quantization(source_path: Path, output_path: Path) -> None:
    quantize_dynamic(
        model_input=str(source_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    if not output_path.exists():
        raise RuntimeError(f"INT8 dynamic quantization did not produce output: {output_path}")


def _rewrite_profiles_for_int8(target_root: Path, profile_mapping: dict[str, str]) -> None:
    for relative_path in profile_mapping.values():
        profile_path = target_root / relative_path
        profile = load_profile(profile_path)
        profile.runtime = "onnxruntime"
        profile.device = "cpu"
        profile.gpu_ids = []
        profile.execution["precision"] = "int8"
        _write_yaml(profile_path, profile.model_dump(mode="python", exclude_none=True))


def _validate_optimized_package(loaded_manifest: Any) -> None:
    for _, artifact_path in loaded_manifest.resolved_artifacts.items():
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Optimized artifact missing: {path}")
        if path.suffix.lower() == ".onnx":
            onnx.load(str(path))
        elif path.suffix.lower() == ".ort":
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def list_supported_optimization_types() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for optimization_type in SUPPORTED_OPTIMIZATION_TYPES:
        note = {
            OPTIMIZATION_TYPE_ORT_GRAPH: "生成离线 ORT 图优化后的 ONNX 制品，适合直接替换原始 ONNX 做部署与 benchmark。",
            OPTIMIZATION_TYPE_ORT_FORMAT: "生成 ORT Format 制品，减少运行时图初始化开销，更适合固定环境内长期部署。",
            OPTIMIZATION_TYPE_FP16: "将 ONNX 制品转换为 FP16 版本，适合 GPU 推理场景。",
            OPTIMIZATION_TYPE_INT8_DYNAMIC: "生成 INT8 动态量化制品，当前默认更适合 CPU 推理。",
        }[optimization_type]
        results.append(
            {
                "code": optimization_type,
                "label": OPTIMIZATION_LABELS[optimization_type],
                "alias_suffix": OPTIMIZATION_ALIAS_SUFFIX[optimization_type],
                "note": note,
            }
        )
    return results


def get_model_optimization_summary(model: ModelDetail) -> dict[str, Any]:
    metadata = read_package_metadata(model.summary.package_root)
    return {
        "is_generated": str(metadata.get("origin") or "") == "generated",
        "optimization_type": metadata.get("optimization_type"),
        "optimization_label": metadata.get("optimization_label"),
        "source_model_id": metadata.get("source_model_id"),
        "optimized_artifact_count": metadata.get("optimized_artifact_count"),
    }
