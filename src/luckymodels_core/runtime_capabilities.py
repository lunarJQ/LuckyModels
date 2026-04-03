from __future__ import annotations

from pathlib import Path
from typing import Any

from .settings import Settings


def inspect_runtime_capabilities(settings: Settings) -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        return {
            "onnxruntime_installed": False,
            "detail": str(exc),
            "providers": [],
            "features": {
                "ort_graph_optimization": False,
                "ort_format_export": False,
                "session_config_entries": False,
                "provider_options": False,
                "profiling": False,
            },
        }

    session_options = ort.SessionOptions()
    available_providers = list(ort.get_available_providers())
    probe_model = _find_probe_model(settings.artifacts_root)
    provider_rows = [
        _probe_provider(
            ort=ort,
            provider_name=provider_name,
            available_providers=available_providers,
            probe_model=probe_model,
        )
        for provider_name in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    ]
    return {
        "onnxruntime_installed": True,
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "probe_model_path": str(probe_model) if probe_model else None,
        "providers": provider_rows,
        "features": {
            "ort_graph_optimization": hasattr(session_options, "optimized_model_filepath")
            and hasattr(session_options, "graph_optimization_level"),
            "ort_format_export": hasattr(session_options, "optimized_model_filepath")
            and hasattr(session_options, "add_session_config_entry"),
            "session_config_entries": hasattr(session_options, "add_session_config_entry"),
            "provider_options": True,
            "profiling": hasattr(session_options, "enable_profiling"),
        },
    }


def _find_probe_model(artifacts_root: Path) -> Path | None:
    if not artifacts_root.exists():
        return None
    return next(artifacts_root.rglob("*.onnx"), None)


def _probe_provider(
    *,
    ort: Any,
    provider_name: str,
    available_providers: list[str],
    probe_model: Path | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "provider": provider_name,
        "available_in_wheel": provider_name in available_providers,
        "runtime_usable": False,
        "active_providers": [],
        "detail": "",
    }
    if provider_name not in available_providers:
        row["detail"] = "当前 onnxruntime wheel 未内置该 provider。"
        return row
    _preload_runtime_dlls(ort=ort, provider_name=provider_name)
    if provider_name == "TensorrtExecutionProvider":
        missing_libraries = _missing_tensorrt_runtime_libraries()
        if missing_libraries:
            row["detail"] = f"缺少底层运行库：{', '.join(missing_libraries)}"
            return row
    if probe_model is None:
        row["detail"] = "平台当前没有可用于探测的 ONNX 制品，无法做运行时探测。"
        return row
    try:
        session = ort.InferenceSession(str(probe_model), providers=[provider_name])
        active = list(session.get_providers())
        row["active_providers"] = active
        row["runtime_usable"] = provider_name in active
        if row["runtime_usable"]:
            row["detail"] = "可正常初始化。"
        else:
            row["detail"] = "provider 已在 wheel 中声明，但初始化时回退到了其他执行后端。"
    except Exception as exc:  # noqa: BLE001
        row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def _preload_runtime_dlls(*, ort: Any, provider_name: str) -> None:
    if provider_name not in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        return
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload(directory="")
        except Exception:
            pass


def _missing_tensorrt_runtime_libraries() -> list[str]:
    try:
        import ctypes
    except ImportError:
        return ["ctypes"]

    requirements = [
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
    ]
    missing: list[str] = []
    for library_name in requirements:
        try:
            ctypes.CDLL(library_name)
        except OSError:
            missing.append(library_name)
    return missing
