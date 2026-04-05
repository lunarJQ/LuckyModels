from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import load_profile
from .schemas import BenchmarkRecord, DeploymentRecord, PredictionArchiveRecord, utc_now


REVISION_POLICY: dict[str, dict[str, float | int]] = {
    "initial": {
        "benchmark": 0.0,
        "archive": 0.0,
        "feedback": 0.0,
        "latency_feedback": 0.0,
        "benchmark_target": 1,
        "archive_target": 1,
    },
    "minor": {
        "benchmark": 0.70,
        "archive": 0.80,
        "feedback": 0.90,
        "latency_feedback": 0.60,
        "benchmark_target": 1,
        "archive_target": 3,
    },
    "medium": {
        "benchmark": 0.45,
        "archive": 0.60,
        "feedback": 0.75,
        "latency_feedback": 0.35,
        "benchmark_target": 2,
        "archive_target": 5,
    },
    "major": {
        "benchmark": 0.15,
        "archive": 0.25,
        "feedback": 0.50,
        "latency_feedback": 0.10,
        "benchmark_target": 2,
        "archive_target": 10,
    },
}


def effective_revision_started_at(deployment: DeploymentRecord) -> str:
    return str(deployment.revision_started_at or deployment.created_at or "")


def normalized_revision_change_level(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"minor", "medium", "major"}:
        return normalized
    if normalized == "initial":
        return "initial"
    return "initial"


def revision_policy_for(deployment: DeploymentRecord) -> dict[str, float | int]:
    return dict(REVISION_POLICY[normalized_revision_change_level(deployment.revision_change_level)])


def initialize_deployment_revision_metadata(deployment: DeploymentRecord) -> DeploymentRecord:
    deployment.runtime_revision = max(1, int(deployment.runtime_revision or 1))
    if not str(deployment.revision_started_at or "").strip():
        deployment.revision_started_at = str(deployment.created_at or utc_now())
    if not str(deployment.provider_preference or "").strip():
        deployment.provider_preference = "cuda" if deployment.use_gpu else "cpu"
    deployment.revision_change_level = normalized_revision_change_level(deployment.revision_change_level)
    ensure_runtime_signature(deployment)
    return deployment


def advance_deployment_revision(
    deployment: DeploymentRecord,
    *,
    change_level: str,
    runtime_signature: str | None = None,
) -> DeploymentRecord:
    initialize_deployment_revision_metadata(deployment)
    normalized_level = normalized_revision_change_level(change_level)
    if normalized_level == "initial":
        normalized_level = "minor"
    deployment.previous_revision_started_at = effective_revision_started_at(deployment)
    deployment.runtime_revision = max(1, int(deployment.runtime_revision or 1)) + 1
    deployment.revision_started_at = utc_now()
    deployment.revision_change_level = normalized_level
    deployment.runtime_signature = str(runtime_signature or "").strip() or ensure_runtime_signature(deployment)
    return deployment


def build_runtime_revision_signature(
    *,
    manifest_path: str | None,
    profile_path: str | None,
    use_gpu: bool,
    gpu_ids: list[int] | None,
    provider_preference: str | None,
    profile_runtime: str | None,
    device: str | None,
    execution: dict[str, Any] | None,
) -> str:
    payload = {
        "manifest_path": str(manifest_path or ""),
        "profile_path": str(profile_path or ""),
        "use_gpu": bool(use_gpu),
        "gpu_ids": [int(item) for item in gpu_ids or []],
        "provider_preference": _normalize_provider(provider_preference),
        "profile_runtime": str(profile_runtime or "").strip().lower(),
        "device": str(device or "").strip().lower(),
        "precision": _normalize_precision(execution),
        "execution": _normalize_execution(execution),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def ensure_runtime_signature(deployment: DeploymentRecord) -> str:
    signature = str(deployment.runtime_signature or "").strip()
    if signature:
        return signature
    signature = build_runtime_revision_signature(
        manifest_path=deployment.manifest_path,
        profile_path=deployment.profile_path,
        use_gpu=deployment.use_gpu,
        gpu_ids=deployment.gpu_ids,
        provider_preference=deployment.provider_preference,
        profile_runtime=deployment.profile_runtime,
        device=deployment.device,
        execution=deployment.execution,
    )
    deployment.runtime_signature = signature
    return signature


def classify_runtime_change(
    deployment: DeploymentRecord,
    *,
    manifest_path: str | None,
    profile_name: str,
    profile_path: str | None,
    use_gpu: bool,
    gpu_ids: list[int] | None,
    provider_preference: str | None,
) -> tuple[str, str]:
    current_runtime = _runtime_descriptor_from_deployment(deployment)
    target_runtime = _runtime_descriptor_from_target(
        deployment=deployment,
        manifest_path=manifest_path,
        profile_name=profile_name,
        profile_path=profile_path,
        use_gpu=use_gpu,
        gpu_ids=gpu_ids,
        provider_preference=provider_preference,
    )
    current_signature = build_runtime_revision_signature(**current_runtime)
    target_signature = build_runtime_revision_signature(**target_runtime)
    if current_signature == target_signature:
        return ("none", target_signature)
    if bool(current_runtime["use_gpu"]) != bool(target_runtime["use_gpu"]):
        return ("major", target_signature)
    if _normalize_provider(current_runtime["provider_preference"]) != _normalize_provider(target_runtime["provider_preference"]):
        return ("major", target_signature)
    if _normalize_precision(current_runtime["execution"]) != _normalize_precision(target_runtime["execution"]):
        return ("major", target_signature)
    if str(current_runtime["manifest_path"] or "").strip() != str(target_runtime["manifest_path"] or "").strip():
        return ("major", target_signature)
    if (
        str(current_runtime["profile_path"] or "").strip() != str(target_runtime["profile_path"] or "").strip()
        or _normalize_execution(current_runtime["execution"]) != _normalize_execution(target_runtime["execution"])
    ):
        return ("medium", target_signature)
    if [int(item) for item in current_runtime["gpu_ids"] or []] != [int(item) for item in target_runtime["gpu_ids"] or []]:
        return ("minor", target_signature)
    if str(current_runtime["device"] or "").strip().lower() != str(target_runtime["device"] or "").strip().lower():
        return ("minor", target_signature)
    return ("minor", target_signature)


def benchmark_matches_current_revision(record: BenchmarkRecord, deployment: DeploymentRecord) -> bool:
    if record.deployment_id != deployment.deployment_id:
        return False
    if record.deployment_revision is not None:
        return int(record.deployment_revision) == int(deployment.runtime_revision or 1)
    return _created_at_matches_current_revision(record.created_at, deployment)


def archive_matches_current_revision(record: PredictionArchiveRecord, deployment: DeploymentRecord) -> bool:
    if record.deployment_id != deployment.deployment_id:
        return False
    if record.deployment_revision is not None:
        return int(record.deployment_revision) == int(deployment.runtime_revision or 1)
    return _created_at_matches_current_revision(record.created_at, deployment)


def _created_at_matches_current_revision(created_at: str | None, deployment: DeploymentRecord) -> bool:
    revision_started_at = effective_revision_started_at(deployment)
    started = _parse_datetime(revision_started_at)
    created = _parse_datetime(created_at)
    return created >= started


def _runtime_descriptor_from_deployment(deployment: DeploymentRecord) -> dict[str, Any]:
    profile_runtime = deployment.profile_runtime
    device = deployment.device
    execution = dict(deployment.execution or {})
    if deployment.profile_path and Path(deployment.profile_path).exists():
        try:
            profile = load_profile(Path(deployment.profile_path))
            profile_runtime = profile.runtime
            device = profile.device
            execution = dict(profile.execution or {})
        except Exception:  # noqa: BLE001
            pass
    return {
        "manifest_path": deployment.manifest_path,
        "profile_path": deployment.profile_path,
        "use_gpu": deployment.use_gpu,
        "gpu_ids": list(deployment.gpu_ids or []),
        "provider_preference": deployment.provider_preference,
        "profile_runtime": profile_runtime,
        "device": device,
        "execution": execution,
    }


def _runtime_descriptor_from_target(
    *,
    deployment: DeploymentRecord,
    manifest_path: str | None,
    profile_name: str,
    profile_path: str | None,
    use_gpu: bool,
    gpu_ids: list[int] | None,
    provider_preference: str | None,
) -> dict[str, Any]:
    profile_runtime = deployment.profile_runtime
    device = deployment.device
    execution: dict[str, Any] = {}
    if profile_path and Path(profile_path).exists():
        profile = load_profile(Path(profile_path))
        profile_runtime = profile.runtime
        device = profile.device
        execution = dict(profile.execution or {})
        if not provider_preference:
            provider_preference = str(execution.get("provider_preference", "") or "").strip() or None
    return {
        "manifest_path": manifest_path,
        "profile_path": profile_path,
        "use_gpu": use_gpu,
        "gpu_ids": list(gpu_ids or []),
        "provider_preference": provider_preference,
        "profile_runtime": profile_runtime,
        "device": device,
        "execution": execution,
    }


def _normalize_execution(execution: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(execution or {})
    normalized = {
        "batch_size": payload.get("batch_size"),
        "precision": _normalize_precision(payload),
        "provider_preference": _normalize_provider(payload.get("provider_preference")),
        "graph_optimization_level": str(payload.get("graph_optimization_level", "") or "").strip().lower(),
        "intra_op_threads": payload.get("intra_op_threads"),
        "inter_op_threads": payload.get("inter_op_threads"),
        "session_config_entries": payload.get("session_config_entries") or {},
    }
    return normalized


def _normalize_precision(execution: dict[str, Any] | None) -> str:
    return str((execution or {}).get("precision", "") or "").strip().lower()


def _normalize_provider(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
