from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Iterable

from .schemas import BenchmarkRecord, DeploymentRecord, ModelSummary


def normalize_benchmark_scene(scene: str | None) -> str:
    return str(scene or "").strip().lower()


def is_industrial_dataset_name(dataset_name: str | None) -> bool:
    normalized = str(dataset_name or "").strip().lower()
    return any(
        keyword in normalized
        for keyword in (
            "industrial",
            "equipmentmix",
            "industrialequipmentmix",
            "mix15k",
            "mfgdominant",
        )
    )


def infer_benchmark_scene(
    *,
    model: ModelSummary | None = None,
    explicit_scene: str | None = None,
    language: str | None = None,
    request_text: str | None = None,
) -> str:
    normalized_scene = normalize_benchmark_scene(explicit_scene)
    if normalized_scene:
        return normalized_scene
    model_language = str(model.language if model is not None else (language or "")).strip().lower()
    if model_language == "en":
        return "en_general"
    if model is not None and is_industrial_dataset_name(model.dataset_name):
        return "zh_industrial"
    joined = str(request_text or "").lower()
    if any(
        keyword in joined
        for keyword in ("equipment", "machine", "fault", "device", "装备", "设备", "制造", "故障", "产线")
    ):
        return "zh_industrial"
    return "zh_general"


def default_benchmark_text(scene: str | None) -> str:
    normalized_scene = normalize_benchmark_scene(scene)
    if normalized_scene == "en_general":
        return (
            "NASA awarded SpaceX a contract and the company launched a Falcon 9 rocket from Florida."
        )
    if normalized_scene == "zh_industrial":
        return (
            "某制造企业在轴承外圈故障排查中发现振动异常，维修人员更换传感器后记录了设备、故障现象与处理措施。"
        )
    return "阿里巴巴总部位于杭州，马云曾担任该公司的重要管理者。"


def build_benchmark_scope_key(deployment_id: str, scene: str | None) -> str:
    normalized_scene = normalize_benchmark_scene(scene) or "generic"
    return f"{deployment_id}:{normalized_scene}"


def build_deployment_signature(deployment: DeploymentRecord) -> str:
    payload = {
        "deployment_id": deployment.deployment_id,
        "profile_name": deployment.profile_name,
        "runtime": deployment.runtime,
        "use_gpu": bool(deployment.use_gpu),
        "gpu_ids": [int(item) for item in deployment.gpu_ids or []],
        "device": deployment.device,
        "profile_runtime": deployment.profile_runtime,
        "execution": deployment.execution or {},
        "service_url": deployment.service_url,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def select_latest_effective_benchmarks(
    benchmarks: Iterable[BenchmarkRecord],
    *,
    allowed_statuses: set[str] | None = None,
) -> list[BenchmarkRecord]:
    statuses = set(allowed_statuses or {"completed"})
    grouped: dict[str, list[BenchmarkRecord]] = {}
    for item in benchmarks:
        if statuses and item.status not in statuses:
            continue
        scope_key = item.benchmark_scope_key or build_benchmark_scope_key(item.deployment_id, item.scene)
        grouped.setdefault(scope_key, []).append(item)

    selected: list[BenchmarkRecord] = []
    for items in grouped.values():
        effective_items = [item for item in items if item.is_effective]
        pool = effective_items or items
        selected.append(max(pool, key=_benchmark_order_key))
    selected.sort(key=_benchmark_order_key, reverse=True)
    return selected


def select_scene_relevant_benchmarks(
    benchmarks: Iterable[BenchmarkRecord],
    *,
    request_scene: str | None,
    allowed_statuses: set[str] | None = None,
) -> list[BenchmarkRecord]:
    effective = select_latest_effective_benchmarks(
        benchmarks,
        allowed_statuses=allowed_statuses or {"completed"},
    )
    normalized_scene = normalize_benchmark_scene(request_scene)
    if not normalized_scene:
        return effective

    exact = [item for item in effective if normalize_benchmark_scene(item.scene) == normalized_scene]
    if exact:
        return exact

    scene_free = [item for item in effective if not normalize_benchmark_scene(item.scene)]
    if scene_free:
        return scene_free

    scene_prefix = normalized_scene.split("_", maxsplit=1)[0]
    same_language = [
        item
        for item in effective
        if normalize_benchmark_scene(item.scene).startswith(scene_prefix + "_")
    ]
    if same_language:
        return same_language
    return effective


def _benchmark_order_key(record: BenchmarkRecord) -> tuple[datetime, int, str]:
    timestamp = _parse_datetime(record.created_at)
    return (
        timestamp,
        1 if record.is_effective else 0,
        str(record.benchmark_id or ""),
    )


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
