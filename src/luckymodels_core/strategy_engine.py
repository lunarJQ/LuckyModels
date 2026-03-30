from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any
from urllib.request import Request, urlopen

from .inference_activity_tracker import InferenceActivityTracker
from .schemas import (
    BenchmarkRecord,
    DeploymentPolicyRecommendation,
    DeploymentPolicyResponse,
    DeploymentPolicySummary,
    DeploymentRecord,
    ModelSummary,
    PredictionArchiveRecord,
    TaskRoutingCandidate,
    TaskRoutingExecution,
    TaskRoutingRequest,
    TaskRoutingResponse,
)


def run_task_routing_strategy(
    request: TaskRoutingRequest,
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    prediction_archives: list[PredictionArchiveRecord],
    activity_tracker: InferenceActivityTracker | None = None,
) -> TaskRoutingResponse:
    inferred_language = request.language or infer_language(request.text)
    difficulty = estimate_difficulty(request.text)
    candidates = _build_task_candidates(
        models=models,
        deployments=deployments,
        benchmarks=benchmarks,
        prediction_archives=prediction_archives,
        requested_language=inferred_language,
        candidate_model_ids=request.candidate_model_ids,
        objective=request.objective,
    )

    strategy_mode, selected_candidates = _select_task_candidates(
        candidates,
        objective=request.objective,
        difficulty=difficulty,
        max_models=max(1, min(int(request.max_models or 3), 5)),
    )

    execution_results: list[TaskRoutingExecution] = []
    for candidate in selected_candidates:
        result = _predict_with_runtime(
            candidate,
            text=request.text,
            scene=request.scene,
            options=request.options,
            activity_tracker=activity_tracker,
        )
        execution_results.append(result)

    fused_result = _fuse_execution_results(
        execution_results,
        objective=request.objective,
        inferred_language=inferred_language,
    )

    explanation = build_task_strategy_explanation(
        objective=request.objective,
        difficulty=difficulty,
        strategy_mode=strategy_mode,
        selected_candidates=selected_candidates,
    )

    return TaskRoutingResponse(
        objective=request.objective,
        requested_language=request.language,
        inferred_language=inferred_language,
        difficulty=difficulty,
        strategy_mode=strategy_mode,
        explanation=explanation,
        candidate_count=len(candidates),
        selected_count=len(selected_candidates),
        selected_deployments=selected_candidates,
        execution_results=execution_results,
        fused_result=fused_result,
    )


def build_deployment_policy(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    prediction_archives: list[PredictionArchiveRecord],
    gpu_snapshot: dict[str, Any],
) -> DeploymentPolicyResponse:
    recommendations: list[DeploymentPolicyRecommendation] = []
    benchmark_by_model = _group_completed_benchmarks_by_model(benchmarks)
    archives_by_model = _group_prediction_archives_by_model(prediction_archives)
    active_deployments_by_model: dict[str, list[DeploymentRecord]] = defaultdict(list)
    for item in deployments:
        if item.status in {"running", "starting"}:
            active_deployments_by_model[item.model_id].append(item)

    running_counts = Counter(item.model_id for item in deployments if item.status in {"running", "starting"})
    demand_scores: dict[str, float] = {}
    metrics_by_model: dict[str, dict[str, Any]] = {}
    recent_window_days = 7

    for model in models:
        archive_items = archives_by_model.get(model.model_id, [])
        benchmark_items = benchmark_by_model.get(model.model_id, [])
        total_request_count = len(archive_items)
        recent_request_count = _count_recent_prediction_archives(archive_items, days=recent_window_days)
        avg_p95 = _mean_numeric(item.p95_latency_ms for item in benchmark_items)
        avg_success = _mean_numeric(
            (
                item.success_count / item.total_requests
                if item.total_requests
                else None
            )
            for item in benchmark_items
        )
        running_count = running_counts.get(model.model_id, 0)
        package_size_mb = _directory_size_mb(model.package_root)
        estimated_cold_start_ms = _estimate_cold_start_ms(
            stage_count=model.stage_count,
            package_size_mb=package_size_mb,
            avg_p95_latency_ms=avg_p95,
        )
        cold_start_risk = _classify_cold_start_risk(estimated_cold_start_ms)
        demand_scores[model.model_id] = (
            recent_request_count * 7.0
            + total_request_count * 1.5
            + running_count * 4.0
            + len(benchmark_items) * 1.5
            + ((avg_success or 0.0) * 12.0)
            + (6.0 if avg_p95 is not None and avg_p95 < 80 else 0.0)
            + (2.5 if cold_start_risk == "high" and total_request_count > 0 else 0.0)
        )
        metrics_by_model[model.model_id] = {
            "archive_items": archive_items,
            "benchmark_items": benchmark_items,
            "total_request_count": total_request_count,
            "recent_request_count": recent_request_count,
            "avg_p95": avg_p95,
            "avg_success": avg_success,
            "running_count": running_count,
            "package_size_mb": package_size_mb,
            "estimated_cold_start_ms": estimated_cold_start_ms,
            "cold_start_risk": cold_start_risk,
            "active_deployment_ids": [
                item.deployment_id for item in active_deployments_by_model.get(model.model_id, [])
            ],
        }

    sorted_scores = sorted(demand_scores.values())
    high_bar = _quantile(sorted_scores, 0.67) if sorted_scores else 0.0
    medium_bar = _quantile(sorted_scores, 0.34) if sorted_scores else 0.0
    busiest_model_id = _pick_busiest_model(metrics_by_model, demand_scores)

    for model in models:
        metrics = metrics_by_model.get(model.model_id, {})
        benchmark_items = list(metrics.get("benchmark_items") or [])
        running_count = int(metrics.get("running_count") or 0)
        score = demand_scores.get(model.model_id, 0.0)
        recent_request_count = int(metrics.get("recent_request_count") or 0)
        total_request_count = int(metrics.get("total_request_count") or 0)
        avg_p95 = _mean_numeric(item.p95_latency_ms for item in benchmark_items)
        avg_success = metrics.get("avg_success")
        package_size_mb = metrics.get("package_size_mb")
        estimated_cold_start_ms = metrics.get("estimated_cold_start_ms")
        cold_start_risk = str(metrics.get("cold_start_risk") or "unknown")
        active_deployment_ids = list(metrics.get("active_deployment_ids") or [])

        service_tier = _classify_service_tier(
            demand_score=score,
            recent_request_count=recent_request_count,
            running_count=running_count,
            high_bar=high_bar,
            medium_bar=medium_bar,
        )
        demand_level = {"hot": "high", "warm": "medium", "cold": "low"}[service_tier]
        residency_mode = _select_residency_mode(
            service_tier=service_tier,
            cold_start_risk=cold_start_risk,
            recent_request_count=recent_request_count,
            total_request_count=total_request_count,
        )
        recommended_profile = _select_profile_name(
            service_tier=service_tier,
            cold_start_risk=cold_start_risk,
            avg_p95_latency_ms=avg_p95,
        )
        recommended_instance_count = _select_instance_count(
            service_tier=service_tier,
            recent_request_count=recent_request_count,
        )

        recommended_use_gpu, recommended_gpu_ids, device_label = _recommend_device_allocation(
            service_tier=service_tier,
            residency_mode=residency_mode,
            cold_start_risk=cold_start_risk,
            package_size_mb=package_size_mb,
            avg_p95_latency_ms=avg_p95,
            gpu_snapshot=gpu_snapshot,
        )
        gpu_pressure_action, fallback_profile_name, fallback_use_gpu, fallback_device_label = (
            _recommend_gpu_pressure_fallback(
                service_tier=service_tier,
                cold_start_risk=cold_start_risk,
                recommended_use_gpu=recommended_use_gpu,
            )
        )
        policy_tags = _build_policy_tags(
            service_tier=service_tier,
            residency_mode=residency_mode,
            cold_start_risk=cold_start_risk,
            recommended_use_gpu=recommended_use_gpu,
            gpu_pressure_action=gpu_pressure_action,
        )
        rationale = build_deployment_rationale(
            service_tier=service_tier,
            residency_mode=residency_mode,
            recent_request_count=recent_request_count,
            total_request_count=total_request_count,
            running_count=running_count,
            avg_p95_latency_ms=avg_p95,
            cold_start_risk=cold_start_risk,
            estimated_cold_start_ms=estimated_cold_start_ms,
            device_label=device_label,
            recommended_profile_name=recommended_profile,
            gpu_pressure_action=gpu_pressure_action,
        )
        recommendations.append(
            DeploymentPolicyRecommendation(
                model_id=model.model_id,
                model_name=model.model_name,
                language=model.language,
                demand_level=demand_level,
                demand_score=round(score, 3),
                service_tier=service_tier,
                residency_mode=residency_mode,
                recent_request_count=recent_request_count,
                total_request_count=total_request_count,
                benchmark_count=len(benchmark_items),
                avg_p95_latency_ms=avg_p95,
                benchmark_success_rate=avg_success,
                running_deployment_count=running_count,
                active_deployment_ids=active_deployment_ids,
                package_size_mb=package_size_mb,
                stage_count=model.stage_count,
                estimated_cold_start_ms=estimated_cold_start_ms,
                cold_start_risk=cold_start_risk,
                recommended_profile_name=recommended_profile,
                recommended_instance_count=recommended_instance_count,
                recommended_use_gpu=recommended_use_gpu,
                recommended_gpu_ids=recommended_gpu_ids,
                recommended_device_label=device_label,
                gpu_pressure_action=gpu_pressure_action,
                fallback_profile_name=fallback_profile_name,
                fallback_use_gpu=fallback_use_gpu,
                fallback_device_label=fallback_device_label,
                policy_tags=policy_tags,
                rationale=rationale,
            )
        )

    recommendations.sort(key=lambda item: (-item.demand_score, item.model_id))
    summary = _build_deployment_policy_summary(
        recommendations=recommendations,
        recent_window_days=recent_window_days,
        busiest_model_id=busiest_model_id,
    )
    return DeploymentPolicyResponse(
        gpu_available=bool(gpu_snapshot.get("available")),
        summary=summary,
        recommendations=recommendations,
        gpu_snapshot=gpu_snapshot,
    )


def infer_language(text: str) -> str:
    return "zh" if re.search(r"[\u3400-\u9fff]", text or "") else "en"


def estimate_difficulty(text: str) -> str:
    normalized = str(text or "").strip()
    length = len(normalized)
    clause_count = len(re.findall(r"[，。！？；,.!?;:]", normalized))
    if length <= 80 and clause_count <= 2:
        return "simple"
    if length <= 220 and clause_count <= 6:
        return "medium"
    return "hard"


def build_task_strategy_explanation(
    *,
    objective: str,
    difficulty: str,
    strategy_mode: str,
    selected_candidates: list[TaskRoutingCandidate],
) -> str:
    selected_models = "、".join(item.model_id for item in selected_candidates) or "无可用模型"
    if objective == "latency":
        return (
            f"当前文本被判定为{difficulty}任务，系统优先选择 P95 延迟更低的单模型部署，"
            f"本次使用 {selected_models} 以降低响应时间。"
        )
    if objective == "coverage":
        return (
            f"当前文本被判定为{difficulty}任务，系统采用 {strategy_mode}，"
            f"汇总 {selected_models} 的抽取结果并做并集融合，以提升三元组覆盖率。"
        )
    return (
        f"当前文本被判定为{difficulty}任务，系统采用 {strategy_mode}，"
        f"汇总 {selected_models} 的抽取结果并按出现频次投票，以提升结果稳定性。"
    )


def build_deployment_rationale(
    *,
    service_tier: str,
    residency_mode: str,
    recent_request_count: int,
    total_request_count: int,
    running_count: int,
    avg_p95_latency_ms: float | None,
    cold_start_risk: str,
    estimated_cold_start_ms: float | None,
    device_label: str,
    recommended_profile_name: str,
    gpu_pressure_action: str,
) -> str:
    latency_text = f"{avg_p95_latency_ms:.3f} ms" if avg_p95_latency_ms is not None else "暂无"
    cold_start_text = (
        f"{estimated_cold_start_ms:.0f} ms" if estimated_cold_start_ms is not None else "暂无"
    )
    residency_label = _residency_label(residency_mode)
    fallback_text = _gpu_pressure_action_label(gpu_pressure_action)
    if service_tier == "hot":
        return (
            f"该模型近 7 天请求 {recent_request_count} 次，总计 {total_request_count} 次，属于热模型；"
            f"历史 P95 为 {latency_text}。建议采用 {residency_label}，部署到 {device_label}，"
            f"优先使用 {recommended_profile_name} 模板以控制尾延迟。GPU 紧张时建议动作：{fallback_text}。"
        )
    if service_tier == "warm":
        return (
            f"该模型当前属于温模型，近 7 天请求 {recent_request_count} 次，"
            f"冷启动风险为 {cold_start_risk}（估计 {cold_start_text}）。"
            f"建议采用 {residency_label}，优先部署到 {device_label}，并使用 {recommended_profile_name} 模板。"
        )
    return (
        f"该模型当前属于冷模型，近 7 天请求 {recent_request_count} 次，当前运行实例 {running_count} 个，"
        f"历史 P95 为 {latency_text}，冷启动估计 {cold_start_text}。"
        f"建议采用 {residency_label}，默认落到 {device_label}；当 GPU 资源紧张时，优先 {fallback_text}。"
    )


def _build_task_candidates(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    prediction_archives: list[PredictionArchiveRecord],
    requested_language: str,
    candidate_model_ids: list[str],
    objective: str,
) -> list[TaskRoutingCandidate]:
    model_lookup = {item.model_id: item for item in models}
    completed_benchmarks = [item for item in benchmarks if item.status == "completed"]
    benchmark_by_deployment = defaultdict(list)
    benchmark_by_model = defaultdict(list)
    for item in completed_benchmarks:
        benchmark_by_deployment[item.deployment_id].append(item)
        benchmark_by_model[item.model_id].append(item)

    archives_by_deployment = defaultdict(list)
    for item in prediction_archives:
        archives_by_deployment[item.deployment_id].append(item)

    allowed_models = set(candidate_model_ids or [])
    candidates: list[TaskRoutingCandidate] = []
    for deployment in deployments:
        if deployment.status != "running" or not deployment.service_url:
            continue
        model = model_lookup.get(deployment.model_id)
        if model is None:
            continue
        if model.language != requested_language:
            continue
        if allowed_models and deployment.model_id not in allowed_models:
            continue

        deployment_benchmarks = benchmark_by_deployment.get(deployment.deployment_id) or benchmark_by_model.get(deployment.model_id, [])
        avg_p95 = _mean_numeric(item.p95_latency_ms for item in deployment_benchmarks)
        avg_success = _mean_numeric(
            (
                item.success_count / item.total_requests
                if item.total_requests
                else None
            )
            for item in deployment_benchmarks
        )
        recent_requests = len(archives_by_deployment.get(deployment.deployment_id, []))
        avg_triple_count = _mean_numeric(
            item.triple_count for item in archives_by_deployment.get(deployment.deployment_id, [])
        )
        score = _selection_score(
            objective=objective,
            success_rate=avg_success,
            p95_latency_ms=avg_p95,
            recent_request_count=recent_requests,
            avg_triple_count=avg_triple_count,
        )
        candidates.append(
            TaskRoutingCandidate(
                deployment_id=deployment.deployment_id,
                model_id=deployment.model_id,
                profile_name=deployment.profile_name,
                language=model.language,
                service_url=deployment.service_url,
                selection_score=round(score, 6),
                selection_reason=_selection_reason(
                    objective=objective,
                    success_rate=avg_success,
                    p95_latency_ms=avg_p95,
                    recent_request_count=recent_requests,
                ),
                estimated_p95_latency_ms=avg_p95,
                benchmark_success_rate=avg_success,
                recent_request_count=recent_requests,
                predicted_triple_count=int(round(avg_triple_count))
                if avg_triple_count is not None
                else None,
            )
        )

    reverse = objective != "latency"
    candidates.sort(
        key=lambda item: (
            item.selection_score if item.selection_score is not None else (-math.inf if reverse else math.inf),
            -item.recent_request_count,
            item.model_id,
        ),
        reverse=reverse,
    )
    return candidates


def _selection_score(
    *,
    objective: str,
    success_rate: float | None,
    p95_latency_ms: float | None,
    recent_request_count: int,
    avg_triple_count: float | None,
) -> float:
    if objective == "latency":
        return -float(p95_latency_ms if p95_latency_ms is not None else 10_000.0)
    if objective == "coverage":
        return (
            (success_rate or 0.0) * 40.0
            + min(recent_request_count, 10) * 2.5
            + min(avg_triple_count or 0.0, 20.0)
            - ((p95_latency_ms or 0.0) / 200.0)
        )
    return (
        (success_rate or 0.0) * 60.0
        + min(recent_request_count, 10) * 1.5
        - ((avg_triple_count or 0.0) / 8.0)
        - ((p95_latency_ms or 0.0) / 250.0)
    )


def _selection_reason(
    *,
    objective: str,
    success_rate: float | None,
    p95_latency_ms: float | None,
    recent_request_count: int,
) -> str:
    latency_text = f"{p95_latency_ms:.3f} ms" if p95_latency_ms is not None else "暂无 P95"
    success_text = f"{(success_rate or 0.0) * 100:.1f}%"
    if objective == "latency":
        return f"优先考虑低尾延迟，本候选历史 P95 为 {latency_text}。"
    if objective == "coverage":
        return f"优先考虑覆盖与稳定运行，本候选成功率 {success_text}，近期请求 {recent_request_count} 次。"
    return f"优先考虑稳定性，本候选成功率 {success_text}，历史 P95 为 {latency_text}。"


def _select_task_candidates(
    candidates: list[TaskRoutingCandidate],
    *,
    objective: str,
    difficulty: str,
    max_models: int,
) -> tuple[str, list[TaskRoutingCandidate]]:
    if not candidates:
        return ("none", [])
    if objective == "latency":
        return ("single-low-latency", candidates[:1])

    if objective == "coverage":
        target_count = 2 if difficulty == "simple" else min(max_models, 3)
        return ("multi-union", _unique_models(candidates, target_count))

    if difficulty == "simple":
        return ("single-best-accuracy", candidates[:1])

    target_count = 2 if difficulty == "medium" else min(max_models, 3)
    return ("multi-majority-vote", _unique_models(candidates, target_count))


def _unique_models(candidates: list[TaskRoutingCandidate], limit: int) -> list[TaskRoutingCandidate]:
    results: list[TaskRoutingCandidate] = []
    seen_models: set[str] = set()
    for candidate in candidates:
        if candidate.model_id in seen_models:
            continue
        seen_models.add(candidate.model_id)
        results.append(candidate)
        if len(results) >= limit:
            break
    if results:
        return results
    return candidates[:limit]


def _predict_with_runtime(
    candidate: TaskRoutingCandidate,
    *,
    text: str,
    scene: str | None,
    options: dict[str, Any],
    activity_tracker: InferenceActivityTracker | None = None,
) -> TaskRoutingExecution:
    payload = json.dumps(
        {
            "text": text,
            "scene": scene,
            "options": options or {},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        _resolve_runtime_url(candidate),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    activity_token: str | None = None
    if activity_tracker is not None:
        activity_token = activity_tracker.begin(
            model_id=candidate.model_id,
            deployment_id=candidate.deployment_id,
            source="task_routing",
        )
    try:
        with urlopen(request, timeout=90) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8"))
    finally:
        if activity_tracker is not None:
            activity_tracker.end(activity_token)
    return TaskRoutingExecution(
        deployment_id=candidate.deployment_id,
        model_id=candidate.model_id,
        profile_name=candidate.profile_name,
        selected_for_fusion=True,
        latency_ms=_safe_float(result.get("latency_ms")),
        triple_count=int(result.get("triple_count") or 0),
        triples=list(result.get("triples") or []),
        warnings=[str(item) for item in (result.get("warnings") or [])],
    )


def _resolve_runtime_url(candidate: TaskRoutingCandidate) -> str:
    if not candidate.service_url:
        raise RuntimeError("Missing runtime service URL for selected strategy candidate.")
    return str(candidate.service_url).rstrip("/") + "/predict"


def _fuse_execution_results(
    execution_results: list[TaskRoutingExecution],
    *,
    objective: str,
    inferred_language: str,
) -> dict[str, Any]:
    if not execution_results:
        return {
            "triple_count": 0,
            "triples": [],
            "fusion_mode": "empty",
        }

    if objective == "latency" or len(execution_results) == 1:
        primary = execution_results[0]
        return {
            "triple_count": primary.triple_count,
            "triples": primary.triples,
            "fusion_mode": "single",
        }

    triple_votes: dict[tuple[str, str, str], dict[str, Any]] = {}
    normalized_counter: Counter[tuple[str, str, str]] = Counter()
    for result in execution_results:
        for triple in result.triples:
            signature = _triple_signature(triple, inferred_language)
            normalized_counter[signature] += 1
            payload = triple_votes.get(signature)
            confidence = _safe_float(triple.get("confidence")) or 0.0
            if payload is None or confidence > (_safe_float(payload.get("confidence")) or 0.0):
                triple_votes[signature] = dict(triple)

    if objective == "coverage":
        triples = [
            {
                **triple_votes[signature],
                "vote_count": normalized_counter[signature],
            }
            for signature, _count in normalized_counter.most_common()
        ]
        return {
            "triple_count": len(triples),
            "triples": triples,
            "fusion_mode": "union",
        }

    threshold = (len(execution_results) // 2) + 1
    voted = [
        {
            **triple_votes[signature],
            "vote_count": count,
        }
        for signature, count in normalized_counter.most_common()
        if count >= threshold
    ]
    if not voted:
        voted = [
            {
                **triple_votes[signature],
                "vote_count": count,
            }
            for signature, count in normalized_counter.most_common()
        ][:5]
    return {
        "triple_count": len(voted),
        "triples": voted,
        "fusion_mode": "majority_vote",
        "vote_threshold": threshold,
    }


def _triple_signature(triple: dict[str, Any], language: str) -> tuple[str, str, str]:
    def normalize(text: Any) -> str:
        value = str(text or "").strip()
        if language == "zh":
            value = value.replace(" ", "")
        return value.lower()

    return (
        normalize(triple.get("subject")),
        normalize(triple.get("predicate")),
        normalize(triple.get("object")),
    )


def _group_completed_benchmarks_by_model(
    benchmarks: list[BenchmarkRecord],
) -> dict[str, list[BenchmarkRecord]]:
    grouped: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for item in benchmarks:
        if item.status == "completed":
            grouped[item.model_id].append(item)
    return grouped


def _group_prediction_archives_by_model(
    prediction_archives: list[PredictionArchiveRecord],
) -> dict[str, list[PredictionArchiveRecord]]:
    grouped: dict[str, list[PredictionArchiveRecord]] = defaultdict(list)
    for item in prediction_archives:
        grouped[item.model_id].append(item)
    return grouped


def _recommend_device_allocation(
    *,
    service_tier: str,
    residency_mode: str,
    cold_start_risk: str,
    package_size_mb: float | None,
    avg_p95_latency_ms: float | None,
    gpu_snapshot: dict[str, Any],
) -> tuple[bool, list[int], str]:
    devices = list(gpu_snapshot.get("devices") or [])
    if not gpu_snapshot.get("available") or not devices:
        return (False, [], "CPU")

    ranked = sorted(
        devices,
        key=lambda item: (
            -float((item.get("memory_total_mb") or 0) - (item.get("memory_used_mb") or 0)),
            float(item.get("utilization_gpu_pct") or 0),
            len(item.get("active_processes") or []),
        ),
    )
    if service_tier == "hot":
        selected = ranked[0]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（优先常驻）")
    if service_tier == "warm":
        if residency_mode == "on_demand" and cold_start_risk == "low":
            return (False, [], "CPU")
        selected = ranked[min(1, len(ranked) - 1)]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（共享保温）")
    if cold_start_risk == "high" and (package_size_mb or 0.0) >= 150:
        selected = ranked[-1]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（共享）")
    if avg_p95_latency_ms is not None and avg_p95_latency_ms > 250 and len(ranked) >= 2:
        selected = ranked[-1]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（低优先级共享）")
    return (False, [], "CPU")


def _count_recent_prediction_archives(
    prediction_archives: list[PredictionArchiveRecord],
    *,
    days: int,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count = 0
    for item in prediction_archives:
        timestamp = _parse_iso_datetime(item.created_at)
        if timestamp is not None and timestamp >= cutoff:
            count += 1
    return count


def _directory_size_mb(path: str | None) -> float | None:
    if not path or not os.path.isdir(path):
        return None
    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                total_bytes += os.path.getsize(file_path)
            except OSError:
                continue
    return round(total_bytes / (1024 * 1024), 3)


def _estimate_cold_start_ms(
    *,
    stage_count: int,
    package_size_mb: float | None,
    avg_p95_latency_ms: float | None,
) -> float:
    size_mb = float(package_size_mb or 0.0)
    base = 900.0 + max(stage_count, 1) * 550.0 + min(size_mb, 2048.0) * 2.5
    if avg_p95_latency_ms is not None and avg_p95_latency_ms > 120:
        base += 250.0
    return round(base, 3)


def _classify_cold_start_risk(estimated_cold_start_ms: float | None) -> str:
    if estimated_cold_start_ms is None:
        return "unknown"
    if estimated_cold_start_ms >= 3500:
        return "high"
    if estimated_cold_start_ms >= 1800:
        return "medium"
    return "low"


def _classify_service_tier(
    *,
    demand_score: float,
    recent_request_count: int,
    running_count: int,
    high_bar: float,
    medium_bar: float,
) -> str:
    if recent_request_count >= 20 or (demand_score >= high_bar and demand_score > 0):
        return "hot"
    if recent_request_count >= 3 or running_count > 0 or (demand_score >= medium_bar and demand_score > 0):
        return "warm"
    return "cold"


def _select_residency_mode(
    *,
    service_tier: str,
    cold_start_risk: str,
    recent_request_count: int,
    total_request_count: int,
) -> str:
    if service_tier == "hot":
        return "always_on"
    if service_tier == "warm":
        if cold_start_risk in {"medium", "high"} or recent_request_count >= 5:
            return "keep_warm"
        return "on_demand"
    if cold_start_risk == "high" and total_request_count > 0:
        return "keep_warm"
    return "on_demand"


def _select_profile_name(
    *,
    service_tier: str,
    cold_start_risk: str,
    avg_p95_latency_ms: float | None,
) -> str:
    if service_tier == "hot":
        return "latency"
    if service_tier == "warm":
        if avg_p95_latency_ms is not None and avg_p95_latency_ms > 180:
            return "latency"
        return "default"
    if cold_start_risk == "high":
        return "default"
    return "memory"


def _select_instance_count(
    *,
    service_tier: str,
    recent_request_count: int,
) -> int:
    if service_tier == "hot" and recent_request_count >= 40:
        return 2
    return 1


def _recommend_gpu_pressure_fallback(
    *,
    service_tier: str,
    cold_start_risk: str,
    recommended_use_gpu: bool,
) -> tuple[str, str, bool, str]:
    if service_tier == "hot":
        return ("retain_gpu", "latency", False, "CPU")
    if service_tier == "warm":
        if recommended_use_gpu and cold_start_risk != "high":
            return ("migrate_to_cpu", "default", False, "CPU")
        return ("share_gpu", "default", False, "CPU")
    return ("migrate_to_cpu", "memory", False, "CPU")


def _build_policy_tags(
    *,
    service_tier: str,
    residency_mode: str,
    cold_start_risk: str,
    recommended_use_gpu: bool,
    gpu_pressure_action: str,
) -> list[str]:
    tags = [
        {"hot": "热模型", "warm": "温模型", "cold": "冷模型"}[service_tier],
        _residency_label(residency_mode),
        f"冷启动{cold_start_risk}",
    ]
    tags.append("GPU 优先" if recommended_use_gpu else "CPU 优先")
    if gpu_pressure_action == "migrate_to_cpu":
        tags.append("GPU 紧张转 CPU")
    elif gpu_pressure_action == "share_gpu":
        tags.append("GPU 紧张转共享")
    else:
        tags.append("GPU 保留")
    return tags


def _pick_busiest_model(
    metrics_by_model: dict[str, dict[str, Any]],
    demand_scores: dict[str, float],
) -> str | None:
    ranked = sorted(
        metrics_by_model.items(),
        key=lambda item: (
            -int(item[1].get("recent_request_count") or 0),
            -int(item[1].get("total_request_count") or 0),
            -float(demand_scores.get(item[0], 0.0)),
            item[0],
        ),
    )
    return ranked[0][0] if ranked else None


def _build_deployment_policy_summary(
    *,
    recommendations: list[DeploymentPolicyRecommendation],
    recent_window_days: int,
    busiest_model_id: str | None,
) -> DeploymentPolicySummary:
    hot_count = sum(1 for item in recommendations if item.service_tier == "hot")
    warm_count = sum(1 for item in recommendations if item.service_tier == "warm")
    cold_count = sum(1 for item in recommendations if item.service_tier == "cold")
    always_on_count = sum(1 for item in recommendations if item.residency_mode == "always_on")
    keep_warm_count = sum(1 for item in recommendations if item.residency_mode == "keep_warm")
    on_demand_count = sum(1 for item in recommendations if item.residency_mode == "on_demand")
    gpu_recommended_count = sum(1 for item in recommendations if item.recommended_use_gpu)
    cpu_recommended_count = len(recommendations) - gpu_recommended_count
    cpu_fallback_count = sum(1 for item in recommendations if item.gpu_pressure_action == "migrate_to_cpu")
    strategy_overview = (
        f"平台当前共分析 {len(recommendations)} 个模型，其中热模型 {hot_count} 个、温模型 {warm_count} 个、冷模型 {cold_count} 个。"
        f"建议常驻 {always_on_count} 个、保温 {keep_warm_count} 个、按需启动 {on_demand_count} 个；"
        f"GPU 紧张时有 {cpu_fallback_count} 个低优先级模型可转为 CPU 兜底部署。"
    )
    return DeploymentPolicySummary(
        total_models=len(recommendations),
        recent_window_days=recent_window_days,
        hot_count=hot_count,
        warm_count=warm_count,
        cold_count=cold_count,
        always_on_count=always_on_count,
        keep_warm_count=keep_warm_count,
        on_demand_count=on_demand_count,
        gpu_recommended_count=gpu_recommended_count,
        cpu_recommended_count=cpu_recommended_count,
        cpu_fallback_count=cpu_fallback_count,
        busiest_model_id=busiest_model_id,
        strategy_overview=strategy_overview,
    )


def _residency_label(value: str) -> str:
    return {
        "always_on": "常驻",
        "keep_warm": "保温",
        "on_demand": "按需启动",
    }.get(value, value)


def _gpu_pressure_action_label(value: str) -> str:
    return {
        "retain_gpu": "保留 GPU",
        "share_gpu": "转为共享 GPU",
        "migrate_to_cpu": "转为 CPU 兜底部署",
    }.get(value, value)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _mean_numeric(values) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round(mean(numeric_values), 6)


def _quantile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[index]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
