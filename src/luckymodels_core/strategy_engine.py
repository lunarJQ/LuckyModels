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
from urllib.error import HTTPError, URLError

from .inference_activity_tracker import InferenceActivityTracker
from .schemas import (
    BenchmarkRecord,
    DeploymentPolicyAutomationAction,
    DeploymentPolicyRecommendation,
    DeploymentPolicyResponse,
    DeploymentPolicySummary,
    DeploymentRecord,
    ModelSummary,
    PredictionArchiveRecord,
    TaskRoutingAdaptiveWeightProfile,
    TaskRoutingCandidate,
    TaskRoutingDecisionRecord,
    TaskRoutingDecisionSummary,
    TaskRoutingExecutionPathSummary,
    TaskRoutingExecution,
    TaskRoutingFeedbackOverview,
    TaskRoutingFeedbackRecord,
    TaskRoutingModelAdoptionSummary,
    TaskRoutingModelScoreSummary,
    TaskRoutingObjectiveFeedbackSummary,
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
    decision_records: list[TaskRoutingDecisionRecord],
    feedback_records: list[TaskRoutingFeedbackRecord],
    activity_tracker: InferenceActivityTracker | None = None,
) -> TaskRoutingResponse:
    inferred_language = request.language or infer_language(request.text)
    difficulty = estimate_difficulty(request.text)
    context_features = analyze_request_context(
        text=request.text,
        scene=request.scene,
        inferred_language=inferred_language,
    )
    base_objective_weights = _build_objective_weights(
        objective=request.objective,
        difficulty=difficulty,
        context_features=context_features,
    )
    benchmark_adjustments = _build_benchmark_weight_adjustments(
        models=models,
        deployments=deployments,
        benchmarks=benchmarks,
        requested_language=inferred_language,
        context_features=context_features,
    )
    archive_adjustments, adaptation_reasons, _archive_unsupervised_adjustments, _archive_feedback_adjustments = _build_archive_weight_adjustments(
        models=models,
        deployments=deployments,
        prediction_archives=prediction_archives,
        context_features=context_features,
        requested_language=inferred_language,
        objective=request.objective,
    )
    decision_feedback_adjustments, decision_feedback_reasons = _build_decision_feedback_weight_adjustments(
        decision_records=decision_records,
        feedback_records=feedback_records,
        objective=request.objective,
        requested_language=inferred_language,
        context_features=context_features,
        base_weights=base_objective_weights,
    )
    objective_weights = _apply_adaptive_weight_adjustments(
        base_weights=base_objective_weights,
        benchmark_adjustments=benchmark_adjustments,
        archive_adjustments=archive_adjustments,
        decision_feedback_adjustments=decision_feedback_adjustments,
    )
    adaptation_reasons = _merge_adaptation_reasons(
        benchmark_adjustments=benchmark_adjustments,
        archive_adjustments=archive_adjustments,
        decision_feedback_adjustments=decision_feedback_adjustments,
        archive_reasons=adaptation_reasons,
        decision_feedback_reasons=decision_feedback_reasons,
    )
    candidates = _build_task_candidates(
        models=models,
        deployments=deployments,
        benchmarks=benchmarks,
        prediction_archives=prediction_archives,
        requested_language=inferred_language,
        candidate_model_ids=request.candidate_model_ids,
        objective=request.objective,
        context_features=context_features,
        objective_weights=objective_weights,
    )

    strategy_mode, execution_path, fusion_policy, selected_candidates = _select_task_candidates(
        candidates,
        objective=request.objective,
        difficulty=difficulty,
        context_features=context_features,
        max_models=max(1, min(int(request.max_models or 3), 5)),
    )

    execution_results: list[TaskRoutingExecution] = []
    execution_roles = _resolve_execution_roles(execution_path, len(selected_candidates))
    for candidate, execution_role in zip(selected_candidates, execution_roles, strict=False):
        result = _predict_with_runtime(
            candidate,
            text=request.text,
            scene=request.scene,
            options=request.options,
            activity_tracker=activity_tracker,
            execution_role=execution_role,
        )
        execution_results.append(result)

    fused_result = _fuse_execution_results(
        execution_results,
        objective=request.objective,
        inferred_language=inferred_language,
        fusion_policy=fusion_policy,
    )

    explanation = build_task_strategy_explanation(
        objective=request.objective,
        difficulty=difficulty,
        strategy_mode=strategy_mode,
        execution_path=execution_path,
        fusion_policy=fusion_policy,
        context_features=context_features,
        selected_candidates=selected_candidates,
    )
    decision_trace = _build_task_decision_trace(
        objective=request.objective,
        difficulty=difficulty,
        context_features=context_features,
        objective_weights=objective_weights,
        selected_candidates=selected_candidates,
        execution_path=execution_path,
        fusion_policy=fusion_policy,
    )

    return TaskRoutingResponse(
        objective=request.objective,
        requested_language=request.language,
        inferred_language=inferred_language,
        difficulty=difficulty,
        strategy_mode=strategy_mode,
        execution_path=execution_path,
        fusion_policy=fusion_policy,
        explanation=explanation,
        context_features=context_features,
        objective_weights=objective_weights,
        adaptive_adjustments=_sum_weight_adjustments(
            benchmark_adjustments,
            archive_adjustments,
            decision_feedback_adjustments,
        ),
        adaptation_reasons=adaptation_reasons,
        decision_trace=decision_trace,
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
    decision_records: list[TaskRoutingDecisionRecord] | None = None,
    feedback_records: list[TaskRoutingFeedbackRecord] | None = None,
    automation_actions: list[DeploymentPolicyAutomationAction] | None = None,
) -> DeploymentPolicyResponse:
    recommendations: list[DeploymentPolicyRecommendation] = []
    decision_records = list(decision_records or [])
    feedback_records = list(feedback_records or [])
    automation_actions = list(automation_actions or [])
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

    routing_signals = _build_task_routing_model_signal_map(
        models=models,
        deployments=deployments,
        benchmarks=benchmarks,
        prediction_archives=prediction_archives,
        decision_records=decision_records,
        feedback_records=feedback_records,
    )
    automation_signals = _build_automation_history_signal_map(automation_actions)
    collaborative_scores: dict[str, float] = {}
    for model in models:
        model_id = model.model_id
        base_demand_score = demand_scores.get(model_id, 0.0)
        routing_value_score = float((routing_signals.get(model_id) or {}).get("routing_value_score") or 0.0)
        operational_risk_score = float((automation_signals.get(model_id) or {}).get("operational_risk_score") or 0.0)
        collaborative_scores[model_id] = round(
            max(0.0, base_demand_score + routing_value_score - operational_risk_score * 0.65),
            6,
        )

    sorted_scores = sorted(collaborative_scores.values())
    high_bar = _quantile(sorted_scores, 0.67) if sorted_scores else 0.0
    medium_bar = _quantile(sorted_scores, 0.34) if sorted_scores else 0.0
    busiest_model_id = _pick_busiest_model(metrics_by_model, collaborative_scores)

    for model in models:
        model_id = model.model_id
        metrics = metrics_by_model.get(model.model_id, {})
        routing_signal = routing_signals.get(model_id, {})
        automation_signal = automation_signals.get(model_id, {})
        benchmark_items = list(metrics.get("benchmark_items") or [])
        running_count = int(metrics.get("running_count") or 0)
        base_demand_score = demand_scores.get(model_id, 0.0)
        collaborative_priority_score = collaborative_scores.get(model_id, 0.0)
        routing_value_score = float(routing_signal.get("routing_value_score") or 0.0)
        operational_risk_score = float(automation_signal.get("operational_risk_score") or 0.0)
        recent_request_count = int(metrics.get("recent_request_count") or 0)
        total_request_count = int(metrics.get("total_request_count") or 0)
        avg_p95 = _mean_numeric(item.p95_latency_ms for item in benchmark_items)
        avg_success = metrics.get("avg_success")
        package_size_mb = metrics.get("package_size_mb")
        estimated_cold_start_ms = metrics.get("estimated_cold_start_ms")
        cold_start_risk = str(metrics.get("cold_start_risk") or "unknown")
        active_deployment_ids = list(metrics.get("active_deployment_ids") or [])

        service_tier = _classify_service_tier(
            demand_score=collaborative_priority_score,
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
            model_id=model.model_id,
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
        (
            service_tier,
            demand_level,
            residency_mode,
            recommended_profile,
            recommended_instance_count,
            recommended_use_gpu,
            recommended_gpu_ids,
            device_label,
            gpu_pressure_action,
            fallback_profile_name,
            fallback_use_gpu,
            fallback_device_label,
            cross_layer_tags,
            coordination_note,
        ) = _apply_cross_layer_policy_adjustments(
            model=model,
            service_tier=service_tier,
            demand_level=demand_level,
            residency_mode=residency_mode,
            recommended_profile_name=recommended_profile,
            recommended_instance_count=recommended_instance_count,
            recommended_use_gpu=recommended_use_gpu,
            recommended_gpu_ids=recommended_gpu_ids,
            recommended_device_label=device_label,
            gpu_pressure_action=gpu_pressure_action,
            fallback_profile_name=fallback_profile_name,
            fallback_use_gpu=fallback_use_gpu,
            fallback_device_label=fallback_device_label,
            cold_start_risk=cold_start_risk,
            gpu_snapshot=gpu_snapshot,
            routing_signal=routing_signal,
            automation_signal=automation_signal,
            collaborative_priority_score=collaborative_priority_score,
        )
        policy_tags = _build_policy_tags(
            service_tier=service_tier,
            residency_mode=residency_mode,
            cold_start_risk=cold_start_risk,
            recommended_use_gpu=recommended_use_gpu,
            gpu_pressure_action=gpu_pressure_action,
        )
        policy_tags.extend(tag for tag in cross_layer_tags if tag and tag not in policy_tags)
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
                demand_score=round(collaborative_priority_score, 3),
                base_demand_score=round(base_demand_score, 3),
                routing_value_score=round(routing_value_score, 3),
                operational_risk_score=round(operational_risk_score, 3),
                collaborative_priority_score=round(collaborative_priority_score, 3),
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
                routing_selected_count=int(routing_signal.get("selected_count") or 0),
                routing_feedback_count=int(routing_signal.get("feedback_count") or 0),
                routing_adopted_count=int(routing_signal.get("adopted_count") or 0),
                routing_acceptance_rate=_safe_float(routing_signal.get("acceptance_rate")),
                avg_routing_current_score=_safe_float(routing_signal.get("avg_current_score")),
                routing_dimension_signals={
                    key: round(float(value), 6)
                    for key, value in dict(routing_signal.get("dimension_signals") or {}).items()
                    if abs(float(value)) > 0.000001
                },
                routing_focus_dimensions=list(routing_signal.get("focus_dimensions") or []),
                restart_history_count=int(automation_signal.get("restart_count") or 0),
                rebalance_history_count=int(automation_signal.get("rebalance_count") or 0),
                cpu_migration_history_count=int(automation_signal.get("cpu_migration_count") or 0),
                failed_action_history_count=int(automation_signal.get("failed_count") or 0),
                policy_tags=policy_tags,
                coordination_note=coordination_note,
                rationale=rationale,
            )
        )

    recommendations.sort(key=lambda item: (-item.collaborative_priority_score, -item.base_demand_score, item.model_id))
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


def analyze_request_context(
    *,
    text: str,
    scene: str | None,
    inferred_language: str,
) -> dict[str, Any]:
    normalized = str(text or "").strip()
    clause_count = len(re.findall(r"[，。！？；,.!?;:]", normalized))
    tokens = [item for item in re.split(r"\s+", normalized) if item]
    capitalized_tokens = len(re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", normalized))
    quoted_entities = len(re.findall(r"[\"“”'‘’][^\"“”'‘’]{1,40}[\"“”'‘’]", normalized))
    numeric_mentions = len(re.findall(r"\d+(?:\.\d+)?", normalized))
    entity_hint_count = capitalized_tokens + quoted_entities + numeric_mentions
    domain = infer_domain(text=normalized, scene=scene)
    return {
        "text_length": len(normalized),
        "token_count": len(tokens),
        "clause_count": clause_count,
        "entity_hint_count": entity_hint_count,
        "numeric_mentions": numeric_mentions,
        "domain": domain,
        "scene": scene or "",
        "language": inferred_language,
    }


def infer_domain(*, text: str, scene: str | None) -> str:
    normalized_scene = _normalize_scene(scene)
    if normalized_scene == "zh_industrial":
        return "industrial"
    if normalized_scene in {"zh_general", "en_general"}:
        return "general"

    joined = f"{scene or ''} {text or ''}".lower()
    industrial_keywords = [
        "equipment", "machine", "fault", "device", "manufacturing", "sensor",
        "装备", "设备", "制造", "故障", "部件", "工厂", "产线",
    ]
    scientific_keywords = [
        "dataset", "algorithm", "method", "task", "paper", "model", "science",
        "数据集", "算法", "方法", "模型", "论文", "实验", "研究",
    ]
    if any(keyword in joined for keyword in industrial_keywords):
        return "industrial"
    if any(keyword in joined for keyword in scientific_keywords):
        return "scientific"
    return "general"


def _normalize_scene(scene: str | None) -> str:
    return str(scene or "").strip().lower()


def _is_industrial_dataset_name(dataset_name: str | None) -> bool:
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
    ) or infer_domain(text=str(dataset_name or ""), scene=dataset_name) == "industrial"


def _scene_matches_model(scene: str | None, model: ModelSummary) -> bool:
    normalized_scene = _normalize_scene(scene)
    if not normalized_scene:
        return True
    if normalized_scene == "en_general":
        return model.language == "en"
    if normalized_scene == "zh_general":
        return model.language == "zh" and not _is_industrial_dataset_name(model.dataset_name)
    if normalized_scene == "zh_industrial":
        return model.language == "zh"
    return True


def _build_objective_weights(
    *,
    objective: str,
    difficulty: str,
    context_features: dict[str, Any],
) -> dict[str, float]:
    weights = {
        "accuracy": 0.28,
        "latency": 0.18,
        "coverage": 0.18,
        "stability": 0.14,
        "resource": 0.10,
        "domain": 0.12,
    }
    presets = {
        "accuracy": {"accuracy": 0.38, "stability": 0.22, "coverage": 0.14, "latency": 0.10, "resource": 0.04, "domain": 0.12},
        "coverage": {"accuracy": 0.18, "stability": 0.10, "coverage": 0.34, "latency": 0.08, "resource": 0.06, "domain": 0.24},
        "latency": {"accuracy": 0.12, "stability": 0.12, "coverage": 0.08, "latency": 0.48, "resource": 0.14, "domain": 0.06},
        "balanced": weights,
        "resource": {"accuracy": 0.14, "stability": 0.12, "coverage": 0.08, "latency": 0.18, "resource": 0.38, "domain": 0.10},
        "domain": {"accuracy": 0.20, "stability": 0.10, "coverage": 0.12, "latency": 0.08, "resource": 0.06, "domain": 0.44},
    }
    weights = dict(presets.get(objective, weights))
    if difficulty == "hard":
        weights["accuracy"] += 0.04
        weights["coverage"] += 0.03
        weights["latency"] -= 0.03
        weights["resource"] -= 0.02
    if context_features.get("domain") in {"industrial", "scientific"}:
        weights["domain"] += 0.04
        weights["latency"] -= 0.02
        weights["resource"] -= 0.02
    total = sum(max(value, 0.0) for value in weights.values()) or 1.0
    return {key: round(max(value, 0.0) / total, 4) for key, value in weights.items()}


def _zero_weight_vector() -> dict[str, float]:
    return {
        "accuracy": 0.0,
        "latency": 0.0,
        "coverage": 0.0,
        "stability": 0.0,
        "resource": 0.0,
        "domain": 0.0,
    }


def _progressive_evidence_scale(
    evidence_count: int,
    *,
    target_count: int = 30,
) -> float:
    if evidence_count <= 0:
        return 0.0
    if target_count <= 1:
        return 1.0
    return round(min(1.0, evidence_count / target_count), 6)


def _objective_feedback_gain(objective: str, key: str) -> float:
    normalized_objective = str(objective or "balanced").strip().lower()
    gain_table = {
        "accuracy": {
            "accuracy": 1.3,
            "stability": 1.2,
            "coverage": 0.9,
            "latency": 0.85,
            "resource": 0.85,
            "domain": 1.0,
        },
        "coverage": {
            "accuracy": 0.95,
            "stability": 1.0,
            "coverage": 1.35,
            "latency": 0.85,
            "resource": 0.85,
            "domain": 1.15,
        },
        "latency": {
            "accuracy": 0.85,
            "stability": 0.95,
            "coverage": 0.8,
            "latency": 1.4,
            "resource": 1.2,
            "domain": 0.85,
        },
        "balanced": {
            "accuracy": 1.0,
            "stability": 1.0,
            "coverage": 1.0,
            "latency": 1.0,
            "resource": 1.0,
            "domain": 1.0,
        },
        "resource": {
            "accuracy": 0.9,
            "stability": 0.95,
            "coverage": 0.85,
            "latency": 1.1,
            "resource": 1.35,
            "domain": 0.9,
        },
        "domain": {
            "accuracy": 1.05,
            "stability": 1.0,
            "coverage": 1.15,
            "latency": 0.85,
            "resource": 0.85,
            "domain": 1.4,
        },
    }
    return gain_table.get(normalized_objective, gain_table["balanced"]).get(key, 1.0)


def _build_benchmark_weight_adjustments(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    requested_language: str,
    context_features: dict[str, Any],
) -> dict[str, float]:
    model_lookup = {item.model_id: item for item in models}
    deployment_lookup = {item.deployment_id: item for item in deployments}
    request_domain = str(context_features.get("domain") or "general")
    request_scene = context_features.get("scene")
    relevant: list[BenchmarkRecord] = []
    for benchmark in benchmarks:
        if benchmark.status != "completed":
            continue
        model = model_lookup.get(benchmark.model_id)
        if model is None or (requested_language and model.language != requested_language):
            continue
        if not _scene_matches_model(request_scene, model):
            continue
        if request_domain != "general":
            candidate_domain = infer_domain(text=str(model.dataset_name or ""), scene=model.dataset_name)
            if candidate_domain not in {request_domain, "general"}:
                continue
        relevant.append(benchmark)

    adjustments = _zero_weight_vector()
    if not relevant:
        adjustments["stability"] += 0.03
        adjustments["accuracy"] += 0.02
        return adjustments

    avg_success = _mean_numeric(
        item.success_count / item.total_requests if item.total_requests else None
        for item in relevant
    )
    avg_p95 = _mean_numeric(item.p95_latency_ms for item in relevant)
    gpu_ratio = 0.0
    if relevant:
        gpu_hits = 0
        for item in relevant:
            deployment = deployment_lookup.get(item.deployment_id)
            if deployment and deployment.use_gpu:
                gpu_hits += 1
        gpu_ratio = gpu_hits / len(relevant)

    if avg_success is None or avg_success < 0.82:
        adjustments["accuracy"] += 0.04
        adjustments["stability"] += 0.03
        adjustments["coverage"] -= 0.01
    if avg_p95 is not None and avg_p95 > 220:
        adjustments["latency"] += 0.05
        adjustments["resource"] += 0.02
    elif avg_p95 is not None and avg_p95 < 90 and (avg_success or 0.0) >= 0.9:
        adjustments["coverage"] += 0.02
    if len(relevant) < 3:
        adjustments["stability"] += 0.02
    if gpu_ratio < 0.3:
        adjustments["resource"] += 0.01
    return adjustments


def _build_archive_weight_adjustments(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    prediction_archives: list[PredictionArchiveRecord],
    context_features: dict[str, Any],
    requested_language: str,
    objective: str = "balanced",
) -> tuple[dict[str, float], list[str], dict[str, float], dict[str, float]]:
    model_lookup = {item.model_id: item for item in models}
    deployment_lookup = {item.deployment_id: item for item in deployments}
    request_domain = str(context_features.get("domain") or "general")
    request_scene = str(context_features.get("scene") or "").strip().lower()
    matched_archives: list[PredictionArchiveRecord] = []

    for archive in prediction_archives:
        model = model_lookup.get(archive.model_id)
        if model is None:
            continue
        if not _archive_matches_request_context(
            archive=archive,
            model=model,
            requested_language=requested_language,
            request_scene=request_scene,
            request_domain=request_domain,
        ):
            continue
        matched_archives.append(archive)

    unsupervised_adjustments = _zero_weight_vector()
    feedback_adjustments = _zero_weight_vector()
    reasons: list[str] = []
    if not matched_archives:
        unsupervised_adjustments["accuracy"] += 0.02
        unsupervised_adjustments["stability"] += 0.03
        reasons.append("当前缺少可直接复用的在线归档证据，系统临时提升准确与稳定权重。")
        return _sum_weight_adjustments(unsupervised_adjustments, feedback_adjustments), reasons, unsupervised_adjustments, feedback_adjustments

    archive_count = len(matched_archives)
    empty_count = sum(1 for item in matched_archives if not int(item.triple_count or 0))
    empty_rate = empty_count / archive_count
    avg_latency = _mean_numeric(item.latency_ms for item in matched_archives)
    avg_triple_count = _mean_numeric(item.triple_count for item in matched_archives)
    satisfied_count = sum(1 for item in matched_archives if str(item.feedback_status or "pending").strip().lower() == "satisfied")
    unsatisfied_count = sum(1 for item in matched_archives if str(item.feedback_status or "pending").strip().lower() == "unsatisfied")
    effective_feedback_count = satisfied_count + unsatisfied_count
    missing_too_many_count = sum(
        1
        for item in matched_archives
        if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
        and "missing_too_many" in _parse_archive_feedback_reasons(item.feedback_note)
    )
    false_positive_too_many_count = sum(
        1
        for item in matched_archives
        if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
        and "false_positive_too_many" in _parse_archive_feedback_reasons(item.feedback_note)
    )
    latency_high_count = sum(
        1
        for item in matched_archives
        if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
        and "latency_high" in _parse_archive_feedback_reasons(item.feedback_note)
    )

    archive_scale = _progressive_evidence_scale(archive_count, target_count=30)
    feedback_scale = _progressive_evidence_scale(effective_feedback_count, target_count=10)

    if empty_rate >= 0.45:
        unsupervised_adjustments["accuracy"] += (0.02 + 0.03 * empty_rate) * max(archive_scale, 0.25)
        unsupervised_adjustments["stability"] += (0.02 + 0.02 * empty_rate) * max(archive_scale, 0.25)
        unsupervised_adjustments["coverage"] -= (0.008 + 0.012 * empty_rate) * max(archive_scale, 0.25)
        reasons.append(f"近期在线归档空结果率约为 {empty_rate * 100:.1f}%，系统提升准确与稳定权重。")
    elif empty_rate <= 0.18 and (avg_triple_count or 0.0) >= 2.5:
        unsupervised_adjustments["coverage"] += (0.01 + 0.03 * min(1.0, (avg_triple_count or 0.0) / 6.0)) * max(archive_scale, 0.2)
        unsupervised_adjustments["domain"] += 0.015 * max(archive_scale, 0.2)
        reasons.append("近期在线归档结果较稳定且三元组非空率较高，系统适度提升覆盖与领域权重。")

    if avg_latency is not None and avg_latency > 180:
        unsupervised_adjustments["latency"] += (0.015 + 0.02 * min(1.0, avg_latency / 400.0)) * max(archive_scale, 0.2)
        unsupervised_adjustments["resource"] += 0.01 * max(archive_scale, 0.2)
        reasons.append(f"近期在线归档平均时延约为 {avg_latency:.1f} ms，系统适度提升时延与资源权重。")
    elif avg_latency is not None and avg_latency < 90 and empty_rate <= 0.25:
        unsupervised_adjustments["coverage"] += 0.008 * max(archive_scale, 0.2)

    if effective_feedback_count >= 1:
        satisfied_rate = satisfied_count / effective_feedback_count
        if satisfied_rate >= 0.7:
            feedback_adjustments["accuracy"] += (0.008 + 0.018 * satisfied_rate) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "accuracy")
            feedback_adjustments["coverage"] += (0.006 + 0.012 * satisfied_rate) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "coverage")
            reasons.append(
                f"近期同类在线归档收到 {satisfied_count} 条满意反馈、{unsatisfied_count} 条不满意反馈，系统提升准确与覆盖权重。"
            )
        elif satisfied_rate <= 0.4:
            feedback_adjustments["accuracy"] += (0.012 + 0.02 * (1.0 - satisfied_rate)) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "accuracy")
            feedback_adjustments["stability"] += (0.01 + 0.018 * (1.0 - satisfied_rate)) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "stability")
            feedback_adjustments["coverage"] -= (0.004 + 0.01 * (1.0 - satisfied_rate)) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "coverage")
            reasons.append(
                f"近期同类在线归档中不满意反馈占比较高（{unsatisfied_count}/{effective_feedback_count}），系统提升准确与稳定权重。"
            )

    if unsatisfied_count >= 1 and missing_too_many_count:
        missing_ratio = missing_too_many_count / unsatisfied_count
        signal_scale = max(feedback_scale, 0.1)
        feedback_adjustments["coverage"] += (0.01 + 0.025 * missing_ratio) * signal_scale * _objective_feedback_gain(objective, "coverage")
        feedback_adjustments["domain"] += (0.004 + 0.01 * missing_ratio) * signal_scale * _objective_feedback_gain(objective, "domain")
        reasons.append(
            f"不满意反馈中有 {missing_too_many_count} 条指出“漏抽多”，系统提升覆盖与领域权重。"
        )

    if unsatisfied_count >= 1 and false_positive_too_many_count:
        false_positive_ratio = false_positive_too_many_count / unsatisfied_count
        signal_scale = max(feedback_scale, 0.1)
        feedback_adjustments["accuracy"] += (0.01 + 0.022 * false_positive_ratio) * signal_scale * _objective_feedback_gain(objective, "accuracy")
        feedback_adjustments["stability"] += (0.008 + 0.018 * false_positive_ratio) * signal_scale * _objective_feedback_gain(objective, "stability")
        feedback_adjustments["coverage"] -= 0.006 * false_positive_ratio * signal_scale * _objective_feedback_gain(objective, "coverage")
        reasons.append(
            f"不满意反馈中有 {false_positive_too_many_count} 条指出“误抽多”，系统提升准确与稳定权重。"
        )

    if unsatisfied_count >= 1 and latency_high_count:
        latency_high_ratio = latency_high_count / unsatisfied_count
        signal_scale = max(feedback_scale, 0.1)
        feedback_adjustments["latency"] += (0.01 + 0.022 * latency_high_ratio) * signal_scale * _objective_feedback_gain(objective, "latency")
        feedback_adjustments["resource"] += (0.004 + 0.012 * latency_high_ratio) * signal_scale * _objective_feedback_gain(objective, "resource")
        reasons.append(
            f"不满意反馈中有 {latency_high_count} 条指出“延迟高”，系统提升时延与资源权重。"
        )

    if request_domain != "general" and unsatisfied_count >= 1 and (missing_too_many_count or false_positive_too_many_count):
        domain_mismatch_ratio = min(1.0, (missing_too_many_count + false_positive_too_many_count) / unsatisfied_count)
        feedback_adjustments["domain"] += (0.008 + 0.016 * domain_mismatch_ratio) * max(feedback_scale, 0.1) * _objective_feedback_gain(objective, "domain")
        reasons.append("在当前领域场景下，漏抽或误抽反馈偏多，系统进一步提升领域权重并优先考虑领域专用模型。")

    if archive_count < 5:
        unsupervised_adjustments["stability"] += 0.02
        reasons.append("当前同类在线归档样本较少，系统适度提升稳定性权重。")
    elif archive_count >= 12:
        unsupervised_adjustments["domain"] += 0.01
        reasons.append(f"当前已有 {archive_count} 条同类在线归档证据，系统增强场景与领域相关性。")

    if request_domain != "general":
        domain_matches = 0
        for archive in matched_archives:
            model = model_lookup.get(archive.model_id)
            if model is None:
                continue
            if infer_domain(text=str(model.dataset_name or ""), scene=model.dataset_name) == request_domain:
                domain_matches += 1
        if domain_matches >= max(2, archive_count // 2):
            unsupervised_adjustments["domain"] += 0.02

    unsupervised_adjustments = {key: round(max(min(value, 0.05), -0.05), 4) for key, value in unsupervised_adjustments.items()}
    feedback_adjustments = {key: round(max(min(value, 0.05), -0.05), 4) for key, value in feedback_adjustments.items()}
    combined_adjustments = _sum_weight_adjustments(unsupervised_adjustments, feedback_adjustments)
    return combined_adjustments, reasons, unsupervised_adjustments, feedback_adjustments


def _build_decision_feedback_weight_adjustments(
    *,
    decision_records: list[TaskRoutingDecisionRecord],
    feedback_records: list[TaskRoutingFeedbackRecord],
    objective: str,
    requested_language: str,
    context_features: dict[str, Any],
    base_weights: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    feedback_by_decision = {item.decision_id: item for item in feedback_records}
    request_domain = str(context_features.get("domain") or "general")
    request_scene = _normalize_scene(context_features.get("scene"))
    matched_decisions: list[TaskRoutingDecisionRecord] = []
    for decision in decision_records:
        feedback = feedback_by_decision.get(decision.decision_id)
        if feedback is None:
            continue
        if decision.objective != objective:
            continue
        if requested_language and decision.inferred_language != requested_language:
            continue
        decision_domain = str(decision.context_features.get("domain") or "general")
        if request_domain != "general" and decision_domain not in {request_domain, "general"}:
            continue
        decision_scene = _normalize_scene(decision.scene)
        if request_scene and decision_scene and decision_scene != request_scene:
            continue
        matched_decisions.append(decision)

    adjustments = _zero_weight_vector()
    if not matched_decisions:
        return adjustments, []

    accepted_vectors: list[dict[str, float]] = []
    rejected_vectors: list[dict[str, float]] = []
    for decision in matched_decisions:
        feedback = feedback_by_decision.get(decision.decision_id)
        if feedback is None:
            continue
        if feedback.accepted:
            accepted_vectors.append(decision.objective_weights)
        else:
            rejected_vectors.append(decision.objective_weights)

    reasons: list[str] = []
    evidence_count = len(accepted_vectors) + len(rejected_vectors)
    evidence_scale = _progressive_evidence_scale(evidence_count, target_count=10)
    accepted_avg = _average_weight_vectors(accepted_vectors)
    rejected_avg = _average_weight_vectors(rejected_vectors)

    for key in adjustments:
        delta = 0.0
        if accepted_avg:
            delta += (accepted_avg.get(key, 0.0) - base_weights.get(key, 0.0)) * 0.32 * evidence_scale * _objective_feedback_gain(objective, key)
        if rejected_avg:
            delta -= (rejected_avg.get(key, 0.0) - base_weights.get(key, 0.0)) * 0.24 * evidence_scale * _objective_feedback_gain(objective, key)
        adjustments[key] = round(max(min(delta, 0.035), -0.035), 4)

    non_zero = {key: value for key, value in adjustments.items() if abs(value) > 0.0001}
    if non_zero:
        strongest_key = sorted(non_zero.items(), key=lambda item: abs(item[1]), reverse=True)[0][0]
        reasons.append(
            f"平台参考 {evidence_count} 条同目标调度反馈与历史权重快照，自动修正当前评分重点，当前影响最大的权重项为 {strongest_key}。"
        )
    if accepted_vectors:
        reasons.append(f"其中 {len(accepted_vectors)} 条历史调度结果被标记为满意，系统会向这些权重配置轻量靠拢。")
    if rejected_vectors:
        reasons.append(f"其中 {len(rejected_vectors)} 条历史调度结果被标记为不满意，系统会降低相近权重配置的优先级。")
    return adjustments, reasons[:4]


def _apply_adaptive_weight_adjustments(
    *,
    base_weights: dict[str, float],
    benchmark_adjustments: dict[str, float],
    archive_adjustments: dict[str, float],
    decision_feedback_adjustments: dict[str, float] | None = None,
) -> dict[str, float]:
    merged = {}
    for key, value in base_weights.items():
        merged[key] = (
            value
            + benchmark_adjustments.get(key, 0.0)
            + archive_adjustments.get(key, 0.0)
            + (decision_feedback_adjustments or {}).get(key, 0.0)
        )
    total = sum(max(item, 0.0) for item in merged.values()) or 1.0
    return {key: round(max(value, 0.0) / total, 4) for key, value in merged.items()}


def _sum_weight_adjustments(*vectors: dict[str, float]) -> dict[str, float]:
    summed = _zero_weight_vector()
    for vector in vectors:
        for key in summed:
            summed[key] += float(vector.get(key, 0.0) or 0.0)
    return {key: round(value, 4) for key, value in summed.items() if abs(value) > 0.0001}


def _average_weight_vectors(vectors: list[dict[str, float]]) -> dict[str, float]:
    if not vectors:
        return {}
    keys = {key for vector in vectors for key in vector}
    averaged = {}
    for key in keys:
        averaged[key] = sum(float(vector.get(key, 0.0) or 0.0) for vector in vectors) / len(vectors)
    return averaged


def _merge_adaptation_reasons(
    *,
    benchmark_adjustments: dict[str, float],
    archive_adjustments: dict[str, float],
    decision_feedback_adjustments: dict[str, float] | None = None,
    archive_reasons: list[str],
    decision_feedback_reasons: list[str] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if benchmark_adjustments.get("latency", 0.0) >= 0.04:
        reasons.append("近期基准记录显示尾延迟偏高，系统临时提升时延权重。")
    if benchmark_adjustments.get("stability", 0.0) >= 0.03:
        reasons.append("当前可参考基准较少或成功率偏低，系统适度提升稳定性权重。")
    if archive_adjustments.get("coverage", 0.0) >= 0.04:
        reasons.append("近期在线归档显示三元组产出稳定，系统适度提升覆盖相关权重。")
    if archive_adjustments.get("accuracy", 0.0) >= 0.04:
        reasons.append("近期在线归档显示空结果风险偏高，系统适度提升准确率权重。")
    if (decision_feedback_adjustments or {}).get("accuracy", 0.0) >= 0.03:
        reasons.append("历史任务调度反馈显示当前场景更依赖高准确配置，系统提升准确率权重。")
    if (decision_feedback_adjustments or {}).get("coverage", 0.0) >= 0.02:
        reasons.append("历史任务调度反馈显示当前场景更依赖覆盖能力，系统适度提升覆盖权重。")
    reasons.extend(archive_reasons)
    reasons.extend(decision_feedback_reasons or [])
    return reasons[:5]


def _candidate_domain_match(request_domain: Any, dataset_name: str | None) -> str:
    request_label = str(request_domain or "general")
    candidate_domain = infer_domain(text=str(dataset_name or ""), scene=dataset_name)
    if request_label == candidate_domain:
        return "high"
    if request_label == "general" or candidate_domain == "general":
        return "medium"
    return "low"


def _resource_efficiency_score(deployment: DeploymentRecord) -> float:
    execution = deployment.execution or {}
    instance_count = max(1, int(execution.get("instance_count", 1) or 1))
    intra_threads = max(1, int(execution.get("intra_op_threads", 1) or 1))
    inter_threads = max(1, int(execution.get("inter_op_threads", 1) or 1))
    gpu_penalty = 0.28 if deployment.use_gpu else 0.0
    worker_penalty = min(0.35, (instance_count - 1) * 0.12)
    thread_penalty = min(0.25, max(intra_threads, inter_threads) * 0.03)
    return max(0.0, 1.0 - gpu_penalty - worker_penalty - thread_penalty)


def _resource_cost_level(deployment: DeploymentRecord) -> str:
    score = _resource_efficiency_score(deployment)
    if score >= 0.72:
        return "low"
    if score >= 0.46:
        return "medium"
    return "high"


def _resolve_execution_roles(execution_path: str, count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["primary"]
    if execution_path in {"domain_priority_crosscheck", "cost_aware_primary_plus_reviewer"}:
        roles = ["primary", "reviewer", "supplement"]
        return roles[:count] + ["supplement"] * max(0, count - len(roles))
    if execution_path in {"parallel_multi_union", "multi_consensus_vote", "consensus_with_backfill"}:
        roles = ["primary", "reviewer", "supplement"]
        return roles[:count] + ["supplement"] * max(0, count - len(roles))
    return ["primary"] + ["supplement"] * (count - 1)


def _build_task_decision_trace(
    *,
    objective: str,
    difficulty: str,
    context_features: dict[str, Any],
    objective_weights: dict[str, float],
    selected_candidates: list[TaskRoutingCandidate],
    execution_path: str,
    fusion_policy: str,
) -> list[str]:
    trace = [
        f"识别语言：{context_features.get('language')}",
        f"任务难度：{difficulty}，领域：{context_features.get('domain')}",
        f"实体线索数：{context_features.get('entity_hint_count')}，子句数：{context_features.get('clause_count')}",
        "目标权重：" + " / ".join(f"{key}={value:.2f}" for key, value in objective_weights.items()),
        f"执行路径：{execution_path}，融合策略：{fusion_policy}",
    ]
    if selected_candidates:
        trace.append(
            "选中部署：" + "、".join(
                f"{item.model_id}({item.profile_name}, score={item.selection_score})"
                for item in selected_candidates
            )
        )
    else:
        trace.append(f"未找到满足 {objective} 目标的可用运行部署。")
    return trace


def build_task_strategy_explanation(
    *,
    objective: str,
    difficulty: str,
    strategy_mode: str,
    execution_path: str,
    fusion_policy: str,
    context_features: dict[str, Any],
    selected_candidates: list[TaskRoutingCandidate],
) -> str:
    selected_models = "、".join(item.model_id for item in selected_candidates) or "无可用模型"
    domain = str(context_features.get("domain") or "general")
    entity_hint_count = int(context_features.get("entity_hint_count") or 0)
    if objective == "latency":
        return (
            f"当前文本被判定为{difficulty}任务，系统优先选择 P95 延迟更低的单模型部署，"
            f"在 {domain} 场景下采用 {execution_path} 路径，使用 {selected_models} 以降低响应时间。"
        )
    if objective == "coverage":
        return (
            f"当前文本被判定为{difficulty}任务，系统识别到约 {entity_hint_count} 个潜在实体线索，"
            f"采用 {strategy_mode}，执行 {execution_path} 并以 {fusion_policy} 融合 {selected_models} 的结果，以提升三元组覆盖率。"
        )
    if objective == "resource":
        return (
            f"当前文本被判定为{difficulty}任务，系统在 {domain} 场景下优先控制资源成本，"
            f"采用 {execution_path}，选择 {selected_models} 以降低 GPU/实例开销，并保留可接受的抽取质量。"
        )
    if objective == "domain":
        return (
            f"当前文本被判定为{difficulty}任务，系统识别其更接近 {domain} 领域，"
            f"采用 {execution_path} 优先调用领域匹配更高的模型，并以 {fusion_policy} 进行交叉校验。"
        )
    if objective == "balanced":
        return (
            f"当前文本被判定为{difficulty}任务，系统在准确率、覆盖率、时延与成本之间做折中，"
            f"采用 {execution_path}，并对 {selected_models} 执行 {fusion_policy} 融合。"
        )
    return (
        f"当前文本被判定为{difficulty}任务，系统采用 {strategy_mode}，"
        f"执行 {execution_path}，汇总 {selected_models} 的抽取结果并按 {fusion_policy} 提升结果稳定性。"
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
    context_features: dict[str, Any],
    objective_weights: dict[str, float],
) -> list[TaskRoutingCandidate]:
    model_lookup = {item.model_id: item for item in models}
    completed_benchmarks = [item for item in benchmarks if item.status == "completed"]
    request_scene = context_features.get("scene")
    benchmark_by_deployment = defaultdict(list)
    benchmark_by_model = defaultdict(list)
    for item in completed_benchmarks:
        benchmark_by_deployment[item.deployment_id].append(item)
        benchmark_by_model[item.model_id].append(item)

    archives_by_deployment = defaultdict(list)
    archives_by_model = defaultdict(list)
    for item in prediction_archives:
        archives_by_deployment[item.deployment_id].append(item)
        archives_by_model[item.model_id].append(item)

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
        if not _scene_matches_model(request_scene, model):
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
        domain_match = _candidate_domain_match(context_features.get("domain"), model.dataset_name)
        resource_cost_level = _resource_cost_level(deployment)
        candidate_archives = _match_candidate_archives(
            deployment=deployment,
            model=model,
            deployment_archives=archives_by_deployment.get(deployment.deployment_id, []),
            model_archives=archives_by_model.get(deployment.model_id, []),
            request_scene=request_scene,
            request_domain=str(context_features.get("domain") or "general"),
            requested_language=requested_language,
        )
        archive_feedback_summary = _build_candidate_archive_feedback_summary(candidate_archives)
        score_breakdown = _selection_score_breakdown(
            objective=objective,
            success_rate=avg_success,
            p95_latency_ms=avg_p95,
            recent_request_count=recent_requests,
            avg_triple_count=avg_triple_count,
            domain_match=domain_match,
            deployment=deployment,
            archive_feedback_summary=archive_feedback_summary,
        )
        score = sum(
            score_breakdown.get(key, 0.0) * objective_weights.get(key, 0.0)
            for key in objective_weights
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
                    score_breakdown=score_breakdown,
                    p95_latency_ms=avg_p95,
                    recent_request_count=recent_requests,
                    domain_match=domain_match,
                    resource_cost_level=resource_cost_level,
                    archive_feedback_summary=archive_feedback_summary,
                ),
                score_breakdown={key: round(value, 4) for key, value in score_breakdown.items()},
                estimated_p95_latency_ms=avg_p95,
                benchmark_success_rate=avg_success,
                recent_request_count=recent_requests,
                predicted_triple_count=int(round(avg_triple_count))
                if avg_triple_count is not None
                else None,
                domain_match=domain_match,
                resource_cost_level=resource_cost_level,
            )
        )

    candidates.sort(
        key=lambda item: (
            -(item.selection_score if item.selection_score is not None else -math.inf),
            -item.recent_request_count,
            item.model_id,
        ),
    )
    return candidates


def _selection_score_breakdown(
    *,
    objective: str,
    success_rate: float | None,
    p95_latency_ms: float | None,
    recent_request_count: int,
    avg_triple_count: float | None,
    domain_match: str,
    deployment: DeploymentRecord,
    archive_feedback_summary: dict[str, float | int | None] | None = None,
) -> dict[str, float]:
    accuracy_score = min(1.0, max(0.0, float(success_rate if success_rate is not None else 0.55)))
    latency_score = max(0.0, 1.0 - min(float(p95_latency_ms or 1200.0), 1500.0) / 1500.0)
    coverage_score = min(1.0, ((avg_triple_count or 0.0) / 18.0) + min(recent_request_count, 10) / 30.0)
    stability_score = min(1.0, accuracy_score * 0.7 + min(recent_request_count, 10) / 10.0 * 0.3)
    resource_score = _resource_efficiency_score(deployment)
    domain_score = {"high": 1.0, "medium": 0.65, "low": 0.3}.get(domain_match, 0.45)
    if archive_feedback_summary:
        effective_feedback_count = int(archive_feedback_summary.get("effective_feedback_count") or 0)
        satisfied_rate = archive_feedback_summary.get("satisfied_rate")
        empty_rate = float(archive_feedback_summary.get("empty_rate") or 0.0)
        archive_count = int(archive_feedback_summary.get("archive_count") or 0)
        archive_avg_latency = archive_feedback_summary.get("avg_latency_ms")
        missing_too_many_count = int(archive_feedback_summary.get("missing_too_many_count") or 0)
        false_positive_too_many_count = int(archive_feedback_summary.get("false_positive_too_many_count") or 0)
        latency_high_count = int(archive_feedback_summary.get("latency_high_count") or 0)
        archive_scale = _progressive_evidence_scale(archive_count, target_count=30)
        feedback_scale = _progressive_evidence_scale(effective_feedback_count, target_count=10)
        if satisfied_rate is not None and effective_feedback_count:
            centered_rate = float(satisfied_rate) - 0.5
            accuracy_score = min(1.0, max(0.0, accuracy_score + centered_rate * (0.08 + 0.12 * feedback_scale) * _objective_feedback_gain(objective, "accuracy")))
            stability_score = min(1.0, max(0.0, stability_score + centered_rate * (0.06 + 0.10 * feedback_scale) * _objective_feedback_gain(objective, "stability")))
            if centered_rate > 0 and (archive_feedback_summary.get("avg_triple_count") or 0.0) >= 1.0:
                coverage_score = min(1.0, coverage_score + centered_rate * (0.03 + 0.05 * feedback_scale) * _objective_feedback_gain(objective, "coverage"))
            elif centered_rate < 0:
                coverage_score = max(0.0, coverage_score + centered_rate * (0.02 + 0.04 * feedback_scale) * _objective_feedback_gain(objective, "coverage"))
        if empty_rate >= 0.4:
            coverage_score = max(0.0, coverage_score - min(0.10, empty_rate * 0.10 * max(archive_scale, 0.2)))
            stability_score = max(0.0, stability_score - min(0.08, empty_rate * 0.08 * max(archive_scale, 0.2)))
        if effective_feedback_count and missing_too_many_count:
            missing_ratio = missing_too_many_count / effective_feedback_count
            coverage_score = max(0.0, coverage_score - min(0.12, (0.02 + missing_ratio * 0.08) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "coverage")))
        if effective_feedback_count and false_positive_too_many_count:
            false_positive_ratio = false_positive_too_many_count / effective_feedback_count
            accuracy_score = max(0.0, accuracy_score - min(0.12, (0.03 + false_positive_ratio * 0.07) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "accuracy")))
            stability_score = max(0.0, stability_score - min(0.10, (0.02 + false_positive_ratio * 0.06) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "stability")))
            coverage_score = max(0.0, coverage_score - min(0.05, false_positive_ratio * 0.04 * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "coverage")))
        if effective_feedback_count and latency_high_count:
            latency_high_ratio = latency_high_count / effective_feedback_count
            latency_score = max(0.0, latency_score - min(0.14, (0.03 + latency_high_ratio * 0.08) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "latency")))
            resource_score = max(0.0, resource_score - min(0.08, (0.02 + latency_high_ratio * 0.05) * max(feedback_scale, 0.12) * _objective_feedback_gain(objective, "resource")))
        if p95_latency_ms is None and archive_avg_latency is not None:
            latency_score = max(0.0, 1.0 - min(float(archive_avg_latency), 1500.0) / 1500.0)
    return {
        "accuracy": round(accuracy_score, 6),
        "latency": round(latency_score, 6),
        "coverage": round(coverage_score, 6),
        "stability": round(stability_score, 6),
        "resource": round(resource_score, 6),
        "domain": round(domain_score, 6),
    }


def _selection_reason(
    *,
    objective: str,
    score_breakdown: dict[str, float],
    p95_latency_ms: float | None,
    recent_request_count: int,
    domain_match: str,
    resource_cost_level: str,
    archive_feedback_summary: dict[str, float | int | None] | None = None,
) -> str:
    latency_text = f"{p95_latency_ms:.3f} ms" if p95_latency_ms is not None else "暂无 P95"
    success_text = f"{score_breakdown.get('accuracy', 0.0) * 100:.1f}%"
    feedback_text = _build_archive_feedback_reason_text(archive_feedback_summary)
    if objective == "latency":
        return f"优先考虑低尾延迟，本候选历史 P95 为 {latency_text}。{feedback_text}".strip()
    if objective == "coverage":
        return f"优先考虑覆盖与稳定运行，本候选成功率 {success_text}，近期请求 {recent_request_count} 次。{feedback_text}".strip()
    if objective == "resource":
        return f"优先考虑低资源成本，本候选资源级别 {resource_cost_level}，历史 P95 为 {latency_text}。{feedback_text}".strip()
    if objective == "domain":
        return f"优先考虑领域匹配，本候选领域匹配度 {domain_match}，成功率 {success_text}。{feedback_text}".strip()
    if objective == "balanced":
        return f"兼顾准确、覆盖、时延与成本，本候选成功率 {success_text}，P95 为 {latency_text}。{feedback_text}".strip()
    return f"优先考虑稳定性，本候选成功率 {success_text}，历史 P95 为 {latency_text}。{feedback_text}".strip()


def _archive_matches_request_context(
    *,
    archive: PredictionArchiveRecord,
    model: ModelSummary,
    requested_language: str,
    request_scene: str | None,
    request_domain: str,
) -> bool:
    normalized_scene = _normalize_scene(request_scene)
    if requested_language and model.language != requested_language:
        return False
    if not _scene_matches_model(normalized_scene, model):
        return False
    if request_domain != "general":
        candidate_domain = infer_domain(text=str(model.dataset_name or ""), scene=model.dataset_name)
        if candidate_domain not in {request_domain, "general"}:
            return False
    if normalized_scene and archive.scene:
        archive_scene = _normalize_scene(archive.scene)
        if archive_scene != normalized_scene and archive_scene not in {request_domain, "general"}:
            return False
    return True


def _match_candidate_archives(
    *,
    deployment: DeploymentRecord,
    model: ModelSummary,
    deployment_archives: list[PredictionArchiveRecord],
    model_archives: list[PredictionArchiveRecord],
    request_scene: str | None,
    request_domain: str,
    requested_language: str,
) -> list[PredictionArchiveRecord]:
    matched_deployment_archives = [
        item
        for item in deployment_archives
        if _archive_matches_request_context(
            archive=item,
            model=model,
            requested_language=requested_language,
            request_scene=request_scene,
            request_domain=request_domain,
        )
    ]
    if matched_deployment_archives:
        return matched_deployment_archives
    return [
        item
        for item in model_archives
        if _archive_matches_request_context(
            archive=item,
            model=model,
            requested_language=requested_language,
            request_scene=request_scene,
            request_domain=request_domain,
        )
    ]


def _build_candidate_archive_feedback_summary(
    archives: list[PredictionArchiveRecord],
) -> dict[str, float | int | None]:
    if not archives:
        return {
            "archive_count": 0,
            "effective_feedback_count": 0,
            "satisfied_count": 0,
            "unsatisfied_count": 0,
            "missing_too_many_count": 0,
            "false_positive_too_many_count": 0,
            "latency_high_count": 0,
            "satisfied_rate": None,
            "avg_latency_ms": None,
            "avg_triple_count": None,
            "empty_rate": 0.0,
        }
    satisfied_count = sum(1 for item in archives if str(item.feedback_status or "pending").strip().lower() == "satisfied")
    unsatisfied_count = sum(1 for item in archives if str(item.feedback_status or "pending").strip().lower() == "unsatisfied")
    effective_feedback_count = satisfied_count + unsatisfied_count
    empty_count = sum(1 for item in archives if not int(item.triple_count or 0))
    missing_too_many_count = sum(
        1
        for item in archives
        if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
        and "missing_too_many" in _parse_archive_feedback_reasons(item.feedback_note)
    )
    false_positive_too_many_count = sum(
        1
        for item in archives
        if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
        and "false_positive_too_many" in _parse_archive_feedback_reasons(item.feedback_note)
    )
    latency_high_count = sum(
        1
        for item in archives
        if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
        and "latency_high" in _parse_archive_feedback_reasons(item.feedback_note)
    )
    return {
        "archive_count": len(archives),
        "effective_feedback_count": effective_feedback_count,
        "satisfied_count": satisfied_count,
        "unsatisfied_count": unsatisfied_count,
        "missing_too_many_count": missing_too_many_count,
        "false_positive_too_many_count": false_positive_too_many_count,
        "latency_high_count": latency_high_count,
        "satisfied_rate": (satisfied_count / effective_feedback_count) if effective_feedback_count else None,
        "avg_latency_ms": _mean_numeric(item.latency_ms for item in archives),
        "avg_triple_count": _mean_numeric(item.triple_count for item in archives),
        "empty_rate": empty_count / len(archives),
    }


def _build_archive_feedback_reason_text(
    archive_feedback_summary: dict[str, float | int | None] | None,
) -> str:
    if not archive_feedback_summary:
        return ""
    archive_count = int(archive_feedback_summary.get("archive_count") or 0)
    effective_feedback_count = int(archive_feedback_summary.get("effective_feedback_count") or 0)
    satisfied_count = int(archive_feedback_summary.get("satisfied_count") or 0)
    unsatisfied_count = int(archive_feedback_summary.get("unsatisfied_count") or 0)
    missing_too_many_count = int(archive_feedback_summary.get("missing_too_many_count") or 0)
    false_positive_too_many_count = int(archive_feedback_summary.get("false_positive_too_many_count") or 0)
    latency_high_count = int(archive_feedback_summary.get("latency_high_count") or 0)
    if not archive_count:
        return "当前暂无同类在线归档反馈。"
    if effective_feedback_count:
        detail_parts = []
        if missing_too_many_count:
            detail_parts.append(f"漏抽偏多 {missing_too_many_count} 条")
        if false_positive_too_many_count:
            detail_parts.append(f"误抽偏多 {false_positive_too_many_count} 条")
        if latency_high_count:
            detail_parts.append(f"延迟偏高 {latency_high_count} 条")
        detail_text = f"；其中{'，'.join(detail_parts)}" if detail_parts else ""
        return (
            f"同类在线归档 {archive_count} 条，其中满意 {satisfied_count} 条、不满意 {unsatisfied_count} 条，"
            f"{detail_text}，这些记录会直接参与本次候选评分。"
        )
    return f"当前已有同类在线归档 {archive_count} 条，但尚无明确满意度反馈，平台暂以运行结果作为辅助证据。"


def _parse_archive_feedback_reasons(feedback_note: str | None) -> set[str]:
    supported = {"missing_too_many", "false_positive_too_many", "latency_high"}
    return {
        item.strip().lower()
        for item in str(feedback_note or "").split(",")
        if item.strip().lower() in supported
    }


def _select_task_candidates(
    candidates: list[TaskRoutingCandidate],
    *,
    objective: str,
    difficulty: str,
    context_features: dict[str, Any],
    max_models: int,
) -> tuple[str, str, str, list[TaskRoutingCandidate]]:
    if not candidates:
        return ("none", "none", "empty", [])
    bounded_max_models = max(1, min(max_models, len(candidates)))
    if objective == "latency":
        return ("multi-objective-adaptive", "single_low_latency", "single_primary", candidates[:1])

    if objective == "coverage":
        target_count = min(bounded_max_models, 2 if difficulty == "simple" else bounded_max_models)
        return ("multi-objective-adaptive", "parallel_multi_union", "union_all", _unique_models(candidates, target_count))

    if objective == "resource":
        if difficulty == "hard" and len(candidates) > 1:
            return (
                "multi-objective-adaptive",
                "cost_aware_primary_plus_reviewer",
                "primary_with_fallback",
                _unique_models(candidates, min(2, bounded_max_models)),
            )
        return ("multi-objective-adaptive", "single_low_cost", "single_primary", candidates[:1])

    if objective == "domain":
        domain_sorted = sorted(
            candidates,
            key=lambda item: (
                {"high": 2, "medium": 1, "low": 0}.get(item.domain_match, 0),
                item.selection_score or 0.0,
            ),
            reverse=True,
        )
        target_count = 1 if difficulty == "simple" else bounded_max_models
        return (
            "multi-objective-adaptive",
            "domain_priority_crosscheck" if target_count > 1 else "domain_priority_single",
            "domain_priority",
            _unique_models(domain_sorted, target_count),
        )

    if objective == "balanced":
        target_count = 1 if difficulty == "simple" else bounded_max_models
        return (
            "multi-objective-adaptive",
            "consensus_with_backfill" if target_count > 1 else "single_balanced",
            "consensus_backfill" if target_count > 1 else "single_primary",
            _unique_models(candidates, target_count),
        )

    if difficulty == "simple":
        return ("multi-objective-adaptive", "single_high_confidence", "single_primary", candidates[:1])

    target_count = min(bounded_max_models, 2 if difficulty == "medium" else bounded_max_models)
    return ("multi-objective-adaptive", "multi_consensus_vote", "majority_vote", _unique_models(candidates, target_count))


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
    execution_role: str = "primary",
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
        try:
            with urlopen(request, timeout=90) as response:  # noqa: S310
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return TaskRoutingExecution(
                deployment_id=candidate.deployment_id,
                model_id=candidate.model_id,
                profile_name=candidate.profile_name,
                selected_for_fusion=False,
                execution_role=execution_role,
                latency_ms=None,
                triple_count=0,
                triples=[],
                warnings=[f"runtime_http_error: {exc.code} {exc.reason}"],
            )
        except URLError as exc:
            return TaskRoutingExecution(
                deployment_id=candidate.deployment_id,
                model_id=candidate.model_id,
                profile_name=candidate.profile_name,
                selected_for_fusion=False,
                execution_role=execution_role,
                latency_ms=None,
                triple_count=0,
                triples=[],
                warnings=[f"runtime_unreachable: {exc.reason}"],
            )
        except json.JSONDecodeError as exc:
            return TaskRoutingExecution(
                deployment_id=candidate.deployment_id,
                model_id=candidate.model_id,
                profile_name=candidate.profile_name,
                selected_for_fusion=False,
                execution_role=execution_role,
                latency_ms=None,
                triple_count=0,
                triples=[],
                warnings=[f"runtime_invalid_json: {exc.msg}"],
            )
        except Exception as exc:
            return TaskRoutingExecution(
                deployment_id=candidate.deployment_id,
                model_id=candidate.model_id,
                profile_name=candidate.profile_name,
                selected_for_fusion=False,
                execution_role=execution_role,
                latency_ms=None,
                triple_count=0,
                triples=[],
                warnings=[f"runtime_error: {type(exc).__name__}: {exc}"],
            )
    finally:
        if activity_tracker is not None:
            activity_tracker.end(activity_token)
    return TaskRoutingExecution(
        deployment_id=candidate.deployment_id,
        model_id=candidate.model_id,
        profile_name=candidate.profile_name,
        selected_for_fusion=True,
        execution_role=execution_role,
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
    fusion_policy: str,
) -> dict[str, Any]:
    if not execution_results:
        return {
            "triple_count": 0,
            "triples": [],
            "fusion_mode": "empty",
        }

    if fusion_policy == "single_primary" or objective == "latency" or len(execution_results) == 1:
        primary = execution_results[0]
        return {
            "triple_count": primary.triple_count,
            "triples": _resolve_relation_conflicts(primary.triples, inferred_language, keep_all=False),
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

    if fusion_policy == "union_all" or objective == "coverage":
        triples = [
            {
                **triple_votes[signature],
                "vote_count": normalized_counter[signature],
            }
            for signature, _count in normalized_counter.most_common()
        ]
        triples = _resolve_relation_conflicts(triples, inferred_language, keep_all=True)
        return {
            "triple_count": len(triples),
            "triples": triples,
            "fusion_mode": "union",
        }

    if fusion_policy == "primary_with_fallback":
        primary = execution_results[0]
        seen = {_triple_signature(item, inferred_language) for item in primary.triples}
        triples = [dict(item) for item in primary.triples]
        for result in execution_results[1:]:
            for triple in result.triples:
                signature = _triple_signature(triple, inferred_language)
                if signature in seen:
                    continue
                seen.add(signature)
                triples.append({**dict(triple), "vote_count": 1})
        triples = _resolve_relation_conflicts(triples, inferred_language, keep_all=False)
        return {
            "triple_count": len(triples),
            "triples": triples,
            "fusion_mode": "primary_with_fallback",
        }

    if fusion_policy == "domain_priority":
        primary = execution_results[0]
        supplemental = execution_results[1:] if len(execution_results) > 1 else []
        triples = [dict(item) for item in primary.triples]
        seen = {_triple_signature(item, inferred_language) for item in triples}
        for result in supplemental:
            for triple in result.triples:
                signature = _triple_signature(triple, inferred_language)
                if signature in seen:
                    continue
                seen.add(signature)
                triples.append({**dict(triple), "vote_count": 1})
        triples = _resolve_relation_conflicts(triples, inferred_language, keep_all=False)
        return {
            "triple_count": len(triples),
            "triples": triples,
            "fusion_mode": "domain_priority",
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
    if fusion_policy == "consensus_backfill":
        seen = {_triple_signature(item, inferred_language) for item in voted}
        for signature, count in normalized_counter.most_common():
            if signature in seen:
                continue
            voted.append(
                {
                    **triple_votes[signature],
                    "vote_count": count,
                }
            )
            if len(voted) >= max(5, len(execution_results) * 4):
                break
    voted = _resolve_relation_conflicts(voted, inferred_language, keep_all=False)
    return {
        "triple_count": len(voted),
        "triples": voted,
        "fusion_mode": fusion_policy or "majority_vote",
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


def _resolve_relation_conflicts(
    triples: list[dict[str, Any]],
    language: str,
    *,
    keep_all: bool,
) -> list[dict[str, Any]]:
    if keep_all:
        return triples
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for triple in triples:
        subject = _triple_signature(triple, language)[0]
        obj = _triple_signature(triple, language)[2]
        key = (subject, obj)
        current = chosen.get(key)
        current_vote = int(current.get("vote_count", 1)) if current else -1
        next_vote = int(triple.get("vote_count", 1))
        current_conf = _safe_float(current.get("confidence")) if current else None
        next_conf = _safe_float(triple.get("confidence"))
        if (
            current is None
            or next_vote > current_vote
            or (next_vote == current_vote and (next_conf or 0.0) > (current_conf or 0.0))
        ):
            chosen[key] = triple
    return list(chosen.values())


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


def _scene_context(scene: str) -> tuple[str, dict[str, Any]]:
    normalized_scene = _normalize_scene(scene)
    if normalized_scene == "en_general":
        return ("en", {"domain": "general", "language": "en", "scene": normalized_scene})
    if normalized_scene == "zh_industrial":
        return ("zh", {"domain": "industrial", "language": "zh", "scene": normalized_scene})
    return ("zh", {"domain": "general", "language": "zh", "scene": normalized_scene or "zh_general"})


def _build_model_score_summaries(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    prediction_archives: list[PredictionArchiveRecord],
    decision_records: list[TaskRoutingDecisionRecord],
    feedback_records: list[TaskRoutingFeedbackRecord],
) -> list[TaskRoutingModelScoreSummary]:
    model_lookup = {item.model_id: item for item in models}
    completed_benchmarks = [item for item in benchmarks if item.status == "completed"]
    benchmark_by_deployment = defaultdict(list)
    benchmark_by_model = defaultdict(list)
    for item in completed_benchmarks:
        benchmark_by_deployment[item.deployment_id].append(item)
        benchmark_by_model[item.model_id].append(item)

    archives_by_deployment = defaultdict(list)
    archives_by_model = defaultdict(list)
    for item in prediction_archives:
        archives_by_deployment[item.deployment_id].append(item)
        archives_by_model[item.model_id].append(item)

    objective_order = ["accuracy", "coverage", "latency", "balanced", "resource", "domain"]
    scene_order = ["en_general", "zh_general", "zh_industrial"]
    summaries: list[TaskRoutingModelScoreSummary] = []

    for objective in objective_order:
        for scene in scene_order:
            requested_language, context_features = _scene_context(scene)
            base_weights = _build_objective_weights(
                objective=objective,
                difficulty="medium",
                context_features=context_features,
            )
            benchmark_adjustments = _build_benchmark_weight_adjustments(
                models=models,
                deployments=deployments,
                benchmarks=benchmarks,
                requested_language=requested_language,
                context_features=context_features,
            )
            archive_adjustments, _archive_reasons, _unsupervised_adjustments, _archive_feedback_adjustments = _build_archive_weight_adjustments(
                models=models,
                deployments=deployments,
                prediction_archives=prediction_archives,
                context_features=context_features,
                requested_language=requested_language,
                objective=objective,
            )
            decision_feedback_adjustments, _decision_feedback_reasons = _build_decision_feedback_weight_adjustments(
                decision_records=decision_records,
                feedback_records=feedback_records,
                objective=objective,
                requested_language=requested_language,
                context_features=context_features,
                base_weights=base_weights,
            )
            feedback_adjustments = _sum_weight_adjustments(
                _archive_feedback_adjustments,
                decision_feedback_adjustments,
            )
            adjusted_weights = _apply_adaptive_weight_adjustments(
                base_weights=base_weights,
                benchmark_adjustments=benchmark_adjustments,
                archive_adjustments=archive_adjustments,
                decision_feedback_adjustments=decision_feedback_adjustments,
            )

            for deployment in deployments:
                if deployment.status != "running" or not deployment.service_url:
                    continue
                model = model_lookup.get(deployment.model_id)
                if model is None:
                    continue
                if model.language != requested_language:
                    continue
                if not _scene_matches_model(scene, model):
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
                domain_match = _candidate_domain_match(context_features.get("domain"), model.dataset_name)
                candidate_archives = _match_candidate_archives(
                    deployment=deployment,
                    model=model,
                    deployment_archives=archives_by_deployment.get(deployment.deployment_id, []),
                    model_archives=archives_by_model.get(deployment.model_id, []),
                    request_scene=scene,
                    request_domain=str(context_features.get("domain") or "general"),
                    requested_language=requested_language,
                )
                archive_feedback_summary = _build_candidate_archive_feedback_summary(candidate_archives)
                baseline_breakdown = _selection_score_breakdown(
                    objective=objective,
                    success_rate=avg_success,
                    p95_latency_ms=avg_p95,
                    recent_request_count=recent_requests,
                    avg_triple_count=avg_triple_count,
                    domain_match=domain_match,
                    deployment=deployment,
                    archive_feedback_summary=None,
                )
                adjusted_breakdown = _selection_score_breakdown(
                    objective=objective,
                    success_rate=avg_success,
                    p95_latency_ms=avg_p95,
                    recent_request_count=recent_requests,
                    avg_triple_count=avg_triple_count,
                    domain_match=domain_match,
                    deployment=deployment,
                    archive_feedback_summary=archive_feedback_summary,
                )
                baseline_score = sum(
                    baseline_breakdown.get(key, 0.0) * base_weights.get(key, 0.0)
                    for key in base_weights
                )
                current_score = sum(
                    adjusted_breakdown.get(key, 0.0) * adjusted_weights.get(key, 0.0)
                    for key in adjusted_weights
                )
                contribution_delta = {
                    key: (adjusted_breakdown.get(key, 0.0) * adjusted_weights.get(key, 0.0))
                    - (baseline_breakdown.get(key, 0.0) * base_weights.get(key, 0.0))
                    for key in base_weights
                }
                performance_score_breakdown = {
                    key: (adjusted_breakdown.get(key, 0.0) - baseline_breakdown.get(key, 0.0))
                    * base_weights.get(key, 0.0)
                    for key in base_weights
                    if abs(
                        (adjusted_breakdown.get(key, 0.0) - baseline_breakdown.get(key, 0.0))
                        * base_weights.get(key, 0.0)
                    ) > 0.000001
                }
                weight_shift_score_breakdown = {
                    key: adjusted_breakdown.get(key, 0.0)
                    * (adjusted_weights.get(key, 0.0) - base_weights.get(key, 0.0))
                    for key in base_weights
                    if abs(
                        adjusted_breakdown.get(key, 0.0)
                        * (adjusted_weights.get(key, 0.0) - base_weights.get(key, 0.0))
                    ) > 0.000001
                }
                performance_delta_breakdown = {
                    key: adjusted_breakdown.get(key, 0.0) - baseline_breakdown.get(key, 0.0)
                    for key in base_weights
                    if abs(adjusted_breakdown.get(key, 0.0) - baseline_breakdown.get(key, 0.0)) > 0.000001
                }
                benchmark_score_breakdown = {
                    key: adjusted_breakdown.get(key, 0.0) * benchmark_adjustments.get(key, 0.0)
                    for key in adjusted_breakdown
                    if abs(adjusted_breakdown.get(key, 0.0) * benchmark_adjustments.get(key, 0.0)) > 0.000001
                }
                no_feedback_score_breakdown = {
                    key: adjusted_breakdown.get(key, 0.0) * _unsupervised_adjustments.get(key, 0.0)
                    for key in adjusted_breakdown
                    if abs(adjusted_breakdown.get(key, 0.0) * _unsupervised_adjustments.get(key, 0.0)) > 0.000001
                }
                feedback_score_breakdown = {
                    key: adjusted_breakdown.get(key, 0.0) * feedback_adjustments.get(key, 0.0)
                    for key in adjusted_breakdown
                    if abs(adjusted_breakdown.get(key, 0.0) * feedback_adjustments.get(key, 0.0)) > 0.000001
                }
                benchmark_score_delta = sum(
                    benchmark_score_breakdown.values()
                )
                no_feedback_score_delta = sum(
                    no_feedback_score_breakdown.values()
                )
                feedback_score_delta = sum(
                    feedback_score_breakdown.values()
                )
                performance_score_delta = sum(
                    performance_score_breakdown.values()
                )
                weight_shift_score_delta = sum(
                    weight_shift_score_breakdown.values()
                )
                dominant_dimension, dominant_delta = max(
                    contribution_delta.items(),
                    key=lambda item: abs(item[1]),
                    default=(None, 0.0),
                )
                score_delta = current_score - baseline_score
                if dominant_dimension:
                    if dominant_delta >= 0:
                        change_note = f"{dominant_dimension} 维度抬升最明显，带动当前评分上移。"
                    else:
                        change_note = f"{dominant_dimension} 维度下降最明显，拉低了当前评分。"
                else:
                    change_note = "当前评分变化较小。"
                summaries.append(
                    TaskRoutingModelScoreSummary(
                        objective=objective,
                        scene=scene,
                        deployment_id=deployment.deployment_id,
                        model_id=deployment.model_id,
                        profile_name=deployment.profile_name,
                        language=model.language,
                        dataset_name=model.dataset_name,
                        baseline_score=round(baseline_score, 6),
                        current_score=round(current_score, 6),
                        score_delta=round(score_delta, 6),
                        benchmark_score_delta=round(benchmark_score_delta, 6),
                        no_feedback_score_delta=round(no_feedback_score_delta, 6),
                        feedback_score_delta=round(feedback_score_delta, 6),
                        performance_score_delta=round(performance_score_delta, 6),
                        weight_shift_score_delta=round(weight_shift_score_delta, 6),
                        benchmark_score_breakdown={key: round(value, 6) for key, value in benchmark_score_breakdown.items()},
                        no_feedback_score_breakdown={key: round(value, 6) for key, value in no_feedback_score_breakdown.items()},
                        feedback_score_breakdown={key: round(value, 6) for key, value in feedback_score_breakdown.items()},
                        performance_score_breakdown={key: round(value, 6) for key, value in performance_score_breakdown.items()},
                        weight_shift_score_breakdown={key: round(value, 6) for key, value in weight_shift_score_breakdown.items()},
                        baseline_breakdown={key: round(value, 4) for key, value in baseline_breakdown.items()},
                        current_breakdown={key: round(value, 4) for key, value in adjusted_breakdown.items()},
                        performance_delta_breakdown={key: round(value, 6) for key, value in performance_delta_breakdown.items()},
                        dominant_dimension=dominant_dimension,
                        dominant_delta=round(dominant_delta, 6) if dominant_dimension is not None else None,
                        change_note=change_note,
                    )
                )

    summaries.sort(
        key=lambda item: (
            item.objective,
            item.scene,
            -abs(item.score_delta),
            -(item.current_score or 0.0),
            item.model_id,
        )
    )
    return summaries


def build_task_routing_feedback_overview(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    prediction_archives: list[PredictionArchiveRecord],
    decision_records: list[TaskRoutingDecisionRecord],
    feedback_records: list[TaskRoutingFeedbackRecord],
    limit: int = 12,
) -> TaskRoutingFeedbackOverview:
    feedback_by_decision = {item.decision_id: item for item in feedback_records}
    total_feedback = len(feedback_records)
    accepted_feedback = [item for item in feedback_records if item.accepted]
    acceptance_rate = (len(accepted_feedback) / total_feedback) if total_feedback else None
    last_feedback_at = max((item.updated_at for item in feedback_records), default=None)
    archive_satisfied_count = sum(
        1 for item in prediction_archives if str(item.feedback_status or "pending").strip().lower() == "satisfied"
    )
    archive_unsatisfied_count = sum(
        1 for item in prediction_archives if str(item.feedback_status or "pending").strip().lower() == "unsatisfied"
    )
    archive_pending_count = max(
        len(prediction_archives) - archive_satisfied_count - archive_unsatisfied_count,
        0,
    )

    objective_summaries: list[TaskRoutingObjectiveFeedbackSummary] = []
    decisions_by_objective: dict[str, list[TaskRoutingDecisionRecord]] = defaultdict(list)
    for item in decision_records:
        decisions_by_objective[item.objective].append(item)
    for objective, items in sorted(decisions_by_objective.items()):
        path_counter = Counter(item.execution_path for item in items if item.execution_path)
        feedback_subset = [feedback_by_decision[item.decision_id] for item in items if item.decision_id in feedback_by_decision]
        objective_summaries.append(
            TaskRoutingObjectiveFeedbackSummary(
                objective=objective,
                decision_count=len(items),
                feedback_count=len(feedback_subset),
                acceptance_rate=(sum(item.accepted for item in feedback_subset) / len(feedback_subset)) if feedback_subset else None,
                best_execution_path=path_counter.most_common(1)[0][0] if path_counter else None,
            )
        )

    execution_path_summaries: list[TaskRoutingExecutionPathSummary] = []
    decisions_by_path: dict[str, list[TaskRoutingDecisionRecord]] = defaultdict(list)
    for item in decision_records:
        decisions_by_path[item.execution_path].append(item)
    for execution_path, items in sorted(decisions_by_path.items()):
        feedback_subset = [feedback_by_decision[item.decision_id] for item in items if item.decision_id in feedback_by_decision]
        execution_path_summaries.append(
            TaskRoutingExecutionPathSummary(
                execution_path=execution_path,
                decision_count=len(items),
                feedback_count=len(feedback_subset),
                acceptance_rate=(sum(item.accepted for item in feedback_subset) / len(feedback_subset)) if feedback_subset else None,
            )
        )
    execution_path_summaries.sort(
        key=lambda item: (
            -item.decision_count,
            -item.feedback_count,
            item.execution_path,
        )
    )

    adopted_counter: Counter[str] = Counter()
    for item in feedback_records:
        adopted_model_id = item.adopted_model_id
        if adopted_model_id:
            adopted_counter[adopted_model_id] += 1
    model_adoption_summaries = [
        TaskRoutingModelAdoptionSummary(
            model_id=model_id,
            adoption_count=count,
            adopted_share=round(count / total_feedback, 4) if total_feedback else None,
        )
        for model_id, count in adopted_counter.most_common(8)
    ]

    weight_profiles: list[TaskRoutingAdaptiveWeightProfile] = []
    for objective in ["accuracy", "coverage", "latency", "balanced", "resource", "domain"]:
        base_weights = _build_objective_weights(
            objective=objective,
            difficulty="medium",
            context_features={"domain": "general", "language": ""},
        )
        benchmark_adjustments = _build_benchmark_weight_adjustments(
            models=models,
            deployments=deployments,
            benchmarks=benchmarks,
            requested_language="",
            context_features={"domain": "general", "language": ""},
        )
        archive_adjustments, reasons, unsupervised_adjustments, archive_feedback_adjustments = _build_archive_weight_adjustments(
            models=models,
            deployments=deployments,
            prediction_archives=prediction_archives,
            context_features={"domain": "general", "language": "", "scene": ""},
            requested_language="",
            objective=objective,
        )
        decision_feedback_adjustments, decision_feedback_reasons = _build_decision_feedback_weight_adjustments(
            decision_records=decision_records,
            feedback_records=feedback_records,
            objective=objective,
            requested_language="",
            context_features={"domain": "general", "language": "", "scene": ""},
            base_weights=base_weights,
        )
        weight_profiles.append(
            TaskRoutingAdaptiveWeightProfile(
                objective=objective,
                effective_weights=_apply_adaptive_weight_adjustments(
                    base_weights=base_weights,
                    benchmark_adjustments=benchmark_adjustments,
                    archive_adjustments=archive_adjustments,
                    decision_feedback_adjustments=decision_feedback_adjustments,
                ),
                benchmark_adjustments={key: round(value, 4) for key, value in benchmark_adjustments.items() if abs(value) > 0.0001},
                archive_adjustments={
                    key: round(value, 4)
                    for key, value in _sum_weight_adjustments(archive_adjustments, decision_feedback_adjustments).items()
                    if abs(value) > 0.0001
                },
                unsupervised_adjustments={
                    key: round(value, 4)
                    for key, value in unsupervised_adjustments.items()
                    if abs(value) > 0.0001
                },
                feedback_adjustments={
                    key: round(value, 4)
                    for key, value in _sum_weight_adjustments(archive_feedback_adjustments, decision_feedback_adjustments).items()
                    if abs(value) > 0.0001
                },
                reasons=_merge_adaptation_reasons(
                    benchmark_adjustments=benchmark_adjustments,
                    archive_adjustments=archive_adjustments,
                    decision_feedback_adjustments=decision_feedback_adjustments,
                    archive_reasons=reasons,
                    decision_feedback_reasons=decision_feedback_reasons,
                ),
            )
        )

    recent_decisions = []
    for item in sorted(decision_records, key=lambda record: record.created_at, reverse=True)[: max(1, limit)]:
        feedback = feedback_by_decision.get(item.decision_id)
        recent_decisions.append(
            TaskRoutingDecisionSummary(
                decision_id=item.decision_id,
                objective=item.objective,
                inferred_language=item.inferred_language,
                context_domain=str(item.context_features.get("domain") or "general"),
                execution_path=item.execution_path,
                fusion_policy=item.fusion_policy,
                selected_model_ids=[candidate.model_id for candidate in item.selected_deployments],
                fused_triple_count=int(item.fused_result.get("triple_count") or 0),
                feedback_status="completed" if feedback else "pending",
                accepted=feedback.accepted if feedback else None,
                feedback_type=feedback.feedback_type if feedback else None,
                adopted_model_id=feedback.adopted_model_id if feedback else None,
                created_at=item.created_at,
            )
        )

    insights = _build_feedback_insights(
        total_decisions=len(decision_records),
        total_feedback=total_feedback,
        acceptance_rate=acceptance_rate,
        benchmark_count=len([item for item in benchmarks if item.status == "completed"]),
        archive_count=len(prediction_archives),
        archive_feedback_total=archive_satisfied_count + archive_unsatisfied_count,
        objective_summaries=objective_summaries,
        execution_path_summaries=execution_path_summaries,
        model_adoption_summaries=model_adoption_summaries,
    )
    model_score_summaries = _build_model_score_summaries(
        models=models,
        deployments=deployments,
        benchmarks=benchmarks,
        prediction_archives=prediction_archives,
        decision_records=decision_records,
        feedback_records=feedback_records,
    )

    return TaskRoutingFeedbackOverview(
        total_decisions=len(decision_records),
        total_feedback=total_feedback,
        acceptance_rate=acceptance_rate,
        pending_feedback_count=max(len(decision_records) - total_feedback, 0),
        last_feedback_at=last_feedback_at,
        archive_feedback_total=archive_satisfied_count + archive_unsatisfied_count,
        archive_satisfied_count=archive_satisfied_count,
        archive_unsatisfied_count=archive_unsatisfied_count,
        archive_pending_count=archive_pending_count,
        objective_summaries=objective_summaries,
        execution_path_summaries=execution_path_summaries[:8],
        model_adoption_summaries=model_adoption_summaries,
        model_score_summaries=model_score_summaries,
        weight_profiles=weight_profiles,
        recent_decisions=recent_decisions,
        insights=insights,
    )


def _build_feedback_insights(
    *,
    total_decisions: int,
    total_feedback: int,
    acceptance_rate: float | None,
    benchmark_count: int,
    archive_count: int,
    archive_feedback_total: int,
    objective_summaries: list[TaskRoutingObjectiveFeedbackSummary],
    execution_path_summaries: list[TaskRoutingExecutionPathSummary],
    model_adoption_summaries: list[TaskRoutingModelAdoptionSummary],
) -> list[str]:
    insights: list[str] = []
    if total_decisions <= 0:
        return ["当前还没有任务调度历史，先运行一轮策略或补一组 benchmark 后，这里会显示自校准结论。"]
    if benchmark_count > 0 or archive_count > 0:
        insights.append(
            f"当前策略校准主要依赖 {benchmark_count} 条 benchmark 与 {archive_count} 条在线归档证据；归档中已补充 {archive_feedback_total} 条满意度反馈，作为辅助修正信号。"
        )
        insights.append("当前在线学习采用渐进式校准机制：无用户反馈校准更偏稳健，通常需更长样本窗口；有用户反馈校准会更快响应，累计约 10 条同类反馈后更容易观察到明显变化。")
    if objective_summaries:
        best_objective = sorted(
            objective_summaries,
            key=lambda item: (
                -item.decision_count,
                -item.feedback_count,
                item.objective,
            ),
        )[0]
        insights.append(
            f"{best_objective.objective} 目标当前已有 {best_objective.decision_count} 条历史决策，可作为后续调度自校准的重要参考。"
        )
    if execution_path_summaries:
        best_path = sorted(execution_path_summaries, key=lambda item: (-item.decision_count, item.execution_path))[0]
        insights.append(f"{best_path.execution_path} 是当前出现频次较高的执行路径之一，已有 {best_path.decision_count} 条调度记录。")
    return insights[:4]


def _build_task_routing_model_signal_map(
    *,
    models: list[ModelSummary],
    deployments: list[DeploymentRecord],
    benchmarks: list[BenchmarkRecord],
    prediction_archives: list[PredictionArchiveRecord],
    decision_records: list[TaskRoutingDecisionRecord],
    feedback_records: list[TaskRoutingFeedbackRecord],
) -> dict[str, dict[str, Any]]:
    selected_counts: Counter[str] = Counter()
    recent_selected_counts: Counter[str] = Counter()
    feedback_counters: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "feedback_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "adopted_count": 0,
        }
    )
    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    for record in decision_records:
        selected_model_ids = {item.model_id for item in record.selected_deployments if item.model_id}
        record_created_at = _parse_iso_datetime(record.created_at)
        for model_id in selected_model_ids:
            selected_counts[model_id] += 1
            if record_created_at is not None and record_created_at >= recent_cutoff:
                recent_selected_counts[model_id] += 1

    for record in feedback_records:
        selected_model_ids = set(record.selected_model_ids or [])
        if record.adopted_model_id:
            selected_model_ids.add(record.adopted_model_id)
            feedback_counters[record.adopted_model_id]["adopted_count"] += 1
        for model_id in selected_model_ids:
            feedback_counters[model_id]["feedback_count"] += 1
            if record.accepted:
                feedback_counters[model_id]["accepted_count"] += 1
            else:
                feedback_counters[model_id]["rejected_count"] += 1

    model_score_summaries = _build_model_score_summaries(
        models=models,
        deployments=deployments,
        benchmarks=benchmarks,
        prediction_archives=prediction_archives,
        decision_records=decision_records,
        feedback_records=feedback_records,
    )
    scores_by_model: dict[str, list[TaskRoutingModelScoreSummary]] = defaultdict(list)
    for item in model_score_summaries:
        scores_by_model[item.model_id].append(item)

    signals: dict[str, dict[str, Any]] = {}
    model_ids = {item.model_id for item in models}
    for model_id in model_ids:
        model_scores = scores_by_model.get(model_id, [])
        current_scores = [float(item.current_score or 0.0) for item in model_scores]
        dimension_signals: dict[str, float] = defaultdict(float)
        for item in model_scores:
            for breakdown in (
                item.benchmark_score_breakdown,
                item.no_feedback_score_breakdown,
                item.feedback_score_breakdown,
            ):
                for key, value in (breakdown or {}).items():
                    numeric_value = float(value or 0.0)
                    if numeric_value > 0:
                        dimension_signals[key] += numeric_value

        focus_dimensions = [
            key
            for key, value in sorted(
                dimension_signals.items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
            if value > 0.0001
        ]
        feedback_count = int(feedback_counters[model_id]["feedback_count"])
        accepted_count = int(feedback_counters[model_id]["accepted_count"])
        acceptance_rate = (accepted_count / feedback_count) if feedback_count else None
        avg_current_score = _mean_numeric(current_scores)
        selected_count = int(selected_counts.get(model_id, 0))
        recent_selected_count = int(recent_selected_counts.get(model_id, 0))
        adopted_count = int(feedback_counters[model_id]["adopted_count"])
        routing_value_score = (
            min(18.0, recent_selected_count * 4.0 + max(0, selected_count - recent_selected_count) * 1.2)
            + min(12.0, adopted_count * 5.0)
            + min(16.0, float(avg_current_score or 0.0) * 16.0)
            + min(8.0, (acceptance_rate or 0.0) * 8.0 if feedback_count else 0.0)
            + min(10.0, sum(dimension_signals.values()) * 35.0)
        )
        signals[model_id] = {
            "selected_count": selected_count,
            "recent_selected_count": recent_selected_count,
            "feedback_count": feedback_count,
            "accepted_count": accepted_count,
            "rejected_count": int(feedback_counters[model_id]["rejected_count"]),
            "adopted_count": adopted_count,
            "acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
            "avg_current_score": avg_current_score,
            "routing_value_score": round(routing_value_score, 6),
            "dimension_signals": {key: round(value, 6) for key, value in dimension_signals.items() if value > 0.000001},
            "focus_dimensions": focus_dimensions,
        }
    return signals


def _build_automation_history_signal_map(
    automation_actions: list[DeploymentPolicyAutomationAction],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "restart_count": 0.0,
            "busy_restart_count": 0.0,
            "rebalance_count": 0.0,
            "cpu_migration_count": 0.0,
            "failed_count": 0.0,
        }
    )
    for action in automation_actions:
        model_id = str(action.model_id or "").strip()
        if not model_id or model_id == "-":
            continue
        if action.action_type == "restart_high_gpu_memory" and action.status in {"restarted", "failed", "busy"}:
            grouped[model_id]["restart_count"] += 1.0
            if action.status == "busy":
                grouped[model_id]["busy_restart_count"] += 1.0
        if action.action_type == "rebalance_gpu" and action.status in {"migrated", "applied"}:
            grouped[model_id]["rebalance_count"] += 1.0
        if action.action_type == "migrate_to_cpu" and action.status in {"migrated", "applied"}:
            grouped[model_id]["cpu_migration_count"] += 1.0
        if action.status == "failed":
            grouped[model_id]["failed_count"] += 1.0

    signals: dict[str, dict[str, Any]] = {}
    for model_id, values in grouped.items():
        operational_risk_score = min(
            24.0,
            values["restart_count"] * 4.5
            + values["failed_count"] * 4.0
            + values["cpu_migration_count"] * 2.5
            + max(values["rebalance_count"] - 1.0, 0.0) * 0.8
            + values["busy_restart_count"] * 1.2,
        )
        signals[model_id] = {
            "restart_count": int(values["restart_count"]),
            "busy_restart_count": int(values["busy_restart_count"]),
            "rebalance_count": int(values["rebalance_count"]),
            "cpu_migration_count": int(values["cpu_migration_count"]),
            "failed_count": int(values["failed_count"]),
            "operational_risk_score": round(operational_risk_score, 6),
        }
    return signals


def _rank_gpu_devices(gpu_snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    devices = list(gpu_snapshot.get("devices") or [])
    if not gpu_snapshot.get("available") or not devices:
        return ([], [])

    def _device_free_mb(item: dict[str, Any]) -> float:
        return max(float(item.get("memory_total_mb") or 0.0) - float(item.get("memory_used_mb") or 0.0), 0.0)

    ranked = sorted(
        devices,
        key=lambda item: (
            -_device_free_mb(item),
            float(item.get("utilization_gpu_pct") or 0.0),
            len(item.get("active_processes") or []),
        ),
    )
    eligible = []
    for item in ranked:
        total = float(item.get("memory_total_mb") or 0.0)
        free_ratio = (_device_free_mb(item) / total) if total > 0 else 1.0
        utilization = float(item.get("utilization_gpu_pct") or 0.0)
        if free_ratio <= 0.06 or utilization >= 98.0:
            continue
        eligible.append(item)
    if not eligible:
        eligible = ranked
    return (ranked, eligible)


def _pick_gpu_device(
    pool: list[dict[str, Any]],
    *,
    model_id: str,
    prefer_first: bool = False,
) -> dict[str, Any] | None:
    if not pool:
        return None
    if prefer_first:
        return pool[0]
    hash_seed = sum(ord(ch) for ch in str(model_id or ""))
    return pool[hash_seed % len(pool)]


def _apply_cross_layer_policy_adjustments(
    *,
    model: ModelSummary,
    service_tier: str,
    demand_level: str,
    residency_mode: str,
    recommended_profile_name: str,
    recommended_instance_count: int,
    recommended_use_gpu: bool,
    recommended_gpu_ids: list[int],
    recommended_device_label: str,
    gpu_pressure_action: str,
    fallback_profile_name: str,
    fallback_use_gpu: bool,
    fallback_device_label: str,
    cold_start_risk: str,
    gpu_snapshot: dict[str, Any],
    routing_signal: dict[str, Any],
    automation_signal: dict[str, Any],
    collaborative_priority_score: float,
) -> tuple[str, str, str, str, int, bool, list[int], str, str, str, bool, str, list[str], str]:
    routing_value_score = float(routing_signal.get("routing_value_score") or 0.0)
    selected_count = int(routing_signal.get("selected_count") or 0)
    dimension_signals = dict(routing_signal.get("dimension_signals") or {})
    focus_dimensions = list(routing_signal.get("focus_dimensions") or [])
    operational_risk_score = float(automation_signal.get("operational_risk_score") or 0.0)
    restart_count = int(automation_signal.get("restart_count") or 0)
    cpu_migration_count = int(automation_signal.get("cpu_migration_count") or 0)

    latency_priority = float(dimension_signals.get("latency") or 0.0)
    domain_priority = float(dimension_signals.get("domain") or 0.0)
    coverage_priority = float(dimension_signals.get("coverage") or 0.0)

    high_value = collaborative_priority_score >= 26.0 or routing_value_score >= 18.0 or selected_count >= 3
    very_high_value = collaborative_priority_score >= 38.0 or routing_value_score >= 28.0
    high_risk = operational_risk_score >= 8.0 or restart_count >= 2
    very_high_risk = operational_risk_score >= 14.0 or restart_count >= 3

    tags: list[str] = []
    if high_value:
        tags.append("调度高价值")
    if domain_priority >= 0.01:
        tags.append("领域优先")
    elif latency_priority >= 0.01:
        tags.append("时延优先")
    elif coverage_priority >= 0.01:
        tags.append("覆盖补全")
    if restart_count > 0:
        tags.append(f"重启{restart_count}次")
    if cpu_migration_count > 0:
        tags.append("有 CPU 回退史")

    if high_value and service_tier == "cold":
        service_tier = "warm"
        demand_level = "medium"
    if high_value and residency_mode == "on_demand" and cold_start_risk in {"medium", "high"}:
        residency_mode = "keep_warm"
    if very_high_value and recommended_instance_count < 2:
        recommended_instance_count = 2

    ranked, eligible = _rank_gpu_devices(gpu_snapshot)
    prefer_stable_gpu = very_high_value or (high_value and (latency_priority >= 0.01 or domain_priority >= 0.01))
    if high_value and not recommended_use_gpu and ranked and service_tier != "cold":
        selected = _pick_gpu_device(
            eligible[: min(2 if prefer_stable_gpu else 3, len(eligible))] or ranked[:1],
            model_id=model.model_id,
            prefer_first=prefer_stable_gpu,
        )
        if selected is not None:
            recommended_use_gpu = True
            recommended_gpu_ids = [int(selected.get("index"))]
            recommended_device_label = f"GPU {selected.get('index')}（协同优先）"

    if high_risk:
        tags.append("运行风险高")
        if not high_value and service_tier in {"warm", "cold"}:
            recommended_use_gpu = False
            recommended_gpu_ids = []
            recommended_device_label = "CPU（风险降级）"
            recommended_profile_name = "default" if service_tier == "warm" else "memory"
            gpu_pressure_action = "migrate_to_cpu"
            coordination_note = (
                f"任务调度价值一般，但最近存在 {restart_count} 次显存重启 / "
                f"{cpu_migration_count} 次 CPU 回退记录，部署层优先降级到 CPU 以降低抖动。"
            )
        else:
            if ranked and recommended_use_gpu:
                selected = _pick_gpu_device(
                    eligible[: min(2, len(eligible))] or ranked[:1],
                    model_id=model.model_id,
                    prefer_first=True,
                )
                if selected is not None:
                    recommended_gpu_ids = [int(selected.get("index"))]
                    recommended_device_label = f"GPU {selected.get('index')}（高价值稳态）"
            if recommended_profile_name == "memory":
                recommended_profile_name = "default"
            coordination_note = (
                f"该模型在任务调度中价值较高，重点维度为 {', '.join(focus_dimensions or ['balanced'])}，"
                f"但近期存在 {restart_count} 次显存重启记录；部署层保留 GPU，并优先分配到更空闲设备。"
            )
    elif high_value:
        coordination_note = (
            f"任务调度侧显示该模型近期被选中 {selected_count} 次，"
            f"重点维度为 {', '.join(focus_dimensions or ['balanced'])}，部署层上调其保障优先级。"
        )
    else:
        coordination_note = "当前推荐综合了基础需求热度、冷启动风险、GPU 压力与跨层协同信号。"

    if very_high_risk and not high_value:
        fallback_profile_name = "memory"
        fallback_use_gpu = False
        fallback_device_label = "CPU"

    return (
        service_tier,
        demand_level,
        residency_mode,
        recommended_profile_name,
        recommended_instance_count,
        recommended_use_gpu,
        recommended_gpu_ids,
        recommended_device_label,
        gpu_pressure_action,
        fallback_profile_name,
        fallback_use_gpu,
        fallback_device_label,
        tags,
        coordination_note,
    )


def _recommend_device_allocation(
    *,
    model_id: str,
    service_tier: str,
    residency_mode: str,
    cold_start_risk: str,
    package_size_mb: float | None,
    avg_p95_latency_ms: float | None,
    gpu_snapshot: dict[str, Any],
) -> tuple[bool, list[int], str]:
    ranked, eligible = _rank_gpu_devices(gpu_snapshot)
    if not ranked:
        return (False, [], "CPU")

    if service_tier == "hot":
        selected = _pick_gpu_device(eligible[: min(2, len(eligible))] or ranked[:1], model_id=model_id)
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（优先常驻）")
    if service_tier == "warm":
        if residency_mode == "on_demand" and cold_start_risk == "low":
            return (False, [], "CPU")
        selected = _pick_gpu_device(eligible[: min(2, len(eligible))] or ranked[:1], model_id=model_id)
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（共享保温）")
    if cold_start_risk == "high" and (package_size_mb or 0.0) >= 150:
        selected = _pick_gpu_device(eligible[: min(3, len(eligible))] or ranked[:1], model_id=model_id)
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（共享）")
    if avg_p95_latency_ms is not None and avg_p95_latency_ms > 250 and len(ranked) >= 2:
        selected = _pick_gpu_device(eligible[: min(3, len(eligible))] or ranked[:1], model_id=model_id)
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
        tags.append("全局高压迁 CPU")
    elif gpu_pressure_action == "share_gpu":
        tags.append("局部压力先换卡")
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
    cross_layer_priority_count = sum(1 for item in recommendations if item.routing_value_score >= 18.0)
    high_risk_count = sum(1 for item in recommendations if item.operational_risk_score >= 8.0)
    strategy_overview = (
        f"平台当前共分析 {len(recommendations)} 个模型，其中热模型 {hot_count} 个、温模型 {warm_count} 个、冷模型 {cold_count} 个。"
        f"任务调度侧识别出 {cross_layer_priority_count} 个高价值模型，运行侧识别出 {high_risk_count} 个高风险模型；"
        f"建议常驻 {always_on_count} 个、保温 {keep_warm_count} 个、按需启动 {on_demand_count} 个；"
        f"局部 GPU 压力时优先执行换卡重平衡；全局 GPU 高压时有 {cpu_fallback_count} 个低优先级模型建议迁移到 CPU 释放显存。"
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
        "share_gpu": "优先换卡并共享 GPU",
        "migrate_to_cpu": "全局高压时迁移到 CPU",
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
