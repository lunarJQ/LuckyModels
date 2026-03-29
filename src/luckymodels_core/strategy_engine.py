from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any
from urllib.request import Request, urlopen

from .schemas import (
    BenchmarkRecord,
    DeploymentPolicyRecommendation,
    DeploymentPolicyResponse,
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
    running_counts = Counter(
        item.model_id for item in deployments if item.status in {"running", "starting"}
    )
    demand_scores: dict[str, float] = {}
    for model in models:
        archive_count = len(archives_by_model.get(model.model_id, []))
        benchmark_items = benchmark_by_model.get(model.model_id, [])
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
        demand_scores[model.model_id] = (
            archive_count * 6.0
            + running_count * 4.0
            + len(benchmark_items) * 1.5
            + ((avg_success or 0.0) * 10.0)
            + (5.0 if avg_p95 is not None and avg_p95 < 80 else 0.0)
        )

    sorted_scores = sorted(demand_scores.values())
    high_bar = _quantile(sorted_scores, 0.67) if sorted_scores else 0.0
    medium_bar = _quantile(sorted_scores, 0.34) if sorted_scores else 0.0

    for model in models:
        benchmark_items = benchmark_by_model.get(model.model_id, [])
        archive_items = archives_by_model.get(model.model_id, [])
        running_count = running_counts.get(model.model_id, 0)
        score = demand_scores.get(model.model_id, 0.0)
        if score >= high_bar and score > 0:
            demand_level = "high"
        elif score >= medium_bar and score > 0:
            demand_level = "medium"
        else:
            demand_level = "low"

        avg_p95 = _mean_numeric(item.p95_latency_ms for item in benchmark_items)
        if demand_level == "high":
            recommended_profile = "latency"
        elif demand_level == "medium":
            recommended_profile = "default"
        else:
            recommended_profile = "memory"

        recommended_use_gpu, recommended_gpu_ids, device_label = _recommend_device_allocation(
            demand_level=demand_level,
            gpu_snapshot=gpu_snapshot,
        )
        rationale = build_deployment_rationale(
            demand_level=demand_level,
            archive_count=len(archive_items),
            running_count=running_count,
            avg_p95_latency_ms=avg_p95,
            device_label=device_label,
        )
        recommendations.append(
            DeploymentPolicyRecommendation(
                model_id=model.model_id,
                model_name=model.model_name,
                language=model.language,
                demand_level=demand_level,
                demand_score=round(score, 3),
                recent_request_count=len(archive_items),
                benchmark_count=len(benchmark_items),
                avg_p95_latency_ms=avg_p95,
                running_deployment_count=running_count,
                recommended_profile_name=recommended_profile,
                recommended_use_gpu=recommended_use_gpu,
                recommended_gpu_ids=recommended_gpu_ids,
                recommended_device_label=device_label,
                rationale=rationale,
            )
        )

    recommendations.sort(key=lambda item: (-item.demand_score, item.model_id))
    return DeploymentPolicyResponse(
        gpu_available=bool(gpu_snapshot.get("available")),
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
    demand_level: str,
    archive_count: int,
    running_count: int,
    avg_p95_latency_ms: float | None,
    device_label: str,
) -> str:
    if demand_level == "high":
        return (
            f"该模型近期请求较多，且需要保持更低尾延迟，建议部署到 {device_label}，"
            "优先分配空闲显存更高、当前负载更低的 GPU。"
        )
    if demand_level == "medium":
        return (
            f"该模型当前处于中等需求等级，近期请求数为 {archive_count}，"
            f"建议以默认配置运行在 {device_label}，兼顾资源占用与可用性。"
        )
    latency_text = f"{avg_p95_latency_ms:.3f} ms" if avg_p95_latency_ms is not None else "暂无"
    return (
        f"该模型近期请求较少，当前运行实例数为 {running_count}，历史 P95 为 {latency_text}，"
        f"建议部署到 {device_label} 或共享低优先级资源。"
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
    with urlopen(request, timeout=90) as response:  # noqa: S310
        result = json.loads(response.read().decode("utf-8"))
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
    demand_level: str,
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
    if demand_level == "high":
        selected = ranked[0]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}")
    if demand_level == "medium":
        selected = ranked[min(1, len(ranked) - 1)]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}")
    if len(ranked) >= 3:
        selected = ranked[-1]
        return (True, [int(selected.get("index"))], f"GPU {selected.get('index')}（共享）")
    return (False, [], "CPU")


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
