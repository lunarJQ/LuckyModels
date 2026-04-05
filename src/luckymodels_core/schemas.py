from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenizerConfig(BaseModel):
    type: str
    vocab_path: str | None = None
    tokenizer_path: str | None = None


class LabelsConfig(BaseModel):
    relation_map: str | None = None
    entity_map: str | None = None
    schema_path: str | None = None


class ThresholdConfig(BaseModel):
    subject: float | None = None
    object: float | None = None


class InferenceConfig(BaseModel):
    max_length: int = 256
    batch_size: int = 1
    thresholds: ThresholdConfig | None = None
    relation_threshold: float | None = None
    entity_threshold: float | None = None
    max_span_size: int | None = None
    max_entity_candidates: int | None = None
    do_lower_case: bool | None = None


class ManifestModel(BaseModel):
    model_id: str
    model_name: str
    model_family: str
    task_type: str
    language: str
    framework: str
    runtime_adapter: str
    version: str
    dataset_name: str | None = None
    dataset_split: str | None = None
    stage_count: int = 1
    predict_ready: bool = True
    backbone: str
    artifacts: dict[str, str]
    tokenizer: TokenizerConfig
    labels: LabelsConfig
    inference: InferenceConfig
    profiles: dict[str, str]
    deployment_requirements: list[str] = Field(default_factory=list)
    runtime_notes: list[str] = Field(default_factory=list)


class LoadedManifest(BaseModel):
    manifest_path: str
    package_root: str
    manifest: ManifestModel
    resolved_artifacts: dict[str, str]
    resolved_profiles: dict[str, str]
    resolved_tokenizer: dict[str, str]
    resolved_labels: dict[str, str]


class ProfileConfig(BaseModel):
    name: str
    runtime: str
    device: str = "cpu"
    gpu_ids: list[int] = Field(default_factory=list)
    execution: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)


class ModelSummary(BaseModel):
    model_id: str
    model_name: str
    model_family: str
    task_type: str
    language: str
    version: str
    framework: str
    backbone: str
    dataset_name: str | None = None
    stage_count: int = 1
    predict_ready: bool = True
    adapter_available: bool = True
    runtime_status: str = "inference_ready"
    runtime_adapter: str
    package_root: str
    package_origin: str = "builtin"
    package_origin_label: str = "本机自带"
    deletable: bool = False


class ModelDetail(BaseModel):
    summary: ModelSummary
    manifest: ManifestModel
    resolved_artifacts: dict[str, str]
    resolved_profiles: dict[str, str]
    resolved_tokenizer: dict[str, str]
    resolved_labels: dict[str, str]
    package_metadata: dict[str, Any] = Field(default_factory=dict)


class ModelImportModeOption(BaseModel):
    code: str
    label: str
    note: str


class ModelImportFamilyOption(BaseModel):
    code: str
    label: str
    default_language: str | None = None
    note: str = ""


class ModelImportOptions(BaseModel):
    modes: list[ModelImportModeOption] = Field(default_factory=list)
    families: list[ModelImportFamilyOption] = Field(default_factory=list)


class ModelImportResult(BaseModel):
    model_id: str
    model_name: str
    model_family: str
    language: str
    dataset_name: str | None = None
    version: str
    import_mode: str
    package_root: str
    archive_filename: str | None = None
    detected_files: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ModelOptimizationOption(BaseModel):
    code: str
    label: str
    alias_suffix: str
    note: str


class ModelOptimizationOptions(BaseModel):
    options: list[ModelOptimizationOption] = Field(default_factory=list)


class ModelOptimizeRequest(BaseModel):
    optimization_type: str
    alias: str | None = None


class ModelOptimizationResult(BaseModel):
    model_id: str
    model_name: str
    model_family: str
    language: str
    dataset_name: str | None = None
    version: str
    package_root: str
    source_model_id: str
    optimization_type: str
    optimization_label: str
    optimized_artifact_count: int = 0
    optimized_artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ModelDeleteResult(BaseModel):
    model_id: str
    deleted_deployment_count: int = 0
    stopped_runtime_count: int = 0
    deleted_benchmark_count: int = 0
    deleted_prediction_archive_count: int = 0
    deleted_task_routing_decision_count: int = 0
    deleted_task_routing_feedback_count: int = 0
    deleted_log_file_count: int = 0
    removed_package_root: str


class CapabilityMatrixRow(BaseModel):
    model_id: str
    model_name: str
    model_family: str
    language: str
    dataset_name: str | None = None
    runtime_status: str
    inference_ready: bool
    supports_online_predict: bool
    supports_benchmark: bool
    supports_batch_predict: bool
    profile_count: int = 0
    benchmark_count: int = 0
    deployment_count: int = 0
    latest_benchmark_id: str | None = None
    latest_deployment_id: str | None = None
    known_limitations: str = ""


class DeploymentCreateRequest(BaseModel):
    model_id: str
    profile_name: str = "default"
    runtime: str = "runtime-service"
    node_name: str = "local-node"
    use_gpu: bool = True
    gpu_ids: list[int] = Field(default_factory=list)
    provider_preference: str | None = None
    launch_runtime: bool = False
    notes: str | None = None


class DeploymentRecord(BaseModel):
    deployment_id: str
    model_id: str
    profile_name: str
    runtime: str
    node_name: str
    use_gpu: bool = True
    gpu_ids: list[int] = Field(default_factory=list)
    provider_preference: str | None = None
    profile_runtime: str | None = None
    device: str | None = None
    execution: dict[str, Any] = Field(default_factory=dict)
    health: dict[str, Any] | None = None
    status: str = "planned"
    service_url: str | None = None
    manifest_path: str | None = None
    profile_path: str | None = None
    log_path: str | None = None
    pid: int | None = None
    runtime_revision: int = 1
    revision_started_at: str | None = None
    previous_revision_started_at: str | None = None
    revision_change_level: str = "initial"
    runtime_signature: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    notes: str | None = None


class SystemSummary(BaseModel):
    model_count: int
    deployment_count: int
    benchmark_count: int = 0
    prediction_archive_count: int = 0
    user_count: int = 0
    admin_count: int = 0
    artifact_root: str
    logs_root: str | None = None


class GpuPoolState(BaseModel):
    auto_detect_all: bool = True
    configured_gpu_ids: list[int] = Field(default_factory=list)
    visible_gpu_ids: list[int] = Field(default_factory=list)
    effective_gpu_ids: list[int] = Field(default_factory=list)
    disabled_gpu_ids: list[int] = Field(default_factory=list)
    unavailable_configured_gpu_ids: list[int] = Field(default_factory=list)
    updated_at: str | None = None


class GpuPoolUpdateRequest(BaseModel):
    allowed_gpu_ids: list[int] = Field(default_factory=list)
    run_policy_optimization: bool = True


class BenchmarkCreateRequest(BaseModel):
    deployment_id: str
    text: str
    iterations: int = 10
    warmup_runs: int = 1
    timeout_seconds: float = 30.0
    label: str | None = None
    scene: str | None = None
    trigger_source: str = "manual"
    options: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRecord(BaseModel):
    benchmark_id: str
    deployment_id: str
    model_id: str
    profile_name: str
    service_url: str
    label: str | None = None
    scene: str | None = None
    trigger_source: str = "manual"
    benchmark_scope_key: str | None = None
    deployment_signature: str | None = None
    deployment_revision: int | None = None
    revision_started_at: str | None = None
    runtime_signature: str | None = None
    is_effective: bool = True
    superseded_by: str | None = None
    request_text: str
    iterations: int
    warmup_runs: int
    timeout_seconds: float
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    throughput_qps: float | None = None
    remote_avg_latency_ms: float | None = None
    status: str = "planned"
    sample_response: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now)


class PredictionArchiveRecord(BaseModel):
    archive_id: str
    deployment_id: str
    model_id: str
    profile_name: str
    archive_source: str = "deployment_predict"
    service_url: str | None = None
    scene: str | None = None
    request_text: str | None = None
    request_texts: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    triple_count: int | None = None
    item_count: int | None = None
    response_payload: dict[str, Any] = Field(default_factory=dict)
    task_routing_decision_id: str | None = None
    task_routing_objective: str | None = None
    task_routing_execution_path: str | None = None
    task_routing_fusion_policy: str | None = None
    task_routing_weight_snapshot: dict[str, float] = Field(default_factory=dict)
    task_routing_adjustment_snapshot: dict[str, float] = Field(default_factory=dict)
    task_routing_unsupervised_adjustment_snapshot: dict[str, float] = Field(default_factory=dict)
    task_routing_feedback_adjustment_snapshot: dict[str, float] = Field(default_factory=dict)
    task_routing_selected_model_ids: list[str] = Field(default_factory=list)
    deployment_revision: int | None = None
    revision_started_at: str | None = None
    runtime_signature: str | None = None
    feedback_status: str = "pending"
    feedback_note: str | None = None
    feedback_updated_at: str | None = None
    created_at: str = Field(default_factory=utc_now)


class PredictionArchiveFeedbackUpdateRequest(BaseModel):
    feedback_status: str = "pending"
    feedback_note: str | None = None


class TaskRoutingRequest(BaseModel):
    text: str
    objective: str = "accuracy"
    language: str | None = None
    scene: str | None = None
    max_models: int | None = None
    model_family: str | None = None
    candidate_model_ids: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class TaskRoutingCandidate(BaseModel):
    deployment_id: str
    model_id: str
    profile_name: str
    language: str
    service_url: str | None = None
    selection_score: float | None = None
    selection_reason: str = ""
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    estimated_p95_latency_ms: float | None = None
    benchmark_success_rate: float | None = None
    recent_request_count: int = 0
    predicted_triple_count: int | None = None
    domain_match: str = "general"
    resource_cost_level: str = "medium"


class TaskRoutingExecution(BaseModel):
    deployment_id: str
    model_id: str
    profile_name: str
    selected_for_fusion: bool = True
    execution_role: str = "primary"
    latency_ms: float | None = None
    triple_count: int = 0
    triples: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TaskRoutingResponse(BaseModel):
    decision_id: str | None = None
    archive_id: str | None = None
    objective: str
    requested_language: str | None = None
    inferred_language: str
    difficulty: str
    strategy_mode: str
    execution_path: str = ""
    fusion_policy: str = ""
    explanation: str
    context_features: dict[str, Any] = Field(default_factory=dict)
    objective_weights: dict[str, float] = Field(default_factory=dict)
    adaptive_adjustments: dict[str, float] = Field(default_factory=dict)
    adaptive_unsupervised_adjustments: dict[str, float] = Field(default_factory=dict)
    adaptive_feedback_adjustments: dict[str, float] = Field(default_factory=dict)
    adaptation_reasons: list[str] = Field(default_factory=list)
    decision_trace: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    selected_count: int = 0
    selected_deployments: list[TaskRoutingCandidate] = Field(default_factory=list)
    execution_results: list[TaskRoutingExecution] = Field(default_factory=list)
    fused_result: dict[str, Any] = Field(default_factory=dict)


class TaskRoutingDecisionRecord(BaseModel):
    decision_id: str
    request_text: str
    scene: str | None = None
    objective: str
    requested_language: str | None = None
    inferred_language: str
    difficulty: str
    strategy_mode: str
    execution_path: str = ""
    fusion_policy: str = ""
    context_features: dict[str, Any] = Field(default_factory=dict)
    objective_weights: dict[str, float] = Field(default_factory=dict)
    adaptive_adjustments: dict[str, float] = Field(default_factory=dict)
    adaptive_unsupervised_adjustments: dict[str, float] = Field(default_factory=dict)
    adaptive_feedback_adjustments: dict[str, float] = Field(default_factory=dict)
    adaptation_reasons: list[str] = Field(default_factory=list)
    selected_deployments: list[TaskRoutingCandidate] = Field(default_factory=list)
    execution_results: list[TaskRoutingExecution] = Field(default_factory=list)
    fused_result: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class TaskRoutingFeedbackSubmitRequest(BaseModel):
    decision_id: str
    accepted: bool = True
    feedback_type: str = "accepted"
    adopted_model_id: str | None = None
    note: str | None = None


class TaskRoutingFeedbackRecord(BaseModel):
    feedback_id: str
    decision_id: str
    accepted: bool = True
    feedback_type: str = "accepted"
    adopted_model_id: str | None = None
    objective: str
    inferred_language: str
    context_domain: str = "general"
    execution_path: str = ""
    selected_model_ids: list[str] = Field(default_factory=list)
    note: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class TaskRoutingDecisionSummary(BaseModel):
    decision_id: str
    objective: str
    inferred_language: str
    context_domain: str = "general"
    execution_path: str = ""
    fusion_policy: str = ""
    selected_model_ids: list[str] = Field(default_factory=list)
    fused_triple_count: int = 0
    feedback_status: str = "pending"
    accepted: bool | None = None
    feedback_type: str | None = None
    adopted_model_id: str | None = None
    created_at: str = Field(default_factory=utc_now)


class TaskRoutingObjectiveFeedbackSummary(BaseModel):
    objective: str
    decision_count: int = 0
    feedback_count: int = 0
    acceptance_rate: float | None = None
    best_execution_path: str | None = None


class TaskRoutingExecutionPathSummary(BaseModel):
    execution_path: str
    decision_count: int = 0
    feedback_count: int = 0
    acceptance_rate: float | None = None


class TaskRoutingModelAdoptionSummary(BaseModel):
    model_id: str
    adoption_count: int = 0
    adopted_share: float | None = None


class TaskRoutingModelScoreSummary(BaseModel):
    objective: str
    scene: str
    deployment_id: str
    model_id: str
    profile_name: str
    language: str
    dataset_name: str | None = None
    baseline_score: float = 0.0
    current_score: float = 0.0
    score_delta: float = 0.0
    benchmark_score_delta: float = 0.0
    no_feedback_score_delta: float = 0.0
    feedback_score_delta: float = 0.0
    benchmark_score_breakdown: dict[str, float] = Field(default_factory=dict)
    no_feedback_score_breakdown: dict[str, float] = Field(default_factory=dict)
    feedback_score_breakdown: dict[str, float] = Field(default_factory=dict)
    performance_score_delta: float = 0.0
    weight_shift_score_delta: float = 0.0
    performance_score_breakdown: dict[str, float] = Field(default_factory=dict)
    weight_shift_score_breakdown: dict[str, float] = Field(default_factory=dict)
    baseline_breakdown: dict[str, float] = Field(default_factory=dict)
    current_breakdown: dict[str, float] = Field(default_factory=dict)
    performance_delta_breakdown: dict[str, float] = Field(default_factory=dict)
    dominant_dimension: str | None = None
    dominant_delta: float | None = None
    change_note: str | None = None


class TaskRoutingAdaptiveWeightProfile(BaseModel):
    objective: str
    effective_weights: dict[str, float] = Field(default_factory=dict)
    benchmark_adjustments: dict[str, float] = Field(default_factory=dict)
    archive_adjustments: dict[str, float] = Field(default_factory=dict)
    unsupervised_adjustments: dict[str, float] = Field(default_factory=dict)
    feedback_adjustments: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class TaskRoutingFeedbackOverview(BaseModel):
    total_decisions: int = 0
    total_feedback: int = 0
    acceptance_rate: float | None = None
    pending_feedback_count: int = 0
    last_feedback_at: str | None = None
    archive_feedback_total: int = 0
    archive_satisfied_count: int = 0
    archive_unsatisfied_count: int = 0
    archive_pending_count: int = 0
    objective_summaries: list[TaskRoutingObjectiveFeedbackSummary] = Field(default_factory=list)
    execution_path_summaries: list[TaskRoutingExecutionPathSummary] = Field(default_factory=list)
    model_adoption_summaries: list[TaskRoutingModelAdoptionSummary] = Field(default_factory=list)
    model_score_summaries: list[TaskRoutingModelScoreSummary] = Field(default_factory=list)
    weight_profiles: list[TaskRoutingAdaptiveWeightProfile] = Field(default_factory=list)
    recent_decisions: list[TaskRoutingDecisionSummary] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)


class DeploymentPolicyRecommendation(BaseModel):
    model_id: str
    model_name: str
    language: str
    demand_level: str
    demand_score: float
    base_demand_score: float = 0.0
    routing_value_score: float = 0.0
    operational_risk_score: float = 0.0
    collaborative_priority_score: float = 0.0
    service_tier: str = "cold"
    residency_mode: str = "on_demand"
    recent_request_count: int = 0
    total_request_count: int = 0
    benchmark_count: int = 0
    avg_p95_latency_ms: float | None = None
    benchmark_success_rate: float | None = None
    running_deployment_count: int = 0
    active_deployment_ids: list[str] = Field(default_factory=list)
    package_size_mb: float | None = None
    stage_count: int = 1
    estimated_cold_start_ms: float | None = None
    cold_start_risk: str = "unknown"
    recommended_profile_name: str
    recommended_instance_count: int = 1
    recommended_use_gpu: bool = True
    recommended_gpu_ids: list[int] = Field(default_factory=list)
    recommended_device_label: str = ""
    gpu_pressure_action: str = "retain_gpu"
    fallback_profile_name: str = "memory"
    fallback_use_gpu: bool = False
    fallback_device_label: str = "CPU"
    routing_selected_count: int = 0
    routing_feedback_count: int = 0
    routing_adopted_count: int = 0
    routing_acceptance_rate: float | None = None
    avg_routing_current_score: float | None = None
    routing_dimension_signals: dict[str, float] = Field(default_factory=dict)
    routing_focus_dimensions: list[str] = Field(default_factory=list)
    restart_history_count: int = 0
    rebalance_history_count: int = 0
    cpu_migration_history_count: int = 0
    failed_action_history_count: int = 0
    policy_tags: list[str] = Field(default_factory=list)
    coordination_note: str = ""
    rationale: str


class DeploymentPolicySummary(BaseModel):
    total_models: int = 0
    recent_window_days: int = 7
    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    always_on_count: int = 0
    keep_warm_count: int = 0
    on_demand_count: int = 0
    gpu_recommended_count: int = 0
    cpu_recommended_count: int = 0
    cpu_fallback_count: int = 0
    busiest_model_id: str | None = None
    strategy_overview: str = ""


class DeploymentPolicyAutomationAction(BaseModel):
    created_at: str = Field(default_factory=utc_now)
    action_type: str
    model_id: str
    deployment_id: str | None = None
    status: str = "planned"
    detail: str = ""


class DeploymentPolicyAutomationState(BaseModel):
    enabled: bool = True
    interval_seconds: int = 900
    protect_active_window_seconds: int = 120
    gpu_memory_restart_enabled: bool = True
    gpu_memory_restart_interval_seconds: int = 60
    gpu_memory_restart_threshold_mb: int = 5120
    last_run_at: str | None = None
    last_gpu_memory_restart_at: str | None = None
    last_policy_generated_at: str | None = None
    last_status: str = "idle"
    run_count: int = 0
    gpu_memory_restart_run_count: int = 0
    last_summary: DeploymentPolicySummary = Field(default_factory=DeploymentPolicySummary)
    recent_actions: list[DeploymentPolicyAutomationAction] = Field(default_factory=list)


class DeploymentPolicyAutomationUpdateRequest(BaseModel):
    enabled: bool = True
    interval_seconds: int = 900
    protect_active_window_seconds: int = 120
    gpu_memory_restart_enabled: bool = True
    gpu_memory_restart_interval_seconds: int = 60
    gpu_memory_restart_threshold_mb: int = 5120


class GpuPoolUpdateResult(BaseModel):
    state: GpuPoolState = Field(default_factory=GpuPoolState)
    previous_effective_gpu_ids: list[int] = Field(default_factory=list)
    enabled_gpu_ids: list[int] = Field(default_factory=list)
    disabled_gpu_ids: list[int] = Field(default_factory=list)
    affected_deployment_ids: list[str] = Field(default_factory=list)
    migrated_deployment_ids: list[str] = Field(default_factory=list)
    cpu_fallback_deployment_ids: list[str] = Field(default_factory=list)
    busy_deployment_ids: list[str] = Field(default_factory=list)
    skipped_deployment_ids: list[str] = Field(default_factory=list)
    actions: list[DeploymentPolicyAutomationAction] = Field(default_factory=list)
    policy_run_triggered: bool = False
    detail: str = ""


class DeploymentPolicyResponse(BaseModel):
    generated_at: str = Field(default_factory=utc_now)
    gpu_available: bool = False
    summary: DeploymentPolicySummary = Field(default_factory=DeploymentPolicySummary)
    recommendations: list[DeploymentPolicyRecommendation] = Field(default_factory=list)
    gpu_snapshot: dict[str, Any] = Field(default_factory=dict)
    gpu_pool_state: GpuPoolState = Field(default_factory=GpuPoolState)


class AccessPermission(BaseModel):
    permission_id: str
    permission_name: str
    category: str
    description: str


class AccessRole(BaseModel):
    role_id: str
    role_name: str
    description: str
    permissions: list[str] = Field(default_factory=list)
    builtin: bool = True


class AccessUser(BaseModel):
    user_id: str
    username: str
    display_name: str
    email: str
    role_ids: list[str] = Field(default_factory=list)
    is_active: bool = True
    is_admin: bool = False
    notes: str | None = None
    password_hash: str | None = Field(default=None, exclude=True)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class AccessUserCreateRequest(BaseModel):
    username: str
    display_name: str
    email: str
    password: str | None = None
    role_ids: list[str] = Field(default_factory=list)
    is_active: bool = True
    notes: str | None = None


class AccessUserUpdateRequest(BaseModel):
    display_name: str | None = None
    email: str | None = None
    role_ids: list[str] | None = None
    is_active: bool | None = None
    notes: str | None = None


class AccessUserToggleRequest(BaseModel):
    is_active: bool


class AccessCurrentUserSwitchRequest(BaseModel):
    user_id: str


class AccessUserPasswordResetRequest(BaseModel):
    password: str


class AccessUserPasswordResetResponse(BaseModel):
    user_id: str
    username: str
    reset_at: str = Field(default_factory=utc_now)


class AccessOverview(BaseModel):
    current_user: AccessUser | None = None
    current_permissions: list[str] = Field(default_factory=list)
    users: list[AccessUser] = Field(default_factory=list)
    roles: list[AccessRole] = Field(default_factory=list)
    permissions: list[AccessPermission] = Field(default_factory=list)


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class AuthSessionResponse(BaseModel):
    authenticated: bool = False
    current_user: AccessUser | None = None
    current_permissions: list[str] = Field(default_factory=list)
