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


class ModelDetail(BaseModel):
    summary: ModelSummary
    manifest: ManifestModel
    resolved_artifacts: dict[str, str]
    resolved_profiles: dict[str, str]
    resolved_tokenizer: dict[str, str]
    resolved_labels: dict[str, str]


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
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    notes: str | None = None


class SystemSummary(BaseModel):
    model_count: int
    deployment_count: int
    benchmark_count: int = 0
    prediction_archive_count: int = 0
    artifact_root: str
    logs_root: str | None = None


class BenchmarkCreateRequest(BaseModel):
    deployment_id: str
    text: str
    iterations: int = 10
    warmup_runs: int = 1
    timeout_seconds: float = 30.0
    label: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRecord(BaseModel):
    benchmark_id: str
    deployment_id: str
    model_id: str
    profile_name: str
    service_url: str
    label: str | None = None
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
    service_url: str | None = None
    scene: str | None = None
    request_text: str | None = None
    request_texts: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float | None = None
    triple_count: int | None = None
    item_count: int | None = None
    response_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class TaskRoutingRequest(BaseModel):
    text: str
    objective: str = "accuracy"
    language: str | None = None
    scene: str | None = None
    max_models: int = 3
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
    estimated_p95_latency_ms: float | None = None
    benchmark_success_rate: float | None = None
    recent_request_count: int = 0
    predicted_triple_count: int | None = None


class TaskRoutingExecution(BaseModel):
    deployment_id: str
    model_id: str
    profile_name: str
    selected_for_fusion: bool = True
    latency_ms: float | None = None
    triple_count: int = 0
    triples: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TaskRoutingResponse(BaseModel):
    objective: str
    requested_language: str | None = None
    inferred_language: str
    difficulty: str
    strategy_mode: str
    explanation: str
    candidate_count: int = 0
    selected_count: int = 0
    selected_deployments: list[TaskRoutingCandidate] = Field(default_factory=list)
    execution_results: list[TaskRoutingExecution] = Field(default_factory=list)
    fused_result: dict[str, Any] = Field(default_factory=dict)


class DeploymentPolicyRecommendation(BaseModel):
    model_id: str
    model_name: str
    language: str
    demand_level: str
    demand_score: float
    recent_request_count: int = 0
    benchmark_count: int = 0
    avg_p95_latency_ms: float | None = None
    running_deployment_count: int = 0
    recommended_profile_name: str
    recommended_use_gpu: bool = True
    recommended_gpu_ids: list[int] = Field(default_factory=list)
    recommended_device_label: str = ""
    rationale: str


class DeploymentPolicyResponse(BaseModel):
    generated_at: str = Field(default_factory=utc_now)
    gpu_available: bool = False
    recommendations: list[DeploymentPolicyRecommendation] = Field(default_factory=list)
    gpu_snapshot: dict[str, Any] = Field(default_factory=dict)
