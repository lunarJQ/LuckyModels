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


class DeploymentCreateRequest(BaseModel):
    model_id: str
    profile_name: str = "default"
    runtime: str = "runtime-service"
    node_name: str = "local-node"
    gpu_ids: list[int] = Field(default_factory=list)
    launch_runtime: bool = False
    notes: str | None = None


class DeploymentRecord(BaseModel):
    deployment_id: str
    model_id: str
    profile_name: str
    runtime: str
    node_name: str
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
