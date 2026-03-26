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
    backbone: str
    artifacts: dict[str, str]
    tokenizer: TokenizerConfig
    labels: LabelsConfig
    inference: InferenceConfig
    profiles: dict[str, str]


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
    language: str
    version: str
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
    notes: str | None = None


class DeploymentRecord(BaseModel):
    deployment_id: str
    model_id: str
    profile_name: str
    runtime: str
    node_name: str
    gpu_ids: list[int] = Field(default_factory=list)
    status: str = "planned"
    service_url: str | None = None
    created_at: str = Field(default_factory=utc_now)
    notes: str | None = None


class SystemSummary(BaseModel):
    model_count: int
    deployment_count: int
    artifact_root: str
