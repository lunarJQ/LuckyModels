from __future__ import annotations

from pathlib import Path

from .adapters.factory import has_adapter
from .manifest import load_manifest, validate_loaded_manifest
from .schemas import LoadedManifest, ModelDetail, ModelSummary


class ArtifactRegistry:
    def __init__(self, artifacts_root: Path) -> None:
        self.artifacts_root = artifacts_root

    def scan(self) -> list[LoadedManifest]:
        if not self.artifacts_root.exists():
            return []

        results: list[LoadedManifest] = []
        for manifest_path in sorted(self.artifacts_root.rglob("manifest.yaml")):
            loaded = load_manifest(manifest_path)
            validate_loaded_manifest(loaded)
            results.append(loaded)
        return results

    def list_models(self) -> list[ModelSummary]:
        return [self._to_summary(item) for item in self.scan()]

    def get_model(self, model_id: str) -> ModelDetail:
        for item in self.scan():
            if item.manifest.model_id == model_id:
                return self._to_detail(item)
        raise KeyError(f"Model not found: {model_id}")

    @staticmethod
    def _to_summary(item: LoadedManifest) -> ModelSummary:
        manifest = item.manifest
        adapter_available = has_adapter(manifest.runtime_adapter)
        runtime_status = (
            "inference_ready"
            if manifest.predict_ready and adapter_available
            else "package_ready"
        )
        return ModelSummary(
            model_id=manifest.model_id,
            model_name=manifest.model_name,
            model_family=manifest.model_family,
            task_type=manifest.task_type,
            language=manifest.language,
            version=manifest.version,
            framework=manifest.framework,
            backbone=manifest.backbone,
            dataset_name=manifest.dataset_name,
            stage_count=manifest.stage_count,
            predict_ready=manifest.predict_ready,
            adapter_available=adapter_available,
            runtime_status=runtime_status,
            runtime_adapter=manifest.runtime_adapter,
            package_root=item.package_root,
        )

    def _to_detail(self, item: LoadedManifest) -> ModelDetail:
        return ModelDetail(
            summary=self._to_summary(item),
            manifest=item.manifest,
            resolved_artifacts=item.resolved_artifacts,
            resolved_profiles=item.resolved_profiles,
            resolved_tokenizer=item.resolved_tokenizer,
            resolved_labels=item.resolved_labels,
        )
