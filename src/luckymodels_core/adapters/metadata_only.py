from __future__ import annotations

from typing import Any

from .base import BaseAdapter


class MetadataOnlyAdapter(BaseAdapter):
    adapter_name = "metadata_only_adapter"

    def load(self) -> None:
        """No-op adapter used for packaged-but-not-yet-runtime-ready models."""

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.adapter_name,
            "model_id": self.manifest.manifest.model_id,
            "sessions_loaded": False,
            "session_backend": "metadata-only",
            "providers": [],
            "predict_ready": False,
            "deployment_requirements": self.manifest.manifest.deployment_requirements,
            "runtime_notes": self.manifest.manifest.runtime_notes,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "model_id": self.manifest.manifest.model_id,
            "model_name": self.manifest.manifest.model_name,
            "version": self.manifest.manifest.version,
            "language": self.manifest.manifest.language,
            "runtime_adapter": self.manifest.manifest.runtime_adapter,
            "dataset_name": self.manifest.manifest.dataset_name,
            "stage_count": self.manifest.manifest.stage_count,
            "predict_ready": False,
            "artifact_count": len(self.manifest.resolved_artifacts),
            "deployment_requirements": self.manifest.manifest.deployment_requirements,
            "runtime_notes": self.manifest.manifest.runtime_notes,
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError(
            "This model package is registered in LuckyModels, but live predict is not "
            "implemented yet. The ONNX export still requires family-specific Python "
            "preprocessing and decoding."
        )
