from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from ..schemas import LoadedManifest, ProfileConfig


class BaseAdapter(ABC):
    adapter_name: str = "base"

    def __init__(self, manifest: LoadedManifest, profile: ProfileConfig) -> None:
        self.manifest = manifest
        self.profile = profile

    def effective_batch_size(self, options: dict[str, Any] | None = None) -> int:
        options = options or {}
        if options.get("batch_size") is not None:
            return max(1, int(options["batch_size"]))
        return max(1, int(self.profile.execution.get("batch_size", self.manifest.manifest.inference.batch_size)))

    def predict_many(self, texts: list[str], options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not texts:
            raise ValueError("texts must not be empty")

        started = perf_counter()
        items = [self.predict(text, options) for text in texts]
        latency_ms = round((perf_counter() - started) * 1000, 3)
        first = items[0]
        return {
            "model_id": first.get("model_id", self.manifest.manifest.model_id),
            "model_name": first.get("model_name", self.manifest.manifest.model_name),
            "version": first.get("version", self.manifest.manifest.version),
            "language": first.get("language", self.manifest.manifest.language),
            "latency_ms": latency_ms,
            "batch_size": len(texts),
            "item_count": len(items),
            "total_triple_count": sum(int(item.get("triple_count", 0) or 0) for item in items),
            "items": items,
            "warnings": _merge_warnings(items),
        }

    @abstractmethod
    def load(self) -> None:
        """Load model-family-specific assets."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return runtime health information."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return runtime metadata for the loaded package."""

    @abstractmethod
    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run inference and return a uniform result."""


def _merge_warnings(items: list[dict[str, Any]]) -> list[str]:
    merged: list[str] = []
    for item in items:
        warnings = item.get("warnings") or []
        for warning in warnings:
            text = str(warning)
            if text not in merged:
                merged.append(text)
    return merged
