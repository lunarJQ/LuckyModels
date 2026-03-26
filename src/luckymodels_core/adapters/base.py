from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import LoadedManifest, ProfileConfig


class BaseAdapter(ABC):
    adapter_name: str = "base"

    def __init__(self, manifest: LoadedManifest, profile: ProfileConfig) -> None:
        self.manifest = manifest
        self.profile = profile

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
