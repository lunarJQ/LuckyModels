from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .base import BaseAdapter


class CasRelAdapter(BaseAdapter):
    adapter_name = "casrel_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.relation_map: Any = None
        self.vocab_size: int | None = None
        self.session_backend: str = "unloaded"
        self.sessions_loaded = False

    def load(self) -> None:
        relation_map_path = self.manifest.resolved_labels.get("relation_map")
        if relation_map_path:
            with Path(relation_map_path).open("r", encoding="utf-8") as handle:
                self.relation_map = json.load(handle)

        vocab_path = self.manifest.resolved_tokenizer.get("vocab_path")
        if vocab_path:
            with Path(vocab_path).open("r", encoding="utf-8") as handle:
                self.vocab_size = sum(1 for _ in handle)

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            self.session_backend = "onnxruntime-not-installed"
            self.sessions_loaded = False
            return

        # The full decode path still needs to be verified against exported model I/O.
        # For the first milestone we validate that the ONNX sessions are loadable.
        _ = ort.InferenceSession(self.manifest.resolved_artifacts["subject_model"])
        _ = ort.InferenceSession(self.manifest.resolved_artifacts["object_model"])
        self.session_backend = "onnxruntime"
        self.sessions_loaded = True

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.adapter_name,
            "model_id": self.manifest.manifest.model_id,
            "sessions_loaded": self.sessions_loaded,
            "session_backend": self.session_backend,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "model_id": self.manifest.manifest.model_id,
            "model_name": self.manifest.manifest.model_name,
            "version": self.manifest.manifest.version,
            "language": self.manifest.manifest.language,
            "runtime_adapter": self.manifest.manifest.runtime_adapter,
            "vocab_size": self.vocab_size,
            "artifact_count": len(self.manifest.resolved_artifacts),
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        started = perf_counter()

        if not self.sessions_loaded:
            raise RuntimeError(
                "CasRel runtime is not fully available. Install the runtime dependencies "
                "and verify the ONNX decode path before enabling live inference."
            )

        # The exported CasRel package is already standardized, but the exact ONNX output
        # names and decode behavior still need to be verified against the exported model.
        # We expose a controlled error here instead of returning misleading fake triples.
        raise NotImplementedError(
            f"CasRel predict() is not finished yet for text length {len(text)}. "
            f"Elapsed {round((perf_counter() - started) * 1000, 3)} ms."
        )
