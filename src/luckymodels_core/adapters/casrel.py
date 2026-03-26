from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter


class CasRelAdapter(BaseAdapter):
    adapter_name = "casrel_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.relation_map: Any = None
        self.id2relation: dict[int, str] = {}
        self.relation2id: dict[str, int] = {}
        self.vocab: dict[str, int] = {}
        self.vocab_size: int | None = None
        self.session_backend: str = "unloaded"
        self.providers: list[str] = []
        self.sessions_loaded = False
        self.tokenizer_backend = "char-vocab"
        self.tokenizer = None
        self.subject_session = None
        self.object_session = None

    def load(self) -> None:
        relation_map_path = self.manifest.resolved_labels.get("relation_map")
        if relation_map_path:
            with Path(relation_map_path).open("r", encoding="utf-8") as handle:
                self.relation_map = json.load(handle)
            if (
                isinstance(self.relation_map, list)
                and len(self.relation_map) == 2
                and isinstance(self.relation_map[0], dict)
                and isinstance(self.relation_map[1], dict)
            ):
                self.id2relation = {
                    int(key): value for key, value in self.relation_map[0].items()
                }
                self.relation2id = {
                    str(key): int(value) for key, value in self.relation_map[1].items()
                }

        vocab_path = self.manifest.resolved_tokenizer.get("vocab_path")
        if vocab_path:
            with Path(vocab_path).open("r", encoding="utf-8") as handle:
                for idx, token in enumerate(handle):
                    self.vocab[token.rstrip("\n")] = idx
            self.vocab_size = len(self.vocab)
            try:
                from transformers import BertTokenizer  # type: ignore

                self.tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
                self.tokenizer_backend = "bert-tokenizer"
            except Exception:
                self.tokenizer = None
                self.tokenizer_backend = "char-vocab"

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError:
            self.session_backend = "onnxruntime-not-installed"
            self.sessions_loaded = False
            return

        available = ort.get_available_providers()
        if self.profile.device.lower().startswith("cuda") and "CUDAExecutionProvider" in available:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

        self.subject_session = ort.InferenceSession(
            self.manifest.resolved_artifacts["subject_model"],
            providers=self.providers,
        )
        self.object_session = ort.InferenceSession(
            self.manifest.resolved_artifacts["object_model"],
            providers=self.providers,
        )
        self.session_backend = "onnxruntime"
        self.sessions_loaded = True

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "adapter": self.adapter_name,
            "model_id": self.manifest.manifest.model_id,
            "sessions_loaded": self.sessions_loaded,
            "session_backend": self.session_backend,
            "providers": self.providers,
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
            "relation_count": len(self.id2relation),
            "tokenizer_backend": self.tokenizer_backend,
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        started = perf_counter()

        if not self.sessions_loaded:
            raise RuntimeError(
                "CasRel runtime is not fully available. Install the runtime dependencies "
                "and verify the ONNX decode path before enabling live inference."
            )

        options = options or {}
        prepared = self._prepare_inputs(text)
        subject_threshold = self._resolve_threshold("subject", options)
        object_threshold = self._resolve_threshold("object", options)

        subject_outputs = self.subject_session.run(
            None,
            {
                "tokens": prepared["tokens"],
                "segments": prepared["segments"],
            },
        )
        pred_sub_heads, pred_sub_tails = subject_outputs
        subject_spans = self._decode_subjects(
            pred_sub_heads[0],
            pred_sub_tails[0],
            prepared["text_length"],
            subject_threshold,
        )

        triples = []
        seen = set()
        for subject_start, subject_end in subject_spans:
            object_outputs = self.object_session.run(
                None,
                {
                    "tokens": prepared["tokens"],
                    "segments": prepared["segments"],
                    "sub_head": np.array([[subject_start]], dtype=np.int32),
                    "sub_tail": np.array([[subject_end]], dtype=np.int32),
                },
            )
            pred_obj_heads, pred_obj_tails = object_outputs
            triples.extend(
                self._decode_objects(
                    text=prepared["normalized_text"],
                    token_ids=prepared["token_ids"],
                    pred_obj_heads=pred_obj_heads[0],
                    pred_obj_tails=pred_obj_tails[0],
                    subject_span=(subject_start, subject_end),
                    text_length=prepared["text_length"],
                    threshold=object_threshold,
                    seen=seen,
                )
            )

        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        return {
            "model_id": self.manifest.manifest.model_id,
            "model_name": self.manifest.manifest.model_name,
            "version": self.manifest.manifest.version,
            "language": self.manifest.manifest.language,
            "latency_ms": elapsed_ms,
            "text": prepared["normalized_text"],
            "triple_count": len(triples),
            "triples": triples,
        }

    def _prepare_inputs(self, text: str) -> dict[str, Any]:
        max_length = int(
            self.profile.execution.get(
                "max_length", self.manifest.manifest.inference.max_length
            )
        )
        normalized_text = text[:max_length]
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                normalized_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            token_ids = encoded["input_ids"]
            segments = encoded.get("token_type_ids", [0] * len(token_ids))
            attention_mask = encoded["attention_mask"]
            attention_length = int(sum(attention_mask))
        else:
            char_tokens = list(normalized_text)
            unk_id = self.vocab.get("[UNK]", 100)
            pad_id = self.vocab.get("[PAD]", 0)
            token_ids = [self.vocab.get(token, unk_id) for token in char_tokens]
            attention_length = len(token_ids)
            if attention_length < max_length:
                token_ids.extend([pad_id] * (max_length - attention_length))
            segments = [0] * max_length

        return {
            "normalized_text": normalized_text,
            "tokens": np.array([token_ids], dtype=np.int32),
            "segments": np.array([segments], dtype=np.int32),
            "text_length": attention_length,
            "token_ids": token_ids,
        }

    def _resolve_threshold(self, name: str, options: dict[str, Any]) -> float:
        option_value = options.get(f"{name}_threshold")
        if option_value is not None:
            return float(option_value)

        thresholds = self.manifest.manifest.inference.thresholds
        manifest_value = getattr(thresholds, name, None) if thresholds else None
        return float(manifest_value if manifest_value is not None else 0.5)

    @staticmethod
    def _decode_subjects(
        pred_sub_heads: np.ndarray,
        pred_sub_tails: np.ndarray,
        text_length: int,
        threshold: float,
    ) -> list[tuple[int, int]]:
        subjects: list[tuple[int, int]] = []
        for start_idx in range(text_length):
            if float(pred_sub_heads[start_idx][0]) > threshold:
                for end_idx in range(start_idx, text_length):
                    if float(pred_sub_tails[end_idx][0]) > threshold:
                        subjects.append((start_idx, end_idx))
                        break
        return subjects

    def _decode_objects(
        self,
        text: str,
        token_ids: list[int],
        pred_obj_heads: np.ndarray,
        pred_obj_tails: np.ndarray,
        subject_span: tuple[int, int],
        text_length: int,
        threshold: float,
        seen: set[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        subject_start, subject_end = subject_span
        subject_text = self._span_to_text(text, token_ids, subject_start, subject_end)
        triples: list[dict[str, Any]] = []

        relation_count = pred_obj_heads.shape[1]
        for relation_id in range(relation_count):
            for object_start in range(text_length):
                start_score = float(pred_obj_heads[object_start][relation_id])
                if start_score <= threshold:
                    continue
                for object_end in range(object_start, text_length):
                    end_score = float(pred_obj_tails[object_end][relation_id])
                    if end_score <= threshold:
                        continue
                    object_text = self._span_to_text(text, token_ids, object_start, object_end)
                    if not subject_text or not object_text:
                        continue
                    predicate = self.id2relation.get(relation_id, str(relation_id))
                    triple_key = (subject_text, predicate, object_text)
                    if triple_key in seen:
                        break
                    seen.add(triple_key)
                    triples.append(
                        {
                            "subject": subject_text,
                            "predicate": predicate,
                            "object": object_text,
                            "confidence": round((start_score + end_score) / 2, 6),
                            "subject_span": [subject_start, subject_end],
                            "object_span": [object_start, object_end],
                            "evidence": text,
                        }
                    )
                    break

        return triples

    def _span_to_text(
        self,
        text: str,
        token_ids: list[int],
        start_idx: int,
        end_idx: int,
    ) -> str:
        if self.tokenizer is not None:
            decoded = self.tokenizer.decode(
                token_ids[start_idx : end_idx + 1],
                skip_special_tokens=True,
            )
            return decoded.replace(" ", "").strip()
        return text[start_idx : end_idx + 1]
