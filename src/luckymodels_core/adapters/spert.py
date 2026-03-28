from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter
from .bert_utils import (
    load_json,
    load_vocab,
    provider_list,
    span_text_from_offsets,
    tokenize_with_wordpiece_vocab,
)


class SpERTAdapter(BaseAdapter):
    adapter_name = "spert_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.entity_session = None
        self.relation_session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.vocab: dict[str, int] = {}
        self.entity_labels: list[str] = []
        self.relation_labels: list[str] = []

    def load(self) -> None:
        import onnxruntime as ort  # type: ignore

        types_payload = load_json(self.manifest.resolved_labels.get("schema_path")) or {}
        entity_types = types_payload.get("entities", {})
        relation_types = types_payload.get("relations", {})
        self.entity_labels = list(entity_types.keys())
        self.relation_labels = list(relation_types.keys())
        self.vocab = load_vocab(self.manifest.resolved_tokenizer["vocab_path"])
        self.providers = provider_list(self.profile.device)
        self.entity_session = ort.InferenceSession(
            self.manifest.resolved_artifacts["entity_encoder"],
            providers=self.providers,
        )
        self.relation_session = ort.InferenceSession(
            self.manifest.resolved_artifacts["relation_head"],
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
            "predict_ready": True,
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
            "entity_label_count": len(self.entity_labels),
            "relation_label_count": len(self.relation_labels),
            "artifact_count": len(self.manifest.resolved_artifacts),
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if (
            not self.sessions_loaded
            or self.entity_session is None
            or self.relation_session is None
        ):
            raise RuntimeError("SpERT runtime is not loaded")

        started = perf_counter()
        options = options or {}
        max_length = int(options.get("max_length") or self.manifest.manifest.inference.max_length)
        max_span_size = int(options.get("max_span_size", 10))
        relation_threshold = float(options.get("relation_threshold", 0.4))
        tokenized = tokenize_with_wordpiece_vocab(
            text,
            self.vocab,
            max_length=max_length,
            do_lower_case=True,
        )

        spans = self._enumerate_spans(tokenized.valid_start, tokenized.valid_end, max_span_size)
        entity_masks = np.zeros((1, len(spans), len(tokenized.input_ids)), dtype=bool)
        entity_sizes = np.zeros((1, len(spans)), dtype=np.int64)
        span_tensor = []
        for index, (start_idx, end_idx) in enumerate(spans):
            entity_masks[0, index, start_idx : end_idx + 1] = True
            entity_sizes[0, index] = end_idx - start_idx + 1
            span_tensor.append((start_idx, end_idx))

        entity_probs, sequence_output, entity_spans_pool, size_embeddings = self.entity_session.run(
            None,
            {
                "encodings": np.array([tokenized.input_ids], dtype=np.int64),
                "context_masks": np.array([tokenized.attention_mask], dtype=bool),
                "entity_masks": entity_masks,
                "entity_sizes": entity_sizes,
            },
        )

        entity_probs = entity_probs[0]
        predicted_entities: list[dict[str, Any]] = []
        for index, probs in enumerate(entity_probs):
            label_id = int(np.argmax(probs))
            if label_id <= 0 or label_id > len(self.entity_labels):
                continue
            start_idx, end_idx = span_tensor[index]
            entity_text = span_text_from_offsets(text, tokenized.offsets, start_idx, end_idx)
            if not entity_text:
                continue
            predicted_entities.append(
                {
                    "span_index": index,
                    "start": start_idx,
                    "end": end_idx,
                    "text": entity_text,
                    "label": self.entity_labels[label_id - 1],
                    "confidence": float(probs[label_id]),
                }
            )

        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        if len(predicted_entities) >= 2:
            pooled = entity_spans_pool[:, [item["span_index"] for item in predicted_entities], :]
            sized = size_embeddings[:, [item["span_index"] for item in predicted_entities], :]
            relations: list[tuple[int, int]] = []
            rel_masks = []
            for subject_index, subject in enumerate(predicted_entities):
                for object_index, obj in enumerate(predicted_entities):
                    if subject_index == object_index:
                        continue
                    relations.append((subject_index, object_index))
                    rel_masks.append(
                        self._relation_mask(
                            len(tokenized.input_ids),
                            subject["start"],
                            subject["end"],
                            obj["start"],
                            obj["end"],
                        )
                    )

            rel_probs = self.relation_session.run(
                None,
                {
                    "sequence_output": sequence_output.astype(np.float32),
                    "entity_spans_pool": pooled.astype(np.float32),
                    "size_embeddings": sized.astype(np.float32),
                    "relations": np.array([relations], dtype=np.int64),
                    "rel_masks": np.array([rel_masks], dtype=bool),
                },
            )[0][0]

            for relation_index, (subject_index, object_index) in enumerate(relations):
                scores = rel_probs[relation_index]
                for label_index, score in enumerate(scores):
                    if label_index >= len(self.relation_labels):
                        continue
                    if float(score) <= relation_threshold:
                        continue
                    subject = predicted_entities[subject_index]
                    obj = predicted_entities[object_index]
                    predicate = self.relation_labels[label_index]
                    triple_key = (subject["text"], predicate, obj["text"])
                    if triple_key in seen:
                        continue
                    seen.add(triple_key)
                    triples.append(
                        {
                            "subject": subject["text"],
                            "predicate": predicate,
                            "object": obj["text"],
                            "confidence": round(
                                (subject["confidence"] + obj["confidence"] + float(score)) / 3,
                                6,
                            ),
                            "subject_span": [subject["start"], subject["end"]],
                            "object_span": [obj["start"], obj["end"]],
                            "evidence": text,
                        }
                    )

        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        result = {
            "model_id": self.manifest.manifest.model_id,
            "model_name": self.manifest.manifest.model_name,
            "version": self.manifest.manifest.version,
            "language": self.manifest.manifest.language,
            "latency_ms": elapsed_ms,
            "text": text,
            "triple_count": len(triples),
            "triples": triples,
            "warnings": [],
        }
        if options.get("debug") or options.get("return_debug"):
            result["debug"] = {
                "entity_candidate_count": len(spans),
                "predicted_entity_count": len(predicted_entities),
                "tokens": tokenized.tokens,
            }
        return result

    @staticmethod
    def _enumerate_spans(
        valid_start: int,
        valid_end: int,
        max_span_size: int,
    ) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for start_idx in range(valid_start, valid_end):
            upper = min(valid_end, start_idx + max_span_size)
            for end_idx in range(start_idx, upper):
                spans.append((start_idx, end_idx))
        return spans

    @staticmethod
    def _relation_mask(
        sequence_length: int,
        subject_start: int,
        subject_end: int,
        object_start: int,
        object_end: int,
    ) -> np.ndarray:
        mask = np.zeros(sequence_length, dtype=bool)
        left_end = min(subject_end, object_end)
        right_start = max(subject_start, object_start)
        if left_end < right_start:
            mask[left_end : right_start + 1] = True
        else:
            mask[min(subject_start, object_start) : max(subject_end, object_end) + 1] = True
        return mask
