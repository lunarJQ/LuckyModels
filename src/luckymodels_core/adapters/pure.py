from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from pathlib import Path

from .base import BaseAdapter
from .bert_utils import (
    create_inference_session,
    load_json,
    load_vocab,
    provider_list,
    softmax,
    span_text_from_offsets,
    tokenize_with_wordpiece_vocab,
)


class PUREAdapter(BaseAdapter):
    adapter_name = "pure_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.entity_session = None
        self.relation_session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.entity_vocab: dict[str, int] = {}
        self.relation_vocab: dict[str, int] = {}
        self.entity_labels: list[str] = []
        self.relation_labels: list[str] = []
        self.special_tokens: dict[str, str] = {}

    def load(self) -> None:
        import onnxruntime as ort  # type: ignore

        relation_config_path = Path(self.manifest.resolved_labels["relation_map"]).resolve().parent
        self.entity_vocab = load_vocab(self.manifest.resolved_tokenizer["vocab_path"])
        self.relation_vocab = load_vocab(
            str(relation_config_path / "vocab.txt")
        )
        self.entity_labels = load_json(self.manifest.resolved_labels.get("entity_map")) or []
        self.relation_labels = load_json(self.manifest.resolved_labels.get("relation_map")) or []
        self.special_tokens = load_json(str(relation_config_path / "special_tokens.json")) or {}
        self.providers = provider_list(self.profile)
        self.entity_session = create_inference_session(
            self.manifest.resolved_artifacts["entity_model"],
            self.providers,
            self.profile,
        )
        self.relation_session = create_inference_session(
            self.manifest.resolved_artifacts["relation_model"],
            self.providers,
            self.profile,
        )
        self.providers = list(self.entity_session.get_providers())
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
            "execution": self.profile.execution,
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
            raise RuntimeError("PURE runtime is not loaded")

        started = perf_counter()
        options = options or {}
        max_length = int(options.get("max_length") or self.manifest.manifest.inference.max_length)
        max_span_length = int(options.get("max_span_length", 8))
        entity_tokenized = tokenize_with_wordpiece_vocab(
            text,
            self.entity_vocab,
            max_length=max_length,
            do_lower_case=True,
        )

        spans = self._enumerate_spans(entity_tokenized.valid_start, entity_tokenized.valid_end, max_span_length)
        span_tensor = np.array(
            [[[start_idx, end_idx, end_idx - start_idx + 1] for start_idx, end_idx in spans]],
            dtype=np.int64,
        )
        span_mask = np.ones((1, len(spans)), dtype=np.int64)
        entity_logits = self.entity_session.run(
            None,
            {
                "input_ids": np.array([entity_tokenized.input_ids], dtype=np.int64),
                "attention_mask": np.array([entity_tokenized.attention_mask], dtype=np.int64),
                "spans": span_tensor,
                "spans_mask": span_mask,
            },
        )[0][0]

        predicted_entities: list[dict[str, Any]] = []
        for index, logits in enumerate(entity_logits):
            label_id = int(np.argmax(logits))
            if label_id <= 0 or label_id >= len(self.entity_labels):
                continue
            start_idx, end_idx = spans[index]
            entity_text = span_text_from_offsets(text, entity_tokenized.offsets, start_idx, end_idx)
            if not entity_text:
                continue
            probs = softmax(logits)
            predicted_entities.append(
                {
                    "start": start_idx,
                    "end": end_idx,
                    "text": entity_text,
                    "label": self.entity_labels[label_id],
                    "confidence": float(probs[label_id]),
                }
            )

        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for subject in predicted_entities:
            for obj in predicted_entities:
                if subject is obj:
                    continue
                relation_inputs = self._build_relation_inputs(
                    entity_tokenized.tokens,
                    subject,
                    obj,
                    max_length=max_length,
                )
                if relation_inputs is None:
                    continue
                relation_logits = self.relation_session.run(
                    None,
                    {
                        "input_ids": np.array([relation_inputs["input_ids"]], dtype=np.int64),
                        "token_type_ids": np.array([relation_inputs["token_type_ids"]], dtype=np.int64),
                        "attention_mask": np.array([relation_inputs["attention_mask"]], dtype=np.int64),
                        "sub_idx": np.array([relation_inputs["sub_idx"]], dtype=np.int64),
                        "obj_idx": np.array([relation_inputs["obj_idx"]], dtype=np.int64),
                    },
                )[0][0]
                relation_id = int(np.argmax(relation_logits))
                if relation_id <= 0 or relation_id >= len(self.relation_labels):
                    continue
                predicate = self.relation_labels[relation_id]
                triple_key = (subject["text"], predicate, obj["text"])
                if triple_key in seen:
                    continue
                seen.add(triple_key)
                probs = softmax(relation_logits)
                triples.append(
                    {
                        "subject": subject["text"],
                        "predicate": predicate,
                        "object": obj["text"],
                        "confidence": round(
                            (subject["confidence"] + obj["confidence"] + float(probs[relation_id])) / 3,
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
                "predicted_entity_count": len(predicted_entities),
                "tokens": entity_tokenized.tokens,
            }
        return result

    @staticmethod
    def _enumerate_spans(valid_start: int, valid_end: int, max_span_length: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for start_idx in range(valid_start, valid_end):
            upper = min(valid_end, start_idx + max_span_length)
            for end_idx in range(start_idx, upper):
                spans.append((start_idx, end_idx))
        return spans

    def _build_relation_inputs(
        self,
        base_tokens: list[str],
        subject: dict[str, Any],
        obj: dict[str, Any],
        max_length: int,
    ) -> dict[str, Any] | None:
        tokens: list[str] = []
        sub_idx = -1
        obj_idx = -1
        for index, token in enumerate(base_tokens):
            if index == subject["start"]:
                marker = self.special_tokens.get(
                    f"SUBJ_START={subject['label']}",
                    self.special_tokens.get("SUBJ_START", "[unused1]"),
                )
                sub_idx = len(tokens)
                tokens.append(marker)
            if index == obj["start"]:
                marker = self.special_tokens.get(
                    f"OBJ_START={obj['label']}",
                    self.special_tokens.get("OBJ_START", "[unused3]"),
                )
                obj_idx = len(tokens)
                tokens.append(marker)

            tokens.append(token)

            if index == subject["end"]:
                tokens.append(
                    self.special_tokens.get(
                        f"SUBJ_END={subject['label']}",
                        self.special_tokens.get("SUBJ_END", "[unused2]"),
                    )
                )
            if index == obj["end"]:
                tokens.append(
                    self.special_tokens.get(
                        f"OBJ_END={obj['label']}",
                        self.special_tokens.get("OBJ_END", "[unused4]"),
                    )
                )

        if sub_idx < 0 or obj_idx < 0 or len(tokens) > max_length:
            return None
        input_ids = [self.relation_vocab.get(token, self.relation_vocab.get("[UNK]", 100)) for token in tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sub_idx": sub_idx,
            "obj_idx": obj_idx,
        }
