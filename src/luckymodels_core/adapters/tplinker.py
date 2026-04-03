from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter
from .bert_utils import (
    build_handshaking_index,
    create_inference_session,
    load_bert_tokenizer,
    load_relation_maps,
    provider_list,
    softmax,
    span_text_from_offsets,
    tokenize_with_offsets,
)


class TPLinkerAdapter(BaseAdapter):
    adapter_name = "tplinker_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.id2relation: dict[int, str] = {}
        self.relation2id: dict[str, int] = {}
        self.tokenizer = None

    def load(self) -> None:
        import onnxruntime as ort  # type: ignore

        self.id2relation, self.relation2id = load_relation_maps(
            self.manifest.resolved_labels.get("relation_map")
        )
        self.tokenizer = load_bert_tokenizer(
            self.manifest.resolved_tokenizer["vocab_path"],
            self.manifest.resolved_tokenizer.get("tokenizer_path"),
            self.manifest.manifest.backbone,
        )
        self.providers = provider_list(self.profile)
        self.session = create_inference_session(
            self.manifest.resolved_artifacts["model"],
            self.providers,
            self.profile,
        )
        self.providers = list(self.session.get_providers())
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
            "relation_count": len(self.id2relation),
            "artifact_count": len(self.manifest.resolved_artifacts),
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.sessions_loaded or self.session is None or self.tokenizer is None:
            raise RuntimeError("TPLinker runtime is not loaded")

        started = perf_counter()
        options = options or {}
        max_length = int(options.get("max_length") or self.manifest.manifest.inference.max_length)
        tokenized = tokenize_with_offsets(self.tokenizer, text, max_length=max_length)
        ent_logits, head_rel_logits, tail_rel_logits = self.session.run(
            None,
            {
                "input_ids": np.array([tokenized.input_ids], dtype=np.int64),
                "attention_mask": np.array([tokenized.attention_mask], dtype=np.int64),
                "token_type_ids": np.array([tokenized.token_type_ids], dtype=np.int64),
            },
        )

        ent_logits = ent_logits[0]
        head_rel_logits = head_rel_logits[0]
        tail_rel_logits = tail_rel_logits[0]
        ent_tags = np.argmax(ent_logits, axis=-1)
        head_tags = np.argmax(head_rel_logits, axis=-1)
        tail_tags = np.argmax(tail_rel_logits, axis=-1)

        shaking_spots, spot_to_index = build_handshaking_index(len(tokenized.input_ids))
        entities_by_start: dict[int, list[dict[str, Any]]] = {}
        entities: list[dict[str, Any]] = []
        for shaking_index, tag in enumerate(ent_tags):
            if int(tag) != 1:
                continue
            start_idx, end_idx = shaking_spots[shaking_index]
            if not (tokenized.valid_start <= start_idx <= end_idx < tokenized.valid_end):
                continue
            span_text = span_text_from_offsets(text, tokenized.offsets, start_idx, end_idx)
            if not span_text:
                continue
            entity = {
                "start": start_idx,
                "end": end_idx,
                "text": span_text,
                "confidence": float(softmax(ent_logits[shaking_index])[1]),
            }
            entities.append(entity)
            entities_by_start.setdefault(start_idx, []).append(entity)

        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation_id in range(head_tags.shape[0]):
            predicate = self.id2relation.get(relation_id, str(relation_id))
            for shaking_index, head_tag in enumerate(head_tags[relation_id]):
                head_tag = int(head_tag)
                if head_tag not in {1, 2}:
                    continue
                left_idx, right_idx = shaking_spots[shaking_index]
                if not (tokenized.valid_start <= left_idx < tokenized.valid_end):
                    continue
                if not (tokenized.valid_start <= right_idx < tokenized.valid_end):
                    continue
                if head_tag == 1:
                    subject_start, object_start = left_idx, right_idx
                else:
                    subject_start, object_start = right_idx, left_idx

                subject_candidates = entities_by_start.get(subject_start, [])
                object_candidates = entities_by_start.get(object_start, [])
                if not subject_candidates or not object_candidates:
                    continue

                for subject in subject_candidates:
                    for obj in object_candidates:
                        tail_pair = tuple(sorted((subject["end"], obj["end"])))
                        tail_index = spot_to_index.get(tail_pair)
                        if tail_index is None:
                            continue
                        expected_tail_tag = 1 if subject["end"] <= obj["end"] else 2
                        if int(tail_tags[relation_id, tail_index]) != expected_tail_tag:
                            continue
                        triple_key = (subject["text"], predicate, obj["text"])
                        if triple_key in seen:
                            continue
                        seen.add(triple_key)
                        head_conf = float(softmax(head_rel_logits[relation_id, shaking_index])[head_tag])
                        tail_conf = float(softmax(tail_rel_logits[relation_id, tail_index])[expected_tail_tag])
                        triples.append(
                            {
                                "subject": subject["text"],
                                "predicate": predicate,
                                "object": obj["text"],
                                "confidence": round(
                                    (subject["confidence"] + obj["confidence"] + head_conf + tail_conf) / 4,
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
                "valid_range": [tokenized.valid_start, tokenized.valid_end],
                "tokens": tokenized.tokens,
                "entity_candidate_count": len(entities),
            }
        return result
