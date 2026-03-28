from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter
from .bert_utils import (
    decode_bio_spans,
    create_inference_session,
    infer_do_lower_case,
    load_vocab,
    load_relation_maps,
    provider_list,
    sigmoid,
    softmax,
    span_text_from_offsets,
    tokenize_with_wordpiece_vocab,
)


class PRGCAdapter(BaseAdapter):
    adapter_name = "prgc_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.stage1_session = None
        self.stage2_session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.id2relation: dict[int, str] = {}
        self.relation2id: dict[str, int] = {}
        self.vocab: dict[str, int] = {}

    def load(self) -> None:
        import onnxruntime as ort  # type: ignore

        self.id2relation, self.relation2id = load_relation_maps(
            self.manifest.resolved_labels.get("relation_map")
        )
        self.vocab = load_vocab(self.manifest.resolved_tokenizer["vocab_path"])
        self.providers = provider_list(self.profile.device)
        self.stage1_session = create_inference_session(
            self.manifest.resolved_artifacts["stage1_model"],
            self.providers,
            self.profile,
        )
        self.stage2_session = create_inference_session(
            self.manifest.resolved_artifacts["stage2_model"],
            self.providers,
            self.profile,
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
        if (
            not self.sessions_loaded
            or self.stage1_session is None
            or self.stage2_session is None
        ):
            raise RuntimeError("PRGC runtime is not loaded")

        started = perf_counter()
        options = options or {}
        max_length = int(options.get("max_length") or self.manifest.manifest.inference.max_length)
        rel_threshold = float(options.get("relation_threshold", options.get("rel_threshold", 0.5)))
        corres_threshold = float(options.get("corres_threshold", 0.5))
        tokenized = tokenize_with_wordpiece_vocab(
            text,
            self.vocab,
            max_length=max_length,
            do_lower_case=infer_do_lower_case(self.manifest.manifest.backbone),
        )

        sequence_output, rel_logits, corres_logits, corres_mask = self.stage1_session.run(
            None,
            {
                "input_ids": np.array([tokenized.input_ids], dtype=np.int64),
                "attention_mask": np.array([tokenized.attention_mask], dtype=np.int64),
            },
        )

        rel_scores = sigmoid(rel_logits[0])
        selected_relations = np.where(rel_scores > rel_threshold)[0].tolist()
        if not selected_relations and rel_scores.size:
            selected_relations = [int(np.argmax(rel_scores))]

        corres_scores = sigmoid(corres_logits[0])
        corres_mask = corres_mask[0].astype(bool)
        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for relation_id in selected_relations[:32]:
            output_sub, output_obj = self.stage2_session.run(
                None,
                {
                    "sequence_output": sequence_output.astype(np.float32),
                    "potential_rels": np.array([relation_id], dtype=np.int64),
                },
            )
            sub_probs = np.apply_along_axis(softmax, -1, output_sub[0])
            obj_probs = np.apply_along_axis(softmax, -1, output_obj[0])
            sub_tags = np.argmax(output_sub[0], axis=-1)
            obj_tags = np.argmax(output_obj[0], axis=-1)
            subject_spans = decode_bio_spans(sub_tags, tokenized.valid_start, tokenized.valid_end)
            object_spans = decode_bio_spans(obj_tags, tokenized.valid_start, tokenized.valid_end)
            predicate = self.id2relation.get(relation_id, str(relation_id))

            for subject_span in subject_spans:
                for object_span in object_spans:
                    subject_head = subject_span[0]
                    object_head = object_span[0]
                    if not corres_mask[subject_head, object_head]:
                        continue
                    if float(corres_scores[subject_head, object_head]) <= corres_threshold:
                        continue
                    subject_text = span_text_from_offsets(
                        text,
                        tokenized.offsets,
                        subject_span[0],
                        subject_span[1],
                    )
                    object_text = span_text_from_offsets(
                        text,
                        tokenized.offsets,
                        object_span[0],
                        object_span[1],
                    )
                    if not subject_text or not object_text:
                        continue
                    triple_key = (subject_text, predicate, object_text)
                    if triple_key in seen:
                        continue
                    seen.add(triple_key)
                    confidence = (
                        float(rel_scores[relation_id])
                        + float(corres_scores[subject_head, object_head])
                        + float(sub_probs[subject_head, 1:].max(initial=0.0))
                        + float(obj_probs[object_head, 1:].max(initial=0.0))
                    ) / 4
                    triples.append(
                        {
                            "subject": subject_text,
                            "predicate": predicate,
                            "object": object_text,
                            "confidence": round(confidence, 6),
                            "subject_span": list(subject_span),
                            "object_span": list(object_span),
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
                "selected_relations": [
                    {
                        "relation_id": relation_id,
                        "predicate": self.id2relation.get(relation_id, str(relation_id)),
                        "score": round(float(rel_scores[relation_id]), 6),
                    }
                    for relation_id in selected_relations[:32]
                ],
                "valid_range": [tokenized.valid_start, tokenized.valid_end],
                "tokens": tokenized.tokens,
            }
        return result
