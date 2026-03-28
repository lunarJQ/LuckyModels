from __future__ import annotations

import unicodedata
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter
from .bert_utils import create_inference_session, load_relation_maps, provider_list, softmax


class OneRelAdapter(BaseAdapter):
    adapter_name = "onerel_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.id2relation: dict[int, str] = {}
        self.relation2id: dict[str, int] = {}
        self.id2tag: dict[int, str] = {}
        self.tag2id: dict[str, int] = {}
        self.vocab: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    def load(self) -> None:
        import onnxruntime as ort  # type: ignore

        self.id2relation, self.relation2id = load_relation_maps(
            self.manifest.resolved_labels.get("relation_map")
        )
        tag_payload = self._load_json(self.manifest.resolved_labels.get("schema_path"))
        if (
            isinstance(tag_payload, list)
            and len(tag_payload) == 2
            and isinstance(tag_payload[0], dict)
            and isinstance(tag_payload[1], dict)
        ):
            self.id2tag = {int(key): str(value) for key, value in tag_payload[0].items()}
            self.tag2id = {str(key): int(value) for key, value in tag_payload[1].items()}

        vocab_path = self.manifest.resolved_tokenizer["vocab_path"]
        with Path(vocab_path).open("r", encoding="utf-8") as handle:
            for index, token in enumerate(handle):
                clean = token.rstrip("\n")
                self.vocab[clean] = index
                self.id2token[index] = clean

        self.providers = provider_list(self.profile.device)
        self.session = create_inference_session(
            self.manifest.resolved_artifacts["model"],
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
            "tag_count": len(self.id2tag),
            "artifact_count": len(self.manifest.resolved_artifacts),
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.sessions_loaded or self.session is None:
            raise RuntimeError("OneRel runtime is not loaded")

        started = perf_counter()
        options = options or {}
        max_length = int(options.get("max_length") or self.manifest.manifest.inference.max_length)
        prepared = self._prepare_inputs(text, max_length=max_length)
        tag_logits, tag_predictions = self.session.run(
            None,
            {
                "token_ids": np.array([prepared["token_ids"]], dtype=np.int64),
                "mask": np.array([prepared["mask"]], dtype=np.int64),
            },
        )
        tag_logits = tag_logits[0]
        tag_predictions = tag_predictions[0]

        hb_tb = self.tag2id.get("HB-TB", 1)
        hb_te = self.tag2id.get("HB-TE", 2)
        he_te = self.tag2id.get("HE-TE", 3)

        triples: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for relation_id in range(tag_predictions.shape[0]):
            rel_matrix = tag_predictions[relation_id]
            predicate = self.id2relation.get(relation_id, str(relation_id))
            head_pairs = np.argwhere(rel_matrix == hb_tb)
            for subject_head, object_head in head_pairs.tolist():
                if not self._in_valid_range(subject_head, prepared) or not self._in_valid_range(object_head, prepared):
                    continue
                object_tail_candidates = np.where(rel_matrix[subject_head] == hb_te)[0]
                for object_tail in object_tail_candidates.tolist():
                    if object_tail < object_head or not self._in_valid_range(object_tail, prepared):
                        continue
                    subject_tail_candidates = np.where(rel_matrix[:, object_tail] == he_te)[0]
                    for subject_tail in subject_tail_candidates.tolist():
                        if subject_tail < subject_head or not self._in_valid_range(subject_tail, prepared):
                            continue
                        subject_text = self._span_to_text(prepared, subject_head, subject_tail)
                        object_text = self._span_to_text(prepared, object_head, object_tail)
                        if not subject_text or not object_text:
                            continue
                        triple_key = (subject_text, predicate, object_text)
                        if triple_key in seen:
                            continue
                        seen.add(triple_key)
                        confidence = (
                            softmax(tag_logits[:, relation_id, subject_head, object_head])[hb_tb]
                            + softmax(tag_logits[:, relation_id, subject_head, object_tail])[hb_te]
                            + softmax(tag_logits[:, relation_id, subject_tail, object_tail])[he_te]
                        ) / 3
                        triples.append(
                            {
                                "subject": subject_text,
                                "predicate": predicate,
                                "object": object_text,
                                "confidence": round(float(confidence), 6),
                                "subject_span": [subject_head, subject_tail],
                                "object_span": [object_head, object_tail],
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
                "valid_range": [prepared["valid_start"], prepared["valid_end"]],
                "token_pieces": prepared["tokens"],
            }
        return result

    def _prepare_inputs(self, text: str, max_length: int) -> dict[str, Any]:
        words: list[tuple[str, int, int]] = []
        index = 0
        text_length = len(text)
        while index < text_length:
            if text[index].isspace():
                index += 1
                continue
            start = index
            while index < text_length and not text[index].isspace():
                index += 1
            words.append((text[start:index], start, index))

        token_ids = [self.vocab.get("[CLS]", 101)]
        offsets: list[tuple[int, int]] = [(0, 0)]
        tokens = ["[CLS]"]
        for word, start_char, end_char in words:
            pieces = self._word_piece_tokenize(word)
            for piece in pieces:
                token_ids.append(self.vocab.get(piece, self.vocab.get("[UNK]", 100)))
                tokens.append(piece)
                offsets.append((start_char, end_char))
            token_ids.append(self.vocab.get("[unused1]", self.vocab.get("[SEP]", 102)))
            tokens.append("[unused1]")
            offsets.append((end_char, end_char))

        token_ids = token_ids[: max(0, max_length - 1)]
        tokens = tokens[: max(0, max_length - 1)]
        offsets = offsets[: max(0, max_length - 1)]
        token_ids.append(self.vocab.get("[SEP]", 102))
        tokens.append("[SEP]")
        offsets.append((0, 0))

        attention_length = len(token_ids)
        if attention_length < max_length:
            pad_count = max_length - attention_length
            token_ids.extend([self.vocab.get("[PAD]", 0)] * pad_count)
            tokens.extend(["[PAD]"] * pad_count)
            offsets.extend([(0, 0)] * pad_count)

        return {
            "token_ids": token_ids,
            "mask": [1] * attention_length + [0] * max(0, max_length - attention_length),
            "tokens": tokens[:attention_length],
            "offsets": offsets,
            "valid_start": 1,
            "valid_end": max(1, attention_length - 1),
            "text": text,
        }

    def _word_piece_tokenize(self, token: str) -> list[str]:
        clean = token
        if clean in self.vocab:
            return [clean]

        chars = list(clean)
        start = 0
        pieces: list[str] = []
        while start < len(chars):
            end = len(chars)
            current_piece: str | None = None
            while start < end:
                piece = "".join(chars[start:end])
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    current_piece = piece
                    break
                end -= 1
            if current_piece is None:
                return ["[UNK]"]
            pieces.append(current_piece)
            start = end
        return pieces

    def _span_to_text(self, prepared: dict[str, Any], start_idx: int, end_idx: int) -> str:
        offsets = prepared["offsets"]
        valid_offsets = [item for item in offsets[start_idx : end_idx + 1] if item[1] > item[0]]
        if not valid_offsets:
            return ""
        start_char = valid_offsets[0][0]
        end_char = valid_offsets[-1][1]
        return prepared.get("text", "")[start_char:end_char].strip() or ""

    @staticmethod
    def _in_valid_range(index: int, prepared: dict[str, Any]) -> bool:
        return prepared["valid_start"] <= index < prepared["valid_end"]

    @staticmethod
    def _load_json(path: str | None) -> Any:
        if not path:
            return None
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
