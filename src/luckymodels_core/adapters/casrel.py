from __future__ import annotations

import json
import re
import unicodedata
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
        self.tokenizer_backend = "wordpiece-vocab"
        self.tokenizer = None
        self.pad_id = 0
        self.unk_id = 100
        self.cls_id = 101
        self.sep_id = 102
        self.mask_id = 103
        self.package_warnings: list[str] = []
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
            self.pad_id = self.vocab.get("[PAD]", 0)
            self.unk_id = self.vocab.get("[UNK]", 100)
            self.cls_id = self.vocab.get("[CLS]", 101)
            self.sep_id = self.vocab.get("[SEP]", 102)
            self.mask_id = self.vocab.get("[MASK]", 103)
            self._validate_vocab_layout()

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
            "package_warnings": self.package_warnings,
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
            "package_warnings": self.package_warnings,
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
        valid_start = int(prepared.get("valid_start_index", 0))
        valid_end = int(prepared.get("valid_end_index", prepared["text_length"]))

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
            valid_start,
            valid_end,
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
                    valid_start=valid_start,
                    valid_end=valid_end,
                    threshold=object_threshold,
                    seen=seen,
                )
            )

        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        result = {
            "model_id": self.manifest.manifest.model_id,
            "model_name": self.manifest.manifest.model_name,
            "version": self.manifest.manifest.version,
            "language": self.manifest.manifest.language,
            "latency_ms": elapsed_ms,
            "text": prepared["normalized_text"],
            "triple_count": len(triples),
            "triples": triples,
            "warnings": self.package_warnings,
        }
        if options.get("debug") or options.get("return_debug"):
            result["debug"] = {
                "token_pieces": ["[CLS]", *prepared["token_pieces"], "[SEP]"],
                "token_ids": prepared["token_ids"][: prepared["text_length"]],
                "text_length": prepared["text_length"],
                "valid_range": [valid_start, valid_end],
                "subject_threshold": subject_threshold,
                "object_threshold": object_threshold,
                "subject_score_summary": self._score_summary(
                    pred_sub_heads[0], pred_sub_tails[0], valid_start, valid_end
                ),
                "subject_candidates": [
                    {
                        "span": [start, end],
                        "text": self._span_to_text(text, prepared["token_ids"], start, end),
                    }
                    for start, end in subject_spans[:20]
                ],
            }
        return result

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
            token_pieces = self.tokenizer.convert_ids_to_tokens(token_ids[:attention_length])
        else:
            token_pieces = self._tokenize_to_wordpieces(normalized_text)
            token_pieces = token_pieces[: max(0, max_length - 2)]
            token_ids = [self.cls_id]
            token_ids.extend(self.vocab.get(piece, self.unk_id) for piece in token_pieces)
            token_ids.append(self.sep_id)
            attention_length = len(token_ids)
            if attention_length < max_length:
                token_ids.extend([self.pad_id] * (max_length - attention_length))
            segments = [0] * max_length

        return {
            "normalized_text": normalized_text,
            "tokens": np.array([token_ids], dtype=np.int32),
            "segments": np.array([segments], dtype=np.int32),
            "text_length": attention_length,
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "valid_start_index": 1 if attention_length >= 2 else 0,
            "valid_end_index": max(1, attention_length - 1) if attention_length >= 2 else attention_length,
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
        valid_start: int,
        valid_end: int,
        threshold: float,
    ) -> list[tuple[int, int]]:
        subjects: list[tuple[int, int]] = []
        for start_idx in range(valid_start, valid_end):
            if float(pred_sub_heads[start_idx][0]) > threshold:
                for end_idx in range(start_idx, valid_end):
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
        valid_start: int,
        valid_end: int,
        threshold: float,
        seen: set[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        subject_start, subject_end = subject_span
        subject_text = self._span_to_text(text, token_ids, subject_start, subject_end)
        triples: list[dict[str, Any]] = []

        relation_count = pred_obj_heads.shape[1]
        for relation_id in range(relation_count):
            for object_start in range(valid_start, valid_end):
                start_score = float(pred_obj_heads[object_start][relation_id])
                if start_score <= threshold:
                    continue
                for object_end in range(object_start, valid_end):
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
        pieces = [self._id_to_token(token_id) for token_id in token_ids[start_idx : end_idx + 1]]
        return self._merge_wordpieces(pieces)

    def _tokenize_to_wordpieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for token in self._basic_tokenize(text):
            pieces.extend(self._wordpiece_tokenize(token))
        return pieces

    def _basic_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        buffer: list[str] = []
        for raw_char in text:
            char = raw_char.lower()
            if self._is_whitespace(char):
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                continue
            if self._is_control(char):
                continue
            if self._is_cjk(char) or self._is_punctuation(char):
                if buffer:
                    tokens.append("".join(buffer))
                    buffer = []
                tokens.append(char)
                continue
            buffer.append(char)

        if buffer:
            tokens.append("".join(buffer))
        return tokens

    def _wordpiece_tokenize(self, token: str) -> list[str]:
        if token in self.vocab:
            return [token]

        chars = list(token)
        start = 0
        sub_tokens: list[str] = []
        while start < len(chars):
            end = len(chars)
            current_substr: str | None = None
            while start < end:
                piece = "".join(chars[start:end])
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    current_substr = piece
                    break
                end -= 1
            if current_substr is None:
                return ["[UNK]"]
            sub_tokens.append(current_substr)
            start = end
        return sub_tokens

    def _validate_vocab_layout(self) -> None:
        if self.cls_id > 10 and self.tokenizer_backend == "wordpiece-vocab":
            self.package_warnings.append(
                "Detected legacy BERT vocab layout with special tokens stored after [unused] blocks; "
                "using custom WordPiece tokenization with raw vocab ids."
            )

        probe_tokens = self._tokenize_to_wordpieces("The 1st runway length in feet is 14000.")
        if probe_tokens and all(token == "[UNK]" for token in probe_tokens):
            self.package_warnings.append(
                "Tokenizer probe collapsed to all [UNK]; uploaded vocab may not match the model weights."
            )

        if self.id2relation:
            sample_relations = {value for _, value in list(self.id2relation.items())[:10]}
            if any("runway" in relation.lower() or "oclc" in relation.lower() for relation in sample_relations):
                self.package_warnings.append(
                    "Relation labels look like a WebNLG-style English schema, not an equipment-domain Chinese schema."
                )

    @staticmethod
    def _score_summary(
        pred_sub_heads: np.ndarray,
        pred_sub_tails: np.ndarray,
        valid_start: int,
        valid_end: int,
    ) -> dict[str, Any]:
        head_scores = pred_sub_heads[valid_start:valid_end, 0]
        tail_scores = pred_sub_tails[valid_start:valid_end, 0]
        head_rank = np.argsort(head_scores)[-10:][::-1].tolist()
        tail_rank = np.argsort(tail_scores)[-10:][::-1].tolist()
        return {
            "head_max": float(head_scores.max()) if len(head_scores) else 0.0,
            "tail_max": float(tail_scores.max()) if len(tail_scores) else 0.0,
            "top_head_positions": [
                {"position": valid_start + idx, "score": float(head_scores[idx])}
                for idx in head_rank
            ],
            "top_tail_positions": [
                {"position": valid_start + idx, "score": float(tail_scores[idx])}
                for idx in tail_rank
            ],
        }

    def _id_to_token(self, token_id: int) -> str:
        for token, idx in self.vocab.items():
            if idx == token_id:
                return token
        return "[UNK]"

    @staticmethod
    def _merge_wordpieces(tokens: list[str]) -> str:
        merged: list[str] = []
        for token in tokens:
            if token in {"[CLS]", "[SEP]", "[PAD]"}:
                continue
            if token.startswith("##") and merged:
                merged[-1] = merged[-1] + token[2:]
            else:
                merged.append(token)
        text = " ".join(merged)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text.strip()

    @staticmethod
    def _is_whitespace(char: str) -> bool:
        return char.isspace()

    @staticmethod
    def _is_control(char: str) -> bool:
        return unicodedata.category(char) in {"Cc", "Cf"}

    @staticmethod
    def _is_punctuation(char: str) -> bool:
        return unicodedata.category(char).startswith("P")

    @staticmethod
    def _is_cjk(char: str) -> bool:
        code = ord(char)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
            or 0x2F800 <= code <= 0x2FA1F
        )
