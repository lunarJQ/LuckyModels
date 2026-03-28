from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter
from .bert_utils import create_inference_session, provider_list


class CopyMTLAdapter(BaseAdapter):
    adapter_name = "copymtl_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.words2id: dict[str, int] = {}
        self.relations2id: dict[str, int] = {}
        self.id2relations: dict[int, str] = {}
        self.max_sentence_length = int(self.manifest.manifest.inference.max_length or 80)
        self.relation_number = 0
        self.eos_word_id = 0
        self.relation_eos_id = 0
        self.treebank = None
        self.nltk_available = False
        self.package_warnings: list[str] = []

    def load(self) -> None:
        words_path = self.manifest.resolved_tokenizer.get("vocab_path")
        relation_path = self.manifest.resolved_labels.get("relation_map")
        if not words_path or not relation_path:
            raise RuntimeError("CopyMTL package is missing words2id or relations2id assets")

        self.words2id = self._load_json(words_path)
        self.relations2id = self._load_json(relation_path)
        self.id2relations = {int(value): str(key) for key, value in self.relations2id.items()}
        self.relation_number = len(self.relations2id)
        self.eos_word_id = len(self.words2id)
        self.relation_eos_id = self.relation_number

        try:
            from nltk.tokenize import TreebankWordTokenizer, word_tokenize  # type: ignore
        except ImportError as exc:
            raise RuntimeError("nltk is required for CopyMTL live inference") from exc

        self._word_tokenize = word_tokenize
        self.treebank = TreebankWordTokenizer()
        self.nltk_available = True

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
            "max_sentence_length": self.max_sentence_length,
            "relation_count": self.relation_number,
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
            "dataset_name": self.manifest.manifest.dataset_name,
            "relation_count": self.relation_number,
            "vocab_size": len(self.words2id),
            "artifact_count": len(self.manifest.resolved_artifacts),
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.sessions_loaded or self.session is None:
            raise RuntimeError("CopyMTL runtime is not loaded")

        started = perf_counter()
        options = options or {}
        strict_vocab = bool(options.get("strict_vocab", False))
        encoded = self._encode(text, strict_vocab=strict_vocab)
        outputs = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: encoded["sentence"],
                self.session.get_inputs()[1].name: encoded["lengths"],
            },
        )
        pred_actions, relation_actions, entity1_actions, entity2_actions = outputs[:4]
        triples = self._decode_actions(
            relation_actions=relation_actions[0].tolist(),
            entity1_actions=entity1_actions[0].tolist(),
            entity2_actions=entity2_actions[0].tolist(),
            tokens=encoded["tokens"],
            evidence=text,
        )
        warnings: list[str] = []
        if encoded["oov_tokens"]:
            warnings.append(
                "Some tokens were missing from CopyMTL vocabulary and were mapped to padding ids: "
                + ", ".join(encoded["oov_tokens"][:12])
                + (" ..." if len(encoded["oov_tokens"]) > 12 else "")
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
            "warnings": warnings,
        }
        if options.get("debug") or options.get("return_debug"):
            result["debug"] = {
                "tokens": encoded["tokens"],
                "token_count": len(encoded["tokens"]),
                "oov_tokens": encoded["oov_tokens"],
                "raw_actions": [int(value) for value in pred_actions[0].tolist()],
            }
        return result

    def _encode(self, text: str, strict_vocab: bool) -> dict[str, Any]:
        tokens = self._tokenize(text)
        if not tokens:
            raise ValueError("Input text produced no tokens after tokenization")

        tokens = tokens[: self.max_sentence_length]
        token_ids: list[int] = []
        oov_tokens: list[str] = []
        kept_tokens: list[str] = []
        for token in tokens:
            token_id = self._lookup_word_id(token)
            if token_id is None:
                oov_tokens.append(token)
                if strict_vocab:
                    continue
                token_id = 0
            token_ids.append(token_id)
            kept_tokens.append(token)

        if strict_vocab and oov_tokens:
            unique_oov = sorted(set(oov_tokens))
            raise ValueError(
                "CopyMTL vocabulary miss: "
                + ", ".join(unique_oov[:20])
                + (" ..." if len(unique_oov) > 20 else "")
            )
        if not token_ids:
            raise ValueError("No usable tokens remain after vocabulary lookup")

        length = len(token_ids)
        padded = token_ids + [0] * (self.max_sentence_length - length)
        return {
            "tokens": kept_tokens,
            "sentence": np.asarray([padded], dtype=np.int64),
            "lengths": np.asarray([length], dtype=np.int64),
            "oov_tokens": sorted(set(oov_tokens)),
        }

    def _decode_actions(
        self,
        relation_actions: list[int],
        entity1_actions: list[int],
        entity2_actions: list[int],
        tokens: list[str],
        evidence: str,
    ) -> list[dict[str, Any]]:
        triples: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int]] = set()

        for relation_id, e1_end, e2_end in zip(relation_actions, entity1_actions, entity2_actions):
            relation_id = int(relation_id)
            e1_end = int(e1_end)
            e2_end = int(e2_end)
            if (
                relation_id == self.relation_eos_id
                and e1_end == self.max_sentence_length
                and e2_end == self.max_sentence_length
            ):
                continue
            if relation_id >= self.relation_number:
                continue
            if e1_end >= len(tokens) or e2_end >= len(tokens):
                continue

            triple_key = (e1_end, e2_end, relation_id)
            if triple_key in seen:
                continue
            seen.add(triple_key)
            triples.append(
                {
                    "subject": tokens[e1_end],
                    "predicate": self.id2relations.get(relation_id, str(relation_id)),
                    "object": tokens[e2_end],
                    "confidence": 1.0,
                    "subject_span": [e1_end, e1_end],
                    "object_span": [e2_end, e2_end],
                    "relation_id": relation_id,
                    "evidence": evidence,
                }
            )
        return triples

    def _tokenize(self, text: str) -> list[str]:
        try:
            return self._word_tokenize(text, preserve_line=True)
        except LookupError:
            return self.treebank.tokenize(text)

    def _lookup_word_id(self, token: str) -> int | None:
        candidates = [
            token,
            self._remove_tone(token),
            token.lower(),
            self._remove_tone(token.lower()),
        ]
        for candidate in candidates:
            if candidate in self.words2id:
                return int(self.words2id[candidate])
        return None

    @staticmethod
    def _remove_tone(text: str) -> str:
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    @staticmethod
    def _load_json(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
