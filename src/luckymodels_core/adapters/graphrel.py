from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .base import BaseAdapter
from .bert_utils import create_inference_session, provider_list


class GraphRelAdapter(BaseAdapter):
    adapter_name = "graphrel_adapter"

    def __init__(self, manifest, profile) -> None:
        super().__init__(manifest, profile)
        self.session = None
        self.providers: list[str] = []
        self.session_backend = "unloaded"
        self.sessions_loaded = False
        self.id2rel: dict[int, str] = {}
        self.pos_tags: list[str] = []
        self.pos2id: dict[str, int] = {}
        self.args: dict[str, Any] = {}
        self.spacy_meta: dict[str, Any] = {}
        self.max_len = int(self.manifest.manifest.inference.max_length or 120)
        self.vector_size = 300
        self.spacy_model_name = "en_core_web_lg"
        self.spacy_source = ""
        self.nlp = None

    def load(self) -> None:
        try:
            import spacy  # type: ignore
        except ImportError as exc:
            raise RuntimeError("spacy is required for GraphRel live inference") from exc

        labels_dir = Path(self.manifest.package_root) / "labels"
        self.id2rel = {
            int(key): value
            for key, value in self._load_json(labels_dir / "id2rel.json").items()
        }
        self.pos_tags = self._load_json(labels_dir / "pos_tags.json")
        self.pos2id = {tag: idx + 1 for idx, tag in enumerate(self.pos_tags)}
        self.args = self._load_json_if_exists(labels_dir / "args.json")
        self.spacy_meta = self._load_json_if_exists(labels_dir / "spacy_meta.json")
        self.max_len = int(
            self.spacy_meta.get("max_len")
            or self.args.get("max_len")
            or self.manifest.manifest.inference.max_length
            or 120
        )
        self.vector_size = int(self.spacy_meta.get("vector_size", 300))
        self.spacy_model_name = str(self.spacy_meta.get("spacy_model", "en_core_web_lg"))

        self.nlp, self.spacy_source = self._load_spacy_pipeline(spacy)
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
            "spacy_model": self.spacy_model_name,
            "spacy_source": self.spacy_source,
            "max_len": self.max_len,
            "relation_count": len(self.id2rel),
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
            "relation_count": len(self.id2rel),
            "vector_size": self.vector_size,
            "spacy_model": self.spacy_model_name,
            "spacy_source": self.spacy_source,
            "artifact_count": len(self.manifest.resolved_artifacts),
            "profile": self.profile.model_dump(),
        }

    def predict(self, text: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.sessions_loaded or self.session is None or self.nlp is None:
            raise RuntimeError("GraphRel runtime is not loaded")

        started = perf_counter()
        options = options or {}
        tokens_override = options.get("tokens")
        token_texts, inp_sent, inp_pos, dep_fw, dep_bw = self.build_inputs(
            text=text,
            tokens=tokens_override if isinstance(tokens_override, list) else None,
        )
        seq_len = len(token_texts)
        outputs = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: inp_sent,
                self.session.get_inputs()[1].name: inp_pos,
                self.session.get_inputs()[2].name: dep_fw,
                self.session.get_inputs()[3].name: dep_bw,
            },
        )
        if len(outputs) < 4:
            raise RuntimeError("GraphRel ONNX output layout is not compatible with expected decoder")
        _, _, ne_pred, rel_pred = outputs[:4]
        entities = self._decode_entities(ne_pred[0], seq_len)
        triples = self._decode_relations(
            token_texts,
            rel_pred[0],
            entities,
            seq_len,
            evidence=text,
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
                "tokens": token_texts,
                "token_count": len(token_texts),
                "spacy_source": self.spacy_source,
            }
        return result

    def build_inputs(
        self,
        text: str | None = None,
        tokens: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if tokens:
            source_text = " ".join(tokens)
        elif text is not None:
            source_text = text
        else:
            raise ValueError("Either text or tokens must be provided")

        doc = self.nlp(source_text)
        if len(doc) == 0:
            raise ValueError("Input produced no spaCy tokens")

        seq_len = min(len(doc), self.max_len)
        inp_sent = np.zeros((1, self.max_len, self.vector_size), dtype=np.float32)
        inp_pos = np.zeros((1, self.max_len), dtype=np.int64)
        dep_fw = np.zeros((1, self.max_len, self.max_len), dtype=np.float32)
        dep_bw = np.zeros((1, self.max_len, self.max_len), dtype=np.float32)

        token_texts: list[str] = []
        for index in range(seq_len):
            token = doc[index]
            token_texts.append(token.text)
            vector = token.vector
            width = min(vector.shape[0], self.vector_size)
            inp_sent[0, index, :width] = vector[:width]
            inp_pos[0, index] = self.pos2id.get(token.tag_, 0)
            dep_fw[0, index, index] = 1.0
            dep_bw[0, index, index] = 1.0

        for index in range(seq_len):
            for child in doc[index].children:
                child_index = int(child.i)
                if child_index >= seq_len:
                    continue
                dep_fw[0, index, child_index] = 1.0
                dep_bw[0, child_index, index] = 1.0

        dep_fw[0, :seq_len, :seq_len] = self._normalize_adjacency(dep_fw[0, :seq_len, :seq_len])
        dep_bw[0, :seq_len, :seq_len] = self._normalize_adjacency(dep_bw[0, :seq_len, :seq_len])
        return token_texts, inp_sent, inp_pos, dep_fw, dep_bw

    def _load_spacy_pipeline(self, spacy_module):
        tokenizer_path = self.manifest.resolved_tokenizer.get("tokenizer_path")
        candidates: list[Path] = []
        if tokenizer_path:
            candidates.append(Path(tokenizer_path))
        candidates.extend(
            [
                Path(self.manifest.package_root) / self.spacy_model_name,
                Path(self.manifest.package_root) / "tokenizer" / self.spacy_model_name,
            ]
        )
        for candidate in candidates:
            resolved = self._resolve_model_data_dir(candidate)
            if resolved is not None:
                return spacy_module.load(str(resolved)), str(resolved)
        try:
            return spacy_module.load(self.spacy_model_name), self.spacy_model_name
        except OSError as exc:
            raise FileNotFoundError(
                "spaCy model not found for GraphRel. Set tokenizer_path in manifest or place "
                "en_core_web_lg under the model package."
            ) from exc

    def _decode_relations(
        self,
        tokens: list[str],
        rel_pred: np.ndarray,
        entities: dict[int, tuple[int, int]],
        seq_len: int,
        evidence: str,
    ) -> list[dict[str, Any]]:
        triples: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int, int, int]] = set()
        for subject_index in range(seq_len):
            for object_index in range(seq_len):
                relation_id = int(rel_pred[subject_index, object_index])
                if relation_id == 0 or subject_index not in entities or object_index not in entities:
                    continue
                subject_span = entities[subject_index]
                object_span = entities[object_index]
                triple_key = (
                    subject_span[0],
                    subject_span[1],
                    object_span[0],
                    object_span[1],
                    relation_id,
                )
                if triple_key in seen:
                    continue
                seen.add(triple_key)
                triples.append(
                    {
                        "subject": self._span_text(tokens, subject_span),
                        "predicate": self.id2rel.get(relation_id, str(relation_id)),
                        "object": self._span_text(tokens, object_span),
                        "confidence": 1.0,
                        "subject_span": [subject_span[0], subject_span[1]],
                        "object_span": [object_span[0], object_span[1]],
                        "relation_id": relation_id,
                        "evidence": evidence,
                    }
                )
        triples.sort(
            key=lambda item: (
                item["subject_span"][0],
                item["subject_span"][1],
                item["object_span"][0],
                item["object_span"][1],
                item["relation_id"],
            )
        )
        return triples

    @staticmethod
    def _normalize_adjacency(matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    @staticmethod
    def _decode_entities(ne_pred: np.ndarray, seq_len: int) -> dict[int, tuple[int, int]]:
        entities: dict[int, tuple[int, int]] = {}
        start = -1
        for index in range(seq_len):
            value = int(ne_pred[index])
            if value == 4:
                entities[index] = (index, index)
                start = -1
            elif value == 1:
                start = index
            elif value == 2:
                continue
            elif value == 3:
                if start != -1:
                    for pos in range(start, index + 1):
                        entities[pos] = (start, index)
                start = -1
            elif value == 0:
                start = -1
        return entities

    @staticmethod
    def _span_text(tokens: list[str], span: tuple[int, int]) -> str:
        return " ".join(tokens[span[0] : span[1] + 1])

    def _resolve_model_data_dir(self, path: Path) -> Path | None:
        if not path.exists():
            return None
        if (path / "config.cfg").exists() and (path / "meta.json").exists():
            return path
        nested = sorted(path.glob(f"{self.spacy_model_name}-*"))
        for candidate in nested:
            if (candidate / "config.cfg").exists() and (candidate / "meta.json").exists():
                return candidate
        return None

    @staticmethod
    def _load_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_json_if_exists(path: Path) -> Any:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
