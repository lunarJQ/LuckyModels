from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TokenizedText:
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    offsets: list[tuple[int, int]]
    tokens: list[str]
    valid_start: int
    valid_end: int


def load_json(path: str | None) -> Any:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_relation_maps(path: str | None) -> tuple[dict[int, str], dict[str, int]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        relation2id = {str(key): int(value) for key, value in payload.items()}
        id2relation = {value: key for key, value in relation2id.items()}
        return id2relation, relation2id
    if (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], dict)
    ):
        relation2id = {str(key): int(value) for key, value in payload[0].items()}
        id2relation = {value: key for key, value in relation2id.items()}
        return id2relation, relation2id
    if (
        isinstance(payload, list)
        and len(payload) == 2
        and isinstance(payload[0], dict)
        and isinstance(payload[1], dict)
    ):
        id2relation = {int(key): str(value) for key, value in payload[0].items()}
        relation2id = {str(key): int(value) for key, value in payload[1].items()}
        return id2relation, relation2id
    return {}, {}


def infer_do_lower_case(backbone: str) -> bool:
    lowered = backbone.lower()
    if "uncased" in lowered:
        return True
    if "cased" in lowered:
        return False
    return False


def load_bert_tokenizer(vocab_path: str, tokenizer_path: str | None, backbone: str):
    try:
        from transformers import BertTokenizerFast
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "transformers is required for BERT-based LuckyModels adapters"
        ) from exc

    kwargs: dict[str, Any] = {
        "vocab_file": vocab_path,
        "do_lower_case": infer_do_lower_case(backbone),
    }
    if tokenizer_path and tokenizer_path.endswith(".json") and Path(tokenizer_path).exists():
        kwargs["tokenizer_file"] = tokenizer_path
    return BertTokenizerFast(**kwargs)


def tokenize_with_offsets(tokenizer, text: str, max_length: int) -> TokenizedText:
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    token_type_ids = list(encoded.get("token_type_ids", [0] * len(input_ids)))
    attention_length = int(sum(attention_mask))
    offsets = [tuple(item) for item in encoded["offset_mapping"]]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[:attention_length])
    valid_start = 1 if attention_length >= 2 else 0
    valid_end = max(valid_start, attention_length - 1)
    return TokenizedText(
        text=text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        offsets=offsets,
        tokens=tokens,
        valid_start=valid_start,
        valid_end=valid_end,
    )


def span_text_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    start_idx: int,
    end_idx: int,
) -> str:
    if start_idx < 0 or end_idx >= len(offsets) or start_idx > end_idx:
        return ""

    span_offsets = [
        (start, end)
        for start, end in offsets[start_idx : end_idx + 1]
        if end > start
    ]
    if not span_offsets:
        return ""

    start_char = span_offsets[0][0]
    end_char = span_offsets[-1][1]
    if end_char <= start_char:
        return ""
    return text[start_char:end_char].strip()


def build_handshaking_index(seq_len: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    spots: list[tuple[int, int]] = []
    spot_to_index: dict[tuple[int, int], int] = {}
    index = 0
    for start in range(seq_len):
        for end in range(start, seq_len):
            spots.append((start, end))
            spot_to_index[(start, end)] = index
            index += 1
    return spots, spot_to_index


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(), 1e-12)


def decode_bio_spans(
    tags: np.ndarray,
    valid_start: int,
    valid_end: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None

    for index in range(valid_start, valid_end):
        tag = int(tags[index])
        if tag == 1:
            if current_start is not None:
                spans.append((current_start, current_end if current_end is not None else current_start))
            current_start = index
            current_end = index
        elif tag == 2 and current_start is not None:
            current_end = index
        else:
            if current_start is not None:
                spans.append((current_start, current_end if current_end is not None else current_start))
                current_start = None
                current_end = None

    if current_start is not None:
        spans.append((current_start, current_end if current_end is not None else current_start))

    return spans


def provider_list(profile_device: str) -> list[str]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency path
        raise RuntimeError(
            "onnxruntime is required for LuckyModels live inference adapters"
        ) from exc

    if profile_device.lower().startswith("cuda"):
        # When the GPU wheel is installed via NVIDIA site packages, preload the
        # CUDA/cuDNN shared objects before checking available providers.
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload(directory="")
            except Exception:
                pass

    available = ort.get_available_providers()
    if profile_device.lower().startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def build_session_options(profile: Any):
    import onnxruntime as ort  # type: ignore

    session_options = ort.SessionOptions()
    intra_threads = int(profile.execution.get("intra_op_threads", 0) or 0)
    inter_threads = int(profile.execution.get("inter_op_threads", 0) or 0)
    if intra_threads > 0:
        session_options.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        session_options.inter_op_num_threads = inter_threads
    return session_options


def create_inference_session(model_path: str, providers: list[str], profile: Any):
    import onnxruntime as ort  # type: ignore

    return ort.InferenceSession(
        model_path,
        sess_options=build_session_options(profile),
        providers=providers,
    )


def load_vocab(vocab_path: str) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with Path(vocab_path).open("r", encoding="utf-8") as handle:
        for index, token in enumerate(handle):
            vocab[token.rstrip("\n")] = index
    return vocab


def tokenize_with_wordpiece_vocab(
    text: str,
    vocab: dict[str, int],
    max_length: int,
    do_lower_case: bool = False,
) -> TokenizedText:
    cls_id = vocab.get("[CLS]", 101)
    sep_id = vocab.get("[SEP]", 102)
    pad_id = vocab.get("[PAD]", 0)
    unk_id = vocab.get("[UNK]", 100)

    tokens: list[str] = ["[CLS]"]
    offsets: list[tuple[int, int]] = [(0, 0)]
    input_ids: list[int] = [cls_id]

    for piece, start_char, end_char in iter_wordpiece_tokens(
        text,
        vocab,
        do_lower_case=do_lower_case,
    ):
        if len(input_ids) >= max_length - 1:
            break
        tokens.append(piece)
        offsets.append((start_char, end_char))
        input_ids.append(vocab.get(piece, unk_id))

    tokens.append("[SEP]")
    offsets.append((0, 0))
    input_ids.append(sep_id)

    attention_length = len(input_ids)
    if attention_length < max_length:
        pad_count = max_length - attention_length
        input_ids.extend([pad_id] * pad_count)
        offsets.extend([(0, 0)] * pad_count)

    attention_mask = [1] * attention_length + [0] * max(0, max_length - attention_length)
    token_type_ids = [0] * max_length
    valid_start = 1 if attention_length >= 2 else 0
    valid_end = max(valid_start, attention_length - 1)
    return TokenizedText(
        text=text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        offsets=offsets,
        tokens=tokens,
        valid_start=valid_start,
        valid_end=valid_end,
    )


def iter_wordpiece_tokens(
    text: str,
    vocab: dict[str, int],
    do_lower_case: bool = False,
):
    for token_text, start_char, end_char in basic_tokenize_with_offsets(
        text,
        do_lower_case=do_lower_case,
    ):
        for piece, piece_start, piece_end in wordpiece_tokenize_with_offsets(
            token_text,
            start_char,
            vocab,
        ):
            yield piece, piece_start, piece_end


def basic_tokenize_with_offsets(text: str, do_lower_case: bool = False):
    current: list[str] = []
    current_start: int | None = None

    def flush() -> tuple[str, int, int] | None:
        nonlocal current, current_start
        if current_start is None or not current:
            current = []
            current_start = None
            return None
        token = "".join(current)
        start = current_start
        end = current_start + len(token)
        current = []
        current_start = None
        if do_lower_case:
            token = unicodedata.normalize("NFD", token)
            token = "".join(ch for ch in token if unicodedata.category(ch) != "Mn")
            token = token.lower()
        return token, start, end

    for index, char in enumerate(text):
        if char.isspace():
            item = flush()
            if item is not None:
                yield item
            continue
        if _is_chinese_char(ord(char)):
            item = flush()
            if item is not None:
                yield item
            yield char, index, index + 1
            continue
        if current_start is None:
            current_start = index
        current.append(char)

    item = flush()
    if item is not None:
        yield item


def _is_chinese_char(codepoint: int) -> bool:
    return any(
        [
            0x4E00 <= codepoint <= 0x9FFF,
            0x3400 <= codepoint <= 0x4DBF,
            0x20000 <= codepoint <= 0x2A6DF,
            0x2A700 <= codepoint <= 0x2B73F,
            0x2B740 <= codepoint <= 0x2B81F,
            0x2B820 <= codepoint <= 0x2CEAF,
            0xF900 <= codepoint <= 0xFAFF,
            0x2F800 <= codepoint <= 0x2FA1F,
        ]
    )


def wordpiece_tokenize_with_offsets(token: str, base_start: int, vocab: dict[str, int]):
    if token in vocab:
        yield token, base_start, base_start + len(token)
        return

    chars = list(token)
    start = 0
    while start < len(chars):
        end = len(chars)
        current_piece: str | None = None
        current_end = end
        while start < end:
            piece = "".join(chars[start:end])
            candidate = piece if start == 0 else "##" + piece
            if candidate in vocab:
                current_piece = candidate
                current_end = end
                break
            end -= 1
        if current_piece is None:
            yield "[UNK]", base_start + start, base_start + len(chars)
            return
        yield current_piece, base_start + start, base_start + current_end
        start = current_end
