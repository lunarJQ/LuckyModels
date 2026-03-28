from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings import get_settings

DATASET_ALIASES = {
    "duie": "duie",
    "duie2.0": "duie",
    "duie2.0-main": "duie",
    "nyt": "nyt",
    "raw_webnlg": "webnlg",
    "scierc": "scierc",
    "webnlg": "webnlg",
}


def resolve_dataset_key(dataset_name: str | None) -> str:
    raw_name = (dataset_name or "").strip().lower()
    if not raw_name:
        raise ValueError("dataset_name is required")
    return DATASET_ALIASES.get(raw_name, raw_name)


def get_dataset_eval_cases(
    dataset_name: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    dataset_key = resolve_dataset_key(dataset_name)
    if dataset_key == "webnlg":
        payload = _load_webnlg_cases()
    elif dataset_key == "nyt":
        payload = _load_nyt_cases()
    elif dataset_key == "scierc":
        payload = _load_scierc_cases()
    elif dataset_key == "duie":
        payload = _load_duie_cases()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    records = payload["records"]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return {
        "dataset_key": dataset_key,
        "dataset_name": payload["dataset_name"],
        "dataset_source": payload["dataset_source"],
        "record_count": len(records),
        "records": records,
    }


def export_dataset_eval_cases(
    dataset_name: str,
    *,
    limit: int = 128,
    output_path: Path | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    payload = get_dataset_eval_cases(dataset_name, limit=limit)
    destination = output_path or (
        settings.dataset_eval_cases_dir / f"{payload['dataset_key']}_eval_cases.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {
        "dataset_key": payload["dataset_key"],
        "dataset_name": payload["dataset_name"],
        "dataset_source": payload["dataset_source"],
        "record_count": payload["record_count"],
        "output_path": _display_path(destination),
    }


def _load_webnlg_cases() -> dict[str, Any]:
    dataset_root = _get_webnlg_root()
    generated_path = dataset_root / "new_dev.json"
    if generated_path.exists():
        source_path = generated_path
        records = _load_jsonl_records(generated_path)
    else:
        source_path = dataset_root / "dev.json"
        records = _rebuild_webnlg_records(dataset_root)

    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        relations = record.get("relationMentions") or []
        if not relations:
            continue
        triples = [
            {
                "subject": item.get("em1Text", ""),
                "predicate": item.get("label", ""),
                "object": item.get("em2Text", ""),
                "evidence": record.get("sentText", ""),
            }
            for item in relations
            if item.get("em1Text") and item.get("em2Text") and item.get("label")
        ]
        if not triples:
            continue
        prepared.append(
            {
                "case_id": f"webnlg-dev-{index:05d}",
                "dataset_index": index,
                "dataset_split": "dev",
                "text": record.get("sentText", ""),
                "triple_count": len(triples),
                "triples": triples,
            }
        )

    return {
        "dataset_name": "WebNLG dev",
        "dataset_source": _display_path(source_path),
        "records": prepared,
    }


def _load_nyt_cases() -> dict[str, Any]:
    source_path = get_settings().root_dir / "datasets" / "NYT" / "test_triples.json"
    with source_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        triple_list = record.get("triple_list") or []
        triples = [
            {
                "subject": item[0],
                "predicate": item[1],
                "object": item[2],
                "evidence": record.get("text", ""),
            }
            for item in triple_list
            if isinstance(item, list) and len(item) == 3
        ]
        if not triples:
            continue
        prepared.append(
            {
                "case_id": f"nyt-test-{index:05d}",
                "dataset_index": index,
                "dataset_split": "test",
                "text": record.get("text", ""),
                "triple_count": len(triples),
                "triples": triples,
            }
        )
    return {
        "dataset_name": "NYT test",
        "dataset_source": _display_path(source_path),
        "records": prepared,
    }


def _load_scierc_cases() -> dict[str, Any]:
    raw_root = get_settings().root_dir / "datasets" / "sciERC_raw" / "raw_data"
    prepared: list[dict[str, Any]] = []
    for ann_path in sorted(raw_root.glob("*.ann")):
        text_path = ann_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8")
        entities: dict[str, dict[str, str]] = {}
        triples: list[dict[str, Any]] = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            if line.startswith("T"):
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                entity_id = parts[0]
                label = parts[1].split(" ", maxsplit=1)[0]
                entities[entity_id] = {
                    "label": label,
                    "text": parts[2],
                }
            elif line.startswith("R"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                fields = parts[1].split()
                if len(fields) < 3:
                    continue
                subject_id = fields[1].split(":", maxsplit=1)[-1]
                object_id = fields[2].split(":", maxsplit=1)[-1]
                subject = entities.get(subject_id)
                obj = entities.get(object_id)
                if not subject or not obj:
                    continue
                triples.append(
                    {
                        "subject": subject["text"],
                        "predicate": _normalize_scierc_relation(fields[0]),
                        "object": obj["text"],
                        "subject_type": subject["label"],
                        "object_type": obj["label"],
                        "evidence": text,
                    }
                )
        if not triples:
            continue
        prepared.append(
            {
                "case_id": f"scierc-{ann_path.stem}",
                "dataset_index": ann_path.stem,
                "dataset_split": "raw",
                "text": text,
                "triple_count": len(triples),
                "triples": triples,
            }
        )
    return {
        "dataset_name": "SciERC raw",
        "dataset_source": _display_path(raw_root),
        "records": prepared,
    }


def _load_duie_cases() -> dict[str, Any]:
    source_path = (
        get_settings().root_dir
        / "datasets"
        / "DuIE2.0-main"
        / "duie_dev.json"
        / "duie_dev.json"
        / "duie_dev.json"
    )
    records = _load_jsonl_records(source_path)
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        spo_list = record.get("spo_list") or []
        triples = []
        for item in spo_list:
            subject = item.get("subject", "")
            predicate = item.get("predicate", "")
            object_text = _extract_duie_object_text(item.get("object"))
            if not subject or not predicate or not object_text:
                continue
            triples.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_text,
                    "subject_type": item.get("subject_type"),
                    "object_type": item.get("object_type"),
                    "evidence": record.get("text", ""),
                }
            )
        if not triples:
            continue
        prepared.append(
            {
                "case_id": f"duie-dev-{index:05d}",
                "dataset_index": index,
                "dataset_split": "dev",
                "text": record.get("text", ""),
                "triple_count": len(triples),
                "triples": triples,
            }
        )
    return {
        "dataset_name": "DuIE2.0 dev",
        "dataset_source": _display_path(source_path),
        "records": prepared,
    }


def _extract_duie_object_text(value: Any) -> str:
    if isinstance(value, dict):
        primary = value.get("@value")
        if isinstance(primary, str) and primary.strip():
            return primary.strip()
        for key in sorted(value.keys()):
            item = value[key]
            if isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _rebuild_webnlg_records(dataset_root: Path) -> list[dict[str, Any]]:
    dev_path = dataset_root / "dev.json"
    rel_path = dataset_root / "relations2id.json"
    words_path = dataset_root / "words2id.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"WebNLG dev split not found: {dev_path}")

    with dev_path.open("r", encoding="utf-8") as handle:
        token_ids, triple_specs = json.load(handle)
    with words_path.open("r", encoding="utf-8") as handle:
        words2id = json.load(handle)
    with rel_path.open("r", encoding="utf-8") as handle:
        rel2id = json.load(handle)

    word_dict = {int(value): key for key, value in words2id.items()}
    rel_dict = {int(value): key for key, value in rel2id.items()}
    records: list[dict[str, Any]] = []
    for tokens, triples in zip(token_ids, triple_specs, strict=False):
        token_text = [word_dict[int(token)] for token in tokens]
        sent_text = " ".join(token_text)
        relation_mentions = []
        for offset in range(0, len(triples), 3):
            em1 = token_text[triples[offset]]
            em2 = token_text[triples[offset + 1]]
            label = rel_dict[int(triples[offset + 2])]
            relation_mentions.append(
                {
                    "em1Text": em1,
                    "em2Text": em2,
                    "label": label,
                }
            )
        records.append(
            {
                "sentText": sent_text,
                "relationMentions": relation_mentions,
            }
        )
    return records


def _normalize_scierc_relation(label: str) -> str:
    normalized = label.replace("_", "-").lower()
    if not normalized:
        return label
    return normalized[0].upper() + normalized[1:]


def _get_webnlg_root() -> Path:
    return get_settings().root_dir / "datasets" / "raw_WebNLG" / "pre_processed_data"


def _display_path(path: Path) -> str:
    settings = get_settings()
    try:
        return str(path.relative_to(settings.root_dir))
    except ValueError:
        return str(path)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
