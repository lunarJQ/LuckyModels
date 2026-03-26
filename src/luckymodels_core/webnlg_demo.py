from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings import get_settings


WEBNLG_DEMO_CASES = [
    {
        "name": "language_case",
        "index": 78,
        "threshold": 0.0001,
        "description": "Clean WebNLG sentence with language-related gold triples.",
    },
    {
        "name": "location_case",
        "index": 96,
        "threshold": 0.0001,
        "description": "Short WebNLG sentence that yields non-empty location-style predictions.",
    },
    {
        "name": "ethnic_group_case",
        "index": 119,
        "threshold": 0.0001,
        "description": "WebNLG sample with non-empty predictions and multiple gold relations.",
    },
]


DEFAULT_WEBNLG_EVAL_LIMIT = 128


def load_webnlg_dev_records() -> list[dict[str, Any]]:
    settings = get_settings()
    dataset_root = _get_dataset_root()
    generated_path = dataset_root / "new_dev.json"
    if generated_path.exists():
        return _load_jsonl_records(generated_path)

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


def get_webnlg_demo_cases() -> dict[str, dict[str, Any]]:
    records = load_webnlg_dev_records()
    cases: dict[str, dict[str, Any]] = {}
    for spec in WEBNLG_DEMO_CASES:
        record = records[spec["index"]]
        cases[spec["name"]] = {
            "name": spec["name"],
            "description": spec["description"],
            "index": spec["index"],
            "threshold": spec["threshold"],
            "sentText": record["sentText"],
            "relationMentions": record["relationMentions"],
        }
    return cases


def export_webnlg_eval_cases(
    output_path: Path | None = None,
    limit: int = DEFAULT_WEBNLG_EVAL_LIMIT,
) -> dict[str, Any]:
    settings = get_settings()
    records = load_webnlg_dev_records()
    prepared_records = []
    for index, record in enumerate(records):
        relations = record.get("relationMentions") or []
        if not relations:
            continue
        prepared_records.append(
            {
                "case_id": f"webnlg-dev-{index:05d}",
                "dataset_index": index,
                "sentText": record["sentText"],
                "relation_count": len(relations),
                "relationMentions": relations,
            }
        )
        if len(prepared_records) >= limit:
            break

    destination = output_path or settings.webnlg_eval_cases_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_name": "WebNLG dev",
        "dataset_source": _display_path(get_webnlg_dataset_source()),
        "record_count": len(prepared_records),
        "records": prepared_records,
    }
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {
        "dataset_name": payload["dataset_name"],
        "dataset_source": payload["dataset_source"],
        "record_count": payload["record_count"],
        "output_path": _display_path(destination),
    }


def get_webnlg_dataset_source() -> Path:
    dataset_root = _get_dataset_root()
    generated_path = dataset_root / "new_dev.json"
    if generated_path.exists():
        return generated_path
    return dataset_root / "dev.json"


def _get_dataset_root() -> Path:
    settings = get_settings()
    return settings.root_dir / "datasets" / "raw_WebNLG" / "pre_processed_data"


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
