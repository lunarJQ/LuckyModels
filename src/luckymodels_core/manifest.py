from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import LoadedManifest, ManifestModel, ProfileConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return data


def _resolve_path(package_root: Path, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    return str((package_root / raw_path).resolve())


def load_manifest(manifest_path: str | Path) -> LoadedManifest:
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    package_root = manifest_file.parent
    manifest = ManifestModel.model_validate(_load_yaml(manifest_file))

    resolved_artifacts = {
        name: _resolve_path(package_root, path) for name, path in manifest.artifacts.items()
    }
    resolved_profiles = {
        name: _resolve_path(package_root, path) for name, path in manifest.profiles.items()
    }
    resolved_tokenizer = {
        "vocab_path": _resolve_path(package_root, manifest.tokenizer.vocab_path),
        "tokenizer_path": _resolve_path(package_root, manifest.tokenizer.tokenizer_path),
    }
    resolved_labels = {
        "relation_map": _resolve_path(package_root, manifest.labels.relation_map),
        "entity_map": _resolve_path(package_root, manifest.labels.entity_map),
        "schema_path": _resolve_path(package_root, manifest.labels.schema_path),
    }

    return LoadedManifest(
        manifest_path=str(manifest_file),
        package_root=str(package_root),
        manifest=manifest,
        resolved_artifacts=resolved_artifacts,
        resolved_profiles=resolved_profiles,
        resolved_tokenizer={k: v for k, v in resolved_tokenizer.items() if v},
        resolved_labels={k: v for k, v in resolved_labels.items() if v},
    )


def validate_loaded_manifest(loaded: LoadedManifest) -> None:
    for _, path in loaded.resolved_artifacts.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")

    for _, path in loaded.resolved_profiles.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Profile path does not exist: {path}")

    for _, path in loaded.resolved_tokenizer.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {path}")

    for _, path in loaded.resolved_labels.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Label path does not exist: {path}")


def load_profile(profile_path: str | Path) -> ProfileConfig:
    profile_file = Path(profile_path).resolve()
    if not profile_file.exists():
        raise FileNotFoundError(f"Profile not found: {profile_file}")
    return ProfileConfig.model_validate(_load_yaml(profile_file))
