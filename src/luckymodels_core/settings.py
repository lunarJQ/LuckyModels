from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    artifacts_root: Path
    state_root: Path
    deployments_file: Path


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    state_root = root_dir / "data" / "state"
    return Settings(
        root_dir=root_dir,
        artifacts_root=root_dir / "artifacts" / "models",
        state_root=state_root,
        deployments_file=state_root / "deployments.json",
    )
