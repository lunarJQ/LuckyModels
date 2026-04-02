from __future__ import annotations

import json
from pathlib import Path

from .schemas import GpuPoolState, utc_now


class GpuPoolStore:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> GpuPoolState:
        if not self.state_file.exists():
            return GpuPoolState()
        with self.state_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        return GpuPoolState.model_validate(payload)

    def resolve(self, visible_gpu_ids: list[int] | tuple[int, ...]) -> GpuPoolState:
        state = self.get()
        visible = sorted({int(item) for item in visible_gpu_ids if item is not None and int(item) >= 0})
        configured = sorted({int(item) for item in state.configured_gpu_ids if item is not None and int(item) >= 0})
        if state.auto_detect_all:
            effective = visible
        else:
            effective = [item for item in configured if item in visible]
        disabled = [item for item in visible if item not in effective]
        unavailable_configured = [item for item in configured if item not in visible]
        return GpuPoolState(
            auto_detect_all=bool(state.auto_detect_all),
            configured_gpu_ids=configured,
            visible_gpu_ids=visible,
            effective_gpu_ids=effective,
            disabled_gpu_ids=disabled,
            unavailable_configured_gpu_ids=unavailable_configured,
            updated_at=state.updated_at,
        )

    def save(self, allowed_gpu_ids: list[int]) -> GpuPoolState:
        normalized = sorted({int(item) for item in allowed_gpu_ids if item is not None and int(item) >= 0})
        state = GpuPoolState(
            auto_detect_all=False,
            configured_gpu_ids=normalized,
            updated_at=utc_now(),
        )
        self._write(state)
        return state

    def _write(self, state: GpuPoolState) -> None:
        with self.state_file.open("w", encoding="utf-8") as handle:
            json.dump(state.model_dump(), handle, ensure_ascii=False, indent=2)
