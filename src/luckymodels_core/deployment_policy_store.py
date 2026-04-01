from __future__ import annotations

import json
from pathlib import Path

from .schemas import (
    DeploymentPolicyAutomationAction,
    DeploymentPolicyAutomationState,
    DeploymentPolicyAutomationUpdateRequest,
)


class DeploymentPolicyAutomationStore:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> DeploymentPolicyAutomationState:
        if not self.state_file.exists():
            return DeploymentPolicyAutomationState()
        with self.state_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        return DeploymentPolicyAutomationState.model_validate(payload)

    def update(self, request: DeploymentPolicyAutomationUpdateRequest) -> DeploymentPolicyAutomationState:
        state = self.get()
        state.enabled = bool(request.enabled)
        state.interval_seconds = max(60, int(request.interval_seconds or 900))
        state.protect_active_window_seconds = max(30, int(request.protect_active_window_seconds or 120))
        state.gpu_memory_restart_enabled = bool(request.gpu_memory_restart_enabled)
        state.gpu_memory_restart_interval_seconds = max(60, int(request.gpu_memory_restart_interval_seconds or 60))
        state.gpu_memory_restart_threshold_mb = max(1024, int(request.gpu_memory_restart_threshold_mb or 5120))
        self._write(state)
        return state

    def save(self, state: DeploymentPolicyAutomationState) -> DeploymentPolicyAutomationState:
        self._write(state)
        return state

    def record_run(
        self,
        *,
        state: DeploymentPolicyAutomationState,
        last_status: str,
        last_run_at: str | None,
        last_gpu_memory_restart_at: str | None,
        last_policy_generated_at: str | None,
        summary=None,
        actions: list[DeploymentPolicyAutomationAction],
    ) -> DeploymentPolicyAutomationState:
        state.last_status = last_status
        if last_run_at is not None:
            state.last_run_at = last_run_at
            state.last_policy_generated_at = last_policy_generated_at
            state.run_count += 1
            if summary is not None:
                state.last_summary = summary
        if last_gpu_memory_restart_at is not None:
            state.last_gpu_memory_restart_at = last_gpu_memory_restart_at
            state.gpu_memory_restart_run_count += 1
        state.recent_actions = (actions + state.recent_actions)[:20]
        self._write(state)
        return state

    def _write(self, state: DeploymentPolicyAutomationState) -> None:
        with self.state_file.open("w", encoding="utf-8") as handle:
            json.dump(state.model_dump(), handle, ensure_ascii=False, indent=2)
