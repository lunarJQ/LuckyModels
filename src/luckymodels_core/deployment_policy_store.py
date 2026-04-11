from __future__ import annotations

import json
import logging
from pathlib import Path

from .schemas import (
    DeploymentPolicyAutomationAction,
    DeploymentPolicyAutomationState,
    DeploymentPolicyAutomationUpdateRequest,
)

logger = logging.getLogger(__name__)


class DeploymentPolicyAutomationStore:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def get(self) -> DeploymentPolicyAutomationState:
        if not self.state_file.exists():
            return DeploymentPolicyAutomationState()
        raw = self.state_file.read_text(encoding="utf-8").strip()
        if not raw:
            logger.warning("Deployment policy automation state file is empty: %s", self.state_file)
            return DeploymentPolicyAutomationState()
        try:
            payload = json.loads(raw) or {}
        except json.JSONDecodeError:
            logger.exception("Deployment policy automation state file is invalid JSON: %s", self.state_file)
            return DeploymentPolicyAutomationState()
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
        temp_file = self.state_file.with_suffix(f"{self.state_file.suffix}.tmp")
        with temp_file.open("w", encoding="utf-8") as handle:
            json.dump(state.model_dump(), handle, ensure_ascii=False, indent=2)
        temp_file.replace(self.state_file)
